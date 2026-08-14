"""
CircleID — Pen 3 vs 7 Binary Specialist

Trains a binary classifier ONLY on pen 3 vs pen 7 samples at higher resolution.
Used to override the main ensemble when it's uncertain between pen 3 and 7.


"""

import os
import argparse
import warnings
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from PIL import Image
from torchvision import transforms
import timm
from scipy.special import softmax

BASE_DIR = Path(__file__).resolve().parent.parent  # repo root
DATA_DIR = BASE_DIR / "icdar-2026-circleid-pen-classification"


@dataclass
class CFG:
    output_dir: Path = BASE_DIR / "outputs_pen37"
    checkpoint_dir: Path = BASE_DIR / "checkpoints_pen37"
    # -- Backbone --
    backbone: str = "convnextv2_tiny.fcmae_ft_in22k_in1k"
    image_size: int = 448  
    drop_path_rate: float = 0.2
    # -- Head --
    head_dropout: float = 0.3
    # -- Training --
    epochs: int = 40
    batch_size: int = 10 
    grad_accum_steps: int = 6  
    lr_backbone: float = 1.5e-5
    lr_head: float = 5e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 3
    grad_clip: float = 1.0
    label_smoothing: float = 0.05  
    # -- K-Fold --
    n_folds: int = 5
    # -- EMA --
    use_ema: bool = True
    ema_decay: float = 0.999
    ema_start_epoch: int = 3
    # -- Augmentation --
    jitter_brightness: float = 0.10
    jitter_contrast: float = 0.10
    jitter_saturation: float = 0.05
    jitter_hue: float = 0.02
    rotation_deg: float = 15.0
    crop_scale_min: float = 0.75  
    crop_scale_max: float = 1.0
    gaussian_blur_prob: float = 0.3
    gaussian_blur_sigma: tuple = (0.3, 1.0)
    affine_translate: float = 0.05
    affine_scale: tuple = (0.95, 1.05)
    # -- Inference --
    tta_hflip: bool = True
    # -- System --
    num_workers: int = 2
    pin_memory: bool = True
    device: str = "cuda"
    seed: int = 42


EXCLUDE_WRITERS = {"W41", "W50"}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def get_train_transforms(cfg):
    return transforms.Compose([
        transforms.RandomResizedCrop(
            cfg.image_size,
            scale=(cfg.crop_scale_min, cfg.crop_scale_max),
            interpolation=transforms.InterpolationMode.LANCZOS,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(cfg.rotation_deg),
        transforms.ColorJitter(
            brightness=cfg.jitter_brightness,
            contrast=cfg.jitter_contrast,
            saturation=cfg.jitter_saturation,
            hue=cfg.jitter_hue,
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(cfg.affine_translate, cfg.affine_translate),
            scale=cfg.affine_scale,
        ),
        transforms.RandomApply([transforms.GaussianBlur(5, cfg.gaussian_blur_sigma)],
                               p=cfg.gaussian_blur_prob),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def get_val_transforms(cfg):
    return transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size),
                          interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def get_tta_hflip_transforms(cfg):
    return transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size),
                          interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


class BinaryPenDataset(Dataset):
    def __init__(self, df, data_dir, transform):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.transform = transform
        # Binary: pen 3 = 0, pen 7 = 1
        self.label_map = {3: 0, 7: 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.data_dir / row["image_path"]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = self.label_map[row["pen_id"]]
        return image, label


class TestImageDataset(Dataset):
    def __init__(self, df, data_dir, transform):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.data_dir / row["image_path"]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, idx


class PenModel(nn.Module):
    def __init__(self, backbone_name, num_classes=2, drop_path_rate=0.2,
                 head_dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0,
            drop_path_rate=drop_path_rate,
        )
        embed_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(embed_dim, num_classes),
        )
        print(f"  Backbone: {backbone_name} (embed_dim={embed_dim})")
        total = sum(p.numel() for p in self.parameters())
        print(f"  Params: {total:,} (binary classifier)")

    def forward(self, x):
        return self.head(self.backbone(x))


class EMAModel:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def init_shadow(self, model):
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters()
                       if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}


def train_one_fold(fold, train_df, val_df, cfg):
    print(f"\n{'='*60}")
    print(f"  FOLD {fold} — Pen 3 vs 7 Specialist")
    print(f"  Train: {len(train_df)} (pen3={len(train_df[train_df['pen_id']==3])}, pen7={len(train_df[train_df['pen_id']==7])})")
    print(f"  Val: {len(val_df)} (pen3={len(val_df[val_df['pen_id']==3])}, pen7={len(val_df[val_df['pen_id']==7])})")
    print(f"{'='*60}")

    train_ds = BinaryPenDataset(train_df, DATA_DIR, get_train_transforms(cfg))
    val_ds = BinaryPenDataset(val_df, DATA_DIR, get_val_transforms(cfg))

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size * 2, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
    )

    model = PenModel(cfg.backbone, num_classes=2, drop_path_rate=cfg.drop_path_rate,
                     head_dropout=cfg.head_dropout)
    model.to(cfg.device)

    backbone_params = list(model.backbone.parameters())
    head_params = list(model.head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg.lr_backbone},
        {"params": head_params, "lr": cfg.lr_head},
    ], weight_decay=cfg.weight_decay)

    steps_per_epoch = len(train_loader) // cfg.grad_accum_steps
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    scaler = torch.amp.GradScaler("cuda")

    ema = EMAModel(model, cfg.ema_decay) if cfg.use_ema else None

    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    patience_limit = 10
    ckpt_path = cfg.checkpoint_dir / f"fold{fold}"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"E{epoch+1:02d}", leave=False)
        for step, (images, labels) in enumerate(pbar):
            images = images.to(cfg.device, non_blocking=True)
            labels = labels.to(cfg.device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, labels) / cfg.grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % cfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            train_loss += loss.item() * cfg.grad_accum_steps
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            if ema and epoch >= cfg.ema_start_epoch:
                ema.update(model)

            pbar.set_postfix(loss=f"{train_loss/(step+1):.4f}",
                           acc=f"{train_correct/train_total:.3f}")

        if ema and epoch == cfg.ema_start_epoch:
            ema.init_shadow(model)
            print(f"  EMA shadow initialized at epoch {epoch}")

        avg_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # Validate
        if ema and epoch >= cfg.ema_start_epoch:
            ema.apply_shadow(model)

        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(cfg.device, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    logits = model(images)
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_labels.extend(labels.numpy())

        if ema and epoch >= cfg.ema_start_epoch:
            ema.restore(model)

        val_acc = accuracy_score(val_labels, val_preds)
        # Per-class accuracy
        val_preds_arr = np.array(val_preds)
        val_labels_arr = np.array(val_labels)
        pen3_mask = val_labels_arr == 0
        pen7_mask = val_labels_arr == 1
        pen3_acc = (val_preds_arr[pen3_mask] == 0).mean() if pen3_mask.any() else 0
        pen7_acc = (val_preds_arr[pen7_mask] == 1).mean() if pen7_mask.any() else 0

        lr_bb = optimizer.param_groups[0]["lr"]
        print(f"  E{epoch:02d} | loss={avg_loss:.4f} | train_acc={train_acc:.4f} | "
              f"val_acc={val_acc:.4f} | pen3={pen3_acc:.3f} pen7={pen7_acc:.3f} | "
              f"lr_bb={lr_bb:.2e}"
              f"{' *' if val_acc > best_acc else ''}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            save_dict = {
                "epoch": epoch,
                "model": model.state_dict(),
                "val_acc": val_acc,
                "pen3_acc": pen3_acc,
                "pen7_acc": pen7_acc,
            }
            if ema and epoch >= cfg.ema_start_epoch:
                save_dict["ema_shadow"] = ema.shadow
            torch.save(save_dict, ckpt_path / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"\n  Fold {fold} best: acc={best_acc:.4f} at epoch {best_epoch}")

    del model, optimizer, scheduler, scaler, ema, train_loader, val_loader
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return best_acc


def run_inference_on_uncertain(cfg):
   
    print(f"\n{'='*60}")
    print("  PEN 3/7 SPECIALIST INFERENCE")
    print(f"{'='*60}")

    # Load ensemble logits
    logit_files = {
        "v9": BASE_DIR / "outputs_v9" / "test_logits_v9.npy",
        "v8": BASE_DIR / "outputs_v8" / "test_logits_v8.npy",
        "v5b": BASE_DIR / "outputs_v5b" / "test_logits_v5b.npy",
    }
    weights = {"v9": 0.55, "v8": 0.30, "v5b": 0.15}

    blended = np.zeros((5905, 8), dtype=np.float64)
    for name, path in logit_files.items():
        if path.exists():
            logits = np.load(path)
            probs = softmax(logits, axis=1)
            blended += weights[name] * probs

    # Find samples where pen 3 and 7 are top-2
    # pen 3 = index 2, pen 7 = index 6
    pen3_prob = blended[:, 2]
    pen7_prob = blended[:, 6]
    ensemble_pred = blended.argmax(axis=1)

    # Samples where prediction is pen 3 or 7
    pen37_mask = (ensemble_pred == 2) | (ensemble_pred == 6)
    pen37_indices = np.where(pen37_mask)[0]
    print(f"  Ensemble predicts pen 3 or 7: {len(pen37_indices)} samples")

    # Load test data
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    val_tfm = get_val_transforms(cfg)
    hflip_tfm = get_tta_hflip_transforms(cfg) if cfg.tta_hflip else None

    # Run specialist on ALL pen 3/7 predicted samples
    subset_df = test_df.iloc[pen37_indices].reset_index(drop=True)
    all_logits = np.zeros((len(subset_df), 2), dtype=np.float64)
    folds_loaded = 0

    for fold in range(cfg.n_folds):
        ckpt_path = cfg.checkpoint_dir / f"fold{fold}" / "best_model.pt"
        if not ckpt_path.exists():
            print(f"  Fold {fold}: checkpoint not found, skipping")
            continue

        ckpt = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
        model = PenModel(cfg.backbone, num_classes=2, drop_path_rate=cfg.drop_path_rate,
                         head_dropout=cfg.head_dropout)

        state = ckpt["model"]
        if "ema_shadow" in ckpt:
            for n in ckpt["ema_shadow"]:
                if n in state:
                    state[n] = ckpt["ema_shadow"][n]
        model.load_state_dict(state)
        model.to(cfg.device)
        model.eval()

        print(f"  Fold {fold}: loaded (val_acc={ckpt.get('val_acc', 0):.4f}, "
              f"pen3={ckpt.get('pen3_acc', 0):.3f}, pen7={ckpt.get('pen7_acc', 0):.3f})")

        # Clean pass
        test_ds = TestImageDataset(subset_df, DATA_DIR, val_tfm)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size * 2, shuffle=False,
                                 num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
        fold_logits = []
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(cfg.device, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    logits = model(images)
                fold_logits.append(logits.cpu().float().numpy())
        fold_logits = np.concatenate(fold_logits, axis=0)

        # H-flip TTA
        if cfg.tta_hflip:
            test_ds_hf = TestImageDataset(subset_df, DATA_DIR, hflip_tfm)
            test_loader_hf = DataLoader(test_ds_hf, batch_size=cfg.batch_size * 2,
                                        shuffle=False, num_workers=cfg.num_workers,
                                        pin_memory=cfg.pin_memory)
            hflip_logits = []
            with torch.no_grad():
                for images, _ in test_loader_hf:
                    images = images.to(cfg.device, non_blocking=True)
                    with torch.amp.autocast("cuda"):
                        logits = model(images)
                    hflip_logits.append(logits.cpu().float().numpy())
            hflip_logits = np.concatenate(hflip_logits, axis=0)
            fold_logits = (fold_logits + hflip_logits) / 2.0

        all_logits += fold_logits
        folds_loaded += 1
        del model
        torch.cuda.empty_cache()

    if folds_loaded == 0:
        print("  ERROR: No specialist checkpoints found!")
        return

    all_logits /= folds_loaded
    specialist_probs = softmax(all_logits, axis=1)
    specialist_pred = all_logits.argmax(axis=1)  # 0=pen3, 1=pen7

    # Map back: 0->pen3(idx2), 1->pen7(idx6)
    pen_map = {0: 2, 1: 6}

    # Save specialist predictions
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(cfg.output_dir / "specialist_logits.npy", all_logits)
    np.save(cfg.output_dir / "specialist_indices.npy", pen37_indices)

    # Stats
    spec_pen3 = (specialist_pred == 0).sum()
    spec_pen7 = (specialist_pred == 1).sum()
    ens_pen3 = (ensemble_pred[pen37_indices] == 2).sum()
    ens_pen7 = (ensemble_pred[pen37_indices] == 6).sum()
    flipped = (specialist_pred != np.array([0 if ensemble_pred[i] == 2 else 1 for i in pen37_indices])).sum()

    print(f"\n  Specialist results on {len(pen37_indices)} samples:")
    print(f"    Ensemble:   pen3={ens_pen3}, pen7={ens_pen7}")
    print(f"    Specialist: pen3={spec_pen3}, pen7={spec_pen7}")
    print(f"    Flipped: {flipped} predictions changed")

    # Confidence of specialist
    spec_conf = specialist_probs.max(axis=1)
    print(f"    Specialist confidence: mean={spec_conf.mean():.3f}, min={spec_conf.min():.3f}")


def apply_specialist(cfg, gap_threshold=0.30):
    """Apply specialist predictions to override ensemble on uncertain pen 3/7 samples."""
    print(f"\n{'='*60}")
    print(f"  APPLYING SPECIALIST (gap_threshold={gap_threshold})")
    print(f"{'='*60}")

    # Load ensemble logits
    logit_files = {
        "v9": BASE_DIR / "outputs_v9" / "test_logits_v9.npy",
        "v8": BASE_DIR / "outputs_v8" / "test_logits_v8.npy",
        "v5b": BASE_DIR / "outputs_v5b" / "test_logits_v5b.npy",
    }
    weights = {"v9": 0.55, "v8": 0.30, "v5b": 0.15}

    blended = np.zeros((5905, 8), dtype=np.float64)
    for name, path in logit_files.items():
        if path.exists():
            logits = np.load(path)
            probs = softmax(logits, axis=1)
            blended += weights[name] * probs

    # Load specialist data
    spec_logits = np.load(cfg.output_dir / "specialist_logits.npy")
    spec_indices = np.load(cfg.output_dir / "specialist_indices.npy")
    spec_probs = softmax(spec_logits, axis=1)

    # Load test/train for pen mapping
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    pen_list = sorted(train_df["pen_id"].unique())

    ensemble_pred = blended.argmax(axis=1)
    pen3_prob = blended[:, 2]
    pen7_prob = blended[:, 6]
    gap = np.abs(pen3_prob - pen7_prob)

    
    overrides = 0
    for i, test_idx in enumerate(spec_indices):
        if gap[test_idx] < gap_threshold:
            spec_pred = 2 if spec_logits[i, 0] > spec_logits[i, 1] else 6
            if spec_pred != ensemble_pred[test_idx]:
                ensemble_pred[test_idx] = spec_pred
                overrides += 1

    pen_preds = [pen_list[p] for p in ensemble_pred]
    sub = pd.DataFrame({"image_id": test_df["image_id"], "pen_id": pen_preds})

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"ensemble_specialist_gap{int(gap_threshold*100)}"
    out_path = cfg.output_dir / f"submission_{tag}.csv"
    sub.to_csv(out_path, index=False)

    print(f"  Overrides applied: {overrides} (gap < {gap_threshold})")
    print(f"  Submission saved: {out_path}")
    print(f"  Pen distribution:")
    print(sub["pen_id"].value_counts().sort_index().to_string())

    # Also generate at different thresholds
    for thresh in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1.0]:
        pred = blended.argmax(axis=1).copy()
        n_override = 0
        for i, test_idx in enumerate(spec_indices):
            if gap[test_idx] < thresh:
                spec_pred = 2 if spec_logits[i, 0] > spec_logits[i, 1] else 6
                if spec_pred != pred[test_idx]:
                    pred[test_idx] = spec_pred
                    n_override += 1
        preds = [pen_list[p] for p in pred]
        s = pd.DataFrame({"image_id": test_df["image_id"], "pen_id": preds})
        t = f"ensemble_specialist_gap{int(thresh*100)}"
        s.to_csv(cfg.output_dir / f"submission_{t}.csv", index=False)
        print(f"  gap<{thresh:.2f}: {n_override} overrides -> {cfg.output_dir / f'submission_{t}.csv'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--infer-only", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--gap", type=float, default=0.30)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    cfg = CFG()
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size

    set_seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.apply:
        apply_specialist(cfg, args.gap)
        return

    if args.infer_only:
        run_inference_on_uncertain(cfg)
        return

    print(f"CircleID Pen 3/7 Specialist")
    print(f"  Backbone: {cfg.backbone}")
    print(f"  Image size: {cfg.image_size} (higher res for ink texture)")
    print(f"  Batch size: {cfg.batch_size} x {cfg.grad_accum_steps} accum = {cfg.batch_size * cfg.grad_accum_steps}")
    print(f"  LR backbone: {cfg.lr_backbone}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load pen 3 and 7 data only
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    train_df = train_df[~train_df["writer_id"].isin(EXCLUDE_WRITERS)]
    train_df = train_df[train_df["pen_id"].isin([3, 7])].reset_index(drop=True)

    add_csv = DATA_DIR / "additional_train.csv"
    additional_df = None
    if add_csv.exists():
        additional_df = pd.read_csv(add_csv)
        additional_df = additional_df[~additional_df["writer_id"].isin(EXCLUDE_WRITERS)]
        additional_df = additional_df[additional_df["pen_id"].isin([3, 7])].reset_index(drop=True)
        print(f"  Additional data: {len(additional_df)} pen 3/7 images")

    print(f"  Train data: {len(train_df)} pen 3/7 images")
    print(f"    Pen 3: {(train_df['pen_id']==3).sum()}, Pen 7: {(train_df['pen_id']==7).sum()}")

    # K-Fold on writers
    writers = train_df["writer_id"].values
    gkf = GroupKFold(n_splits=cfg.n_folds)
    folds = list(gkf.split(train_df, groups=writers))

    fold_accs = []
    folds_to_run = [args.fold] if args.fold is not None else range(cfg.n_folds)

    for fold_idx in folds_to_run:
        train_indices, val_indices = folds[fold_idx]
        fold_train_df = train_df.iloc[train_indices].copy()
        fold_val_df = train_df.iloc[val_indices].copy()

        if additional_df is not None:
            fold_train_df = pd.concat([fold_train_df, additional_df], ignore_index=True)

        acc = train_one_fold(fold_idx, fold_train_df, fold_val_df, cfg)
        fold_accs.append(acc)

    if len(fold_accs) > 1:
        print(f"\n{'='*60}")
        print(f"  ALL FOLDS COMPLETE")
        for i, acc in enumerate(fold_accs):
            print(f"    Fold {i}: acc={acc:.4f}")
        print(f"    Mean acc: {np.mean(fold_accs):.4f} +/- {np.std(fold_accs):.4f}")
        print(f"{'='*60}")

    # Run inference
    run_inference_on_uncertain(cfg)


if __name__ == "__main__":
    main()
