"""
CircleID — Pen Classification: ConvNeXt-V2 Base

For Kaggle notebooks with T4 (16GB) or P100 (16GB).
ConvNeXt-V2 Base (88M params) — pure CNN with strong texture bias.

Key differences from v7 (local):
1. ConvNeXt-V2 Base (88M) vs EfficientNetV2-M (53M) — more capacity
2. Projection head (1024→512→8) vs trivial linear head
3. Backbone LR 5e-5 (not 1e-5) — proper full finetune
4. No RandomErasing — preserves thin ink strokes
5. EMA init order fixed
6. Crop scale 0.75 (not 0.7) — gentler on ~100px images
7. 50 epochs (not 40) — larger model needs more training

"""

import os
import sys
import argparse
import warnings
import time
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

# ── Paths: auto-detect Kaggle vs local ──────────────────────────────────
if Path("/kaggle/input").exists():
    DATA_DIR = Path("/kaggle/input/icdar-2026-circleid-pen-classification")
    BASE_DIR = Path("/kaggle/working")
else:
    BASE_DIR = Path(__file__).resolve().parent.parent  # repo root
    DATA_DIR = BASE_DIR / "icdar-2026-circleid-pen-classification"


@dataclass
class CFG:
    data_dir: Path = DATA_DIR
    train_csv: str = "train.csv"
    test_csv: str = "test.csv"
    additional_csv: str = "additional_train.csv"
    output_dir: Path = BASE_DIR / "outputs_convnext_base"
    checkpoint_dir: Path = BASE_DIR / "checkpoints_convnext_base"
    num_pens: int = 8
    # -- Backbone --
    backbone: str = "convnextv2_base.fcmae_ft_in22k_in1k"
    image_size: int = 336
    drop_path_rate: float = 0.3  
    # -- Head --
    proj_dim: int = 512
    head_dropout: float = 0.3
    # -- Training --
    epochs: int = 50
    batch_size: int = 16
    grad_accum_steps: int = 4  
    lr_backbone: float = 5e-5  
    lr_head: float = 5e-4
    weight_decay: float = 0.02
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    label_smoothing: float = 0.1
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
    # -- Data --
    use_additional_data: bool = True
    # -- System --
    num_workers: int = 2
    pin_memory: bool = True
    device: str = "cuda"
    seed: int = 42


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ══════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


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


class PenDataset(Dataset):
    def __init__(self, df, data_dir, transform, pen2idx):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.pen2idx = pen2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.data_dir / row["image_path"]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = self.pen2idx[row["pen_id"]]
        return image, label


class PenTestDataset(Dataset):
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
        return image, row["image_id"]


# ══════════════════════════════════════════════════════════════════════════
# Model — with projection head (not trivial linear)
# ══════════════════════════════════════════════════════════════════════════

class PenModel(nn.Module):
    def __init__(self, backbone_name, num_classes=8, drop_path_rate=0.3,
                 proj_dim=512, head_dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )
        embed_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(proj_dim, num_classes),
        )
        print(f"  Backbone: {backbone_name} (embed_dim={embed_dim})")
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Params: {trainable:,} trainable / {total:,} total")
        print(f"  Head: {embed_dim} -> {proj_dim} -> {num_classes}")

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


# ══════════════════════════════════════════════════════════════════════════
# EMA (fixed init order)
# ══════════════════════════════════════════════════════════════════════════

class EMAModel:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.initialized = False

    def init_shadow(self, model):
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters()
                       if p.requires_grad}
        self.initialized = True

    @torch.no_grad()
    def update(self, model):
        if not self.initialized:
            return
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        if not self.initialized:
            return
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


# ══════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════

def train_one_fold(fold, train_df, val_df, cfg, pen2idx):
    print(f"\n{'='*60}")
    print(f"  FOLD {fold}")
    print(f"  Train: {len(train_df)} | Val: {len(val_df)}")
    print(f"{'='*60}")

    train_ds = PenDataset(train_df, cfg.data_dir, get_train_transforms(cfg), pen2idx)
    val_ds = PenDataset(val_df, cfg.data_dir, get_val_transforms(cfg), pen2idx)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size * 2, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
    )

    model = PenModel(cfg.backbone, cfg.num_pens, cfg.drop_path_rate,
                      cfg.proj_dim, cfg.head_dropout)
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

    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    patience_limit = 12
    ckpt_path = cfg.checkpoint_dir / f"fold{fold}"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        # ── EMA init (BEFORE training loop — fixed order) ──
        if ema and epoch == cfg.ema_start_epoch:
            ema.init_shadow(model)
            print(f"  EMA shadow initialized at epoch {epoch}")

        # ── Train ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        optimizer.zero_grad()

        for step, (images, labels) in enumerate(train_loader):
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

                # EMA update (inside optimizer step — correct frequency)
                if ema and epoch >= cfg.ema_start_epoch:
                    ema.update(model)

            train_loss += loss.item() * cfg.grad_accum_steps
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        avg_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # ── Validate ──
        if ema and ema.initialized:
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

        if ema and ema.initialized:
            ema.restore(model)

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="macro")
        lr_bb = optimizer.param_groups[0]["lr"]
        lr_hd = optimizer.param_groups[1]["lr"]

        print(f"  E{epoch:02d} | loss={avg_loss:.4f} | train_acc={train_acc:.4f} | "
              f"val_acc={val_acc:.4f} | val_F1={val_f1:.4f} | "
              f"lr_bb={lr_bb:.2e} lr_hd={lr_hd:.2e}"
              f"{' *' if val_f1 > best_f1 else ''}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            save_dict = {
                "epoch": epoch,
                "model": model.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
                "cfg_backbone": cfg.backbone,
            }
            if ema and ema.initialized:
                save_dict["ema_shadow"] = ema.shadow
            torch.save(save_dict, ckpt_path / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"  Early stopping at epoch {epoch} "
                      f"(no improvement for {patience_limit} epochs)")
                break

    print(f"\n  Fold {fold} best: F1={best_f1:.4f} at epoch {best_epoch}")

    # ── Print per-pen confusion ──
    best_ckpt = torch.load(ckpt_path / "best_model.pt", map_location=cfg.device,
                           weights_only=False)
    state = best_ckpt["model"]
    if "ema_shadow" in best_ckpt:
        for n in best_ckpt["ema_shadow"]:
            if n in state:
                state[n] = best_ckpt["ema_shadow"][n]
    model.load_state_dict(state)
    model.eval()

    val_preds = []
    val_labels_all = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(cfg.device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                logits = model(images)
            val_preds.extend(logits.argmax(dim=1).cpu().numpy())
            val_labels_all.extend(labels.numpy())

    cm = confusion_matrix(val_labels_all, val_preds)
    idx2pen = {v: k for k, v in pen2idx.items()}
    print(f"\n  Per-pen accuracy (fold {fold}):")
    for i in range(cfg.num_pens):
        row_sum = cm[i].sum()
        acc = cm[i, i] / row_sum if row_sum > 0 else 0
        print(f"    Pen {idx2pen[i]}: {acc:.3f} ({cm[i,i]}/{row_sum})")

    del model
    torch.cuda.empty_cache()
    return best_f1


# ══════════════════════════════════════════════════════════════════════════
# Inference
# ══════════════════════════════════════════════════════════════════════════

def run_inference(cfg, pen2idx):
    print(f"\n{'='*60}")
    print("  INFERENCE")
    print(f"{'='*60}")

    test_df = pd.read_csv(cfg.data_dir / cfg.test_csv)
    idx2pen = {v: k for k, v in pen2idx.items()}
    n_test = len(test_df)
    print(f"  Test samples: {n_test}")

    val_tfm = get_val_transforms(cfg)
    hflip_tfm = get_tta_hflip_transforms(cfg) if cfg.tta_hflip else None

    all_logits = np.zeros((n_test, cfg.num_pens), dtype=np.float64)
    folds_loaded = 0

    for fold in range(cfg.n_folds):
        ckpt_path = cfg.checkpoint_dir / f"fold{fold}" / "best_model.pt"
        if not ckpt_path.exists():
            print(f"  Fold {fold}: checkpoint not found, skipping")
            continue

        ckpt = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
        model = PenModel(cfg.backbone, cfg.num_pens, cfg.drop_path_rate,
                          cfg.proj_dim, cfg.head_dropout)

        state = ckpt["model"]
        if "ema_shadow" in ckpt:
            for n in ckpt["ema_shadow"]:
                if n in state:
                    state[n] = ckpt["ema_shadow"][n]
        model.load_state_dict(state)
        model.to(cfg.device)
        model.eval()

        print(f"  Fold {fold}: loaded (val_f1={ckpt.get('val_f1', 0):.4f}, "
              f"epoch={ckpt.get('epoch', '?')})")

        # Clean pass
        test_ds = PenTestDataset(test_df, cfg.data_dir, val_tfm)
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
            test_ds_hf = PenTestDataset(test_df, cfg.data_dir, hflip_tfm)
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
        print("  ERROR: No checkpoints found!")
        return

    all_logits /= folds_loaded
    predictions = np.argmax(all_logits, axis=1)
    pen_preds = [idx2pen[p] for p in predictions]

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    submission = pd.DataFrame({
        "image_id": test_df["image_id"].values,
        "pen_id": pen_preds,
    })
    sub_path = cfg.output_dir / "submission_convnext_base.csv"
    submission.to_csv(sub_path, index=False)
    print(f"\n  Submission saved: {sub_path}")
    print(f"  Folds ensembled: {folds_loaded}")
    print(f"  TTA: H-flip={'yes' if cfg.tta_hflip else 'no'}")
    print(f"\n  Pen distribution:")
    print(submission["pen_id"].value_counts().sort_index().to_string())

    # Save logits for ensemble with local models
    logits_path = cfg.output_dir / "test_logits_convnext_base.npy"
    np.save(logits_path, all_logits)
    print(f"  Logits saved: {logits_path}")
    print(f"\n  >> Download this file for local ensemble:")
    print(f"  >> {logits_path}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--infer-only", action="store_true")
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr_backbone", type=float, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--no-additional", action="store_true")
    args = parser.parse_args()

    cfg = CFG()
    if args.backbone:
        cfg.backbone = args.backbone
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr_backbone:
        cfg.lr_backbone = args.lr_backbone
    if args.image_size:
        cfg.image_size = args.image_size
    if args.no_additional:
        cfg.use_additional_data = False

    set_seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"CircleID Pen — ConvNeXt-V2 Base (Kaggle)")
    print(f"  Backbone: {cfg.backbone}")
    print(f"  Image size: {cfg.image_size}")
    print(f"  Batch: {cfg.batch_size} x {cfg.grad_accum_steps} = "
          f"{cfg.batch_size * cfg.grad_accum_steps}")
    print(f"  LR backbone: {cfg.lr_backbone}, LR head: {cfg.lr_head}")
    print(f"  Epochs: {cfg.epochs}, Patience: 12")
    print(f"  Crop scale: ({cfg.crop_scale_min}, {cfg.crop_scale_max})")
    print(f"  Device: {cfg.device}")
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  GPU: {gpu} ({mem:.1f} GB)")

    # ── Load data ──
    train_df = pd.read_csv(cfg.data_dir / cfg.train_csv)

    # Exclude writers with known pen-ID annotation errors 
    EXCLUDE_WRITERS = {"W41", "W50"}
    n_before = len(train_df)
    train_df = train_df[~train_df["writer_id"].isin(EXCLUDE_WRITERS)].reset_index(drop=True)
    print(f"\n  Excluded writers {EXCLUDE_WRITERS}: {n_before} -> {len(train_df)} (-{n_before - len(train_df)})")
    print(f"  Train data: {len(train_df)} images")

    pen_ids = sorted(train_df["pen_id"].unique())
    pen2idx = {p: i for i, p in enumerate(pen_ids)}
    print(f"  Pens: {pen_ids}")

    additional_df = None
    if cfg.use_additional_data:
        add_csv = cfg.data_dir / cfg.additional_csv
        if add_csv.exists():
            additional_df = pd.read_csv(add_csv)
            additional_df = additional_df[additional_df["pen_id"].isin(pen_ids)].copy()
            additional_df = additional_df[~additional_df["writer_id"].isin(EXCLUDE_WRITERS)].reset_index(drop=True)
            print(f"  Additional data: {len(additional_df)} images (W41/W50 excluded)")
        else:
            print(f"  Additional data: not found")

    if args.infer_only:
        run_inference(cfg, pen2idx)
        return

    # ── K-Fold ──
    writers = train_df["writer_id"].values
    gkf = GroupKFold(n_splits=cfg.n_folds)
    folds = list(gkf.split(train_df, groups=writers))

    fold_f1s = []
    folds_to_run = [args.fold] if args.fold is not None else range(cfg.n_folds)

    for fold_idx in folds_to_run:
        train_indices, val_indices = folds[fold_idx]
        fold_train_df = train_df.iloc[train_indices].copy()
        fold_val_df = train_df.iloc[val_indices].copy()

        if additional_df is not None:
            fold_train_df = pd.concat([fold_train_df, additional_df], ignore_index=True)
            print(f"  Fold {fold_idx}: {len(fold_train_df)} train "
                  f"(+{len(additional_df)} additional)")

        f1 = train_one_fold(fold_idx, fold_train_df, fold_val_df, cfg, pen2idx)
        fold_f1s.append(f1)

    if len(fold_f1s) > 1:
        print(f"\n{'='*60}")
        print(f"  ALL FOLDS COMPLETE")
        for i, f1 in enumerate(fold_f1s):
            print(f"    Fold {i}: F1={f1:.4f}")
        print(f"    Mean F1: {np.mean(fold_f1s):.4f} +/- {np.std(fold_f1s):.4f}")
        print(f"{'='*60}")

    run_inference(cfg, pen2idx)


if __name__ == "__main__":
    main()
