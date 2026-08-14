"""
Generate OOF (out-of-fold) predictions from existing checkpoints.
No retraining needed — just loads each fold's best model and predicts on its val set.

"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score
from pathlib import Path
from PIL import Image
import timm

BASE_DIR = Path(__file__).resolve().parent.parent  # repo root
DATA_DIR = BASE_DIR / "icdar-2026-circleid-pen-classification"

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

MODELS = {
    "v9": {
        "backbone": "convnextv2_tiny.fcmae_ft_in22k_in1k",
        "ckpt_dir": BASE_DIR / "checkpoints_v9",
        "image_size": 336,
        "drop_path": 0.2,
        "head_dropout": 0.3,
    },
    "v8": {
        "backbone": "caformer_s36.sail_in22k_ft_in1k_384",
        "ckpt_dir": BASE_DIR / "checkpoints_v8",
        "image_size": 336,
        "drop_path": 0.2,
        "head_dropout": 0.3,
    },
    "v11": {
        "backbone": "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
        "ckpt_dir": BASE_DIR / "checkpoints_v11",
        "image_size": 336,
        "drop_path": 0.2,
        "head_dropout": 0.3,
    },
    "v14": {
        "backbone": "convnextv2_base.fcmae_ft_in22k_in1k",
        "ckpt_dir": BASE_DIR / "checkpoints_v14_seed42",
        "image_size": 336,
        "drop_path": 0.3,
        "head_dropout": 0.3,
    },
}


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


class PenModel(nn.Module):
    def __init__(self, backbone_name, num_classes=8, drop_path_rate=0.2,
                 head_dropout=0.3, image_size=336):
        super().__init__()
        kwargs = dict(pretrained=False, num_classes=0, drop_path_rate=drop_path_rate)
        if "swin" in backbone_name:
            kwargs["img_size"] = image_size
        self.backbone = timm.create_model(backbone_name, **kwargs)
        embed_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


def generate_oof(model_name, model_cfg):
    print(f"\n{'='*60}")
    print(f"  Generating OOF for {model_name}")
    print(f"  Backbone: {model_cfg['backbone']}")
    print(f"{'='*60}")

    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    EXCLUDE_WRITERS = {"W41", "W50"}
    train_df = train_df[~train_df["writer_id"].isin(EXCLUDE_WRITERS)].reset_index(drop=True)
    pen_ids = sorted(train_df["pen_id"].unique())
    pen2idx = {p: i for i, p in enumerate(pen_ids)}
    num_pens = len(pen_ids)

    # Same fold split as training
    writers = train_df["writer_id"].values
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(train_df, groups=writers))

    val_tfm = transforms.Compose([
        transforms.Resize((model_cfg["image_size"], model_cfg["image_size"]),
                          interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    # Collect OOF predictions for all samples
    oof_logits = np.zeros((len(train_df), num_pens), dtype=np.float64)
    oof_labels = np.zeros(len(train_df), dtype=np.int64)
    oof_mask = np.zeros(len(train_df), dtype=bool)

    for fold in range(5):
        ckpt_path = model_cfg["ckpt_dir"] / f"fold{fold}" / "best_model.pt"
        if not ckpt_path.exists():
            print(f"  Fold {fold}: not found, skipping")
            continue

        ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
        model = PenModel(model_cfg["backbone"], num_pens, model_cfg["drop_path"],
                         model_cfg["head_dropout"], model_cfg["image_size"])

        state = ckpt["model"]
        if "ema_shadow" in ckpt:
            for n in ckpt["ema_shadow"]:
                if n in state:
                    state[n] = ckpt["ema_shadow"][n]
        model.load_state_dict(state)
        model.to("cuda")
        model.eval()

        _, val_indices = folds[fold]
        val_df = train_df.iloc[val_indices]
        val_ds = PenDataset(val_df, DATA_DIR, val_tfm, pen2idx)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

        fold_logits = []
        fold_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to("cuda", non_blocking=True)
                with torch.amp.autocast("cuda"):
                    logits = model(images)
                fold_logits.append(logits.cpu().float().numpy())
                fold_labels.append(labels.numpy())

        fold_logits = np.concatenate(fold_logits, axis=0)
        fold_labels = np.concatenate(fold_labels, axis=0)

        oof_logits[val_indices] = fold_logits
        oof_labels[val_indices] = fold_labels
        oof_mask[val_indices] = True

        f1 = f1_score(fold_labels, fold_logits.argmax(axis=1), average="macro")
        print(f"  Fold {fold}: F1={f1:.4f} (val_f1 in ckpt={ckpt.get('val_f1', 0):.4f})")

        del model
        torch.cuda.empty_cache()

    # Save
    out_dir = model_cfg["ckpt_dir"]
    np.save(out_dir / "oof_logits_all.npy", oof_logits)
    np.save(out_dir / "oof_labels_all.npy", oof_labels)
    np.save(out_dir / "oof_mask_all.npy", oof_mask)

    # Overall OOF score
    valid = oof_mask
    overall_f1 = f1_score(oof_labels[valid], oof_logits[valid].argmax(axis=1), average="macro")
    overall_acc = accuracy_score(oof_labels[valid], oof_logits[valid].argmax(axis=1))
    print(f"\n  {model_name} OOF: F1={overall_f1:.4f}, Acc={overall_acc:.4f} ({valid.sum()} samples)")
    print(f"  Saved to {out_dir}")


def generate_oof_v5b():
    """Generate OOF for v5b — special handling for DINOv2+LoRA+ink features model."""
    print(f"\n{'='*60}")
    print(f"  Generating OOF for v5b (DINOv2 + LoRA + ink features)")
    print(f"{'='*60}")

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_v5b_dinov2 import (
        PenConfig, PenDataset as V5bDataset, build_pen_model,
        _setup_dinov2, set_seed as v5b_set_seed
    )

    _setup_dinov2()
    cfg = PenConfig()
    v5b_set_seed(cfg.seed)

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    EXCLUDE_WRITERS = {"W41", "W50"}
    train_df = train_df[~train_df["writer_id"].isin(EXCLUDE_WRITERS)].reset_index(drop=True)
    pen_ids = sorted(train_df["pen_id"].unique())
    pen2idx = {p: i for i, p in enumerate(pen_ids)}
    num_pens = len(pen_ids)

    # Load full train data (before exclusion) for writer mapping to match training
    full_train_df = pd.read_csv(DATA_DIR / "train.csv")
    all_writers = sorted(full_train_df["writer_id"].unique())
    writer2idx = {w: i for i, w in enumerate(all_writers)}
    num_writers = len(writer2idx)

    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(train_df, groups=train_df["writer_id"].values))

    oof_logits = np.zeros((len(train_df), num_pens), dtype=np.float64)
    oof_labels = np.zeros(len(train_df), dtype=np.int64)
    oof_mask = np.zeros(len(train_df), dtype=bool)

    for fold in range(5):
        ckpt_path = Path(cfg.checkpoint_dir) / f"fold{fold}" / "best_model.pt"
        if not ckpt_path.exists():
            print(f"  Fold {fold}: not found, skipping")
            continue

        _, val_indices = folds[fold]

        val_ds = V5bDataset(
            train_df, list(val_indices), cfg.data_dir, pen2idx, writer2idx,
            cfg.image_size, image_load_mode=cfg.image_load_mode, is_training=False)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                                num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)

        model, _, _ = build_pen_model(cfg, num_writers=num_writers)
        ckpt_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Filter out writer_adversary keys (size mismatch due to writer exclusion)
        model_state = model.state_dict()
        filtered_state = {k: v for k, v in ckpt_state.items()
                         if k in model_state and v.shape == model_state[k].shape}
        model_state.update(filtered_state)
        model.load_state_dict(model_state)
        model.to("cuda")
        model.eval()

        fold_logits = []
        fold_labels = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to("cuda", non_blocking=True)
                masks = batch["mask"].to("cuda", non_blocking=True)
                ink_feats = batch["ink_feats"].to("cuda", non_blocking=True)
                labels = batch["pen_label"]

                with torch.amp.autocast("cuda"):
                    logits = model(images, masks, ink_feats)
                fold_logits.append(logits.cpu().float().numpy())
                fold_labels.append(labels.numpy())

        fold_logits = np.concatenate(fold_logits, axis=0)
        fold_labels = np.concatenate(fold_labels, axis=0)

        oof_logits[val_indices] = fold_logits
        oof_labels[val_indices] = fold_labels
        oof_mask[val_indices] = True

        f1 = f1_score(fold_labels, fold_logits.argmax(axis=1), average="macro")
        print(f"  Fold {fold}: F1={f1:.4f}")

        del model
        torch.cuda.empty_cache()

    out_dir = Path(cfg.checkpoint_dir)
    np.save(out_dir / "oof_logits_all.npy", oof_logits)
    np.save(out_dir / "oof_labels_all.npy", oof_labels)
    np.save(out_dir / "oof_mask_all.npy", oof_mask)

    valid = oof_mask
    overall_f1 = f1_score(oof_labels[valid], oof_logits[valid].argmax(axis=1), average="macro")
    overall_acc = accuracy_score(oof_labels[valid], oof_logits[valid].argmax(axis=1))
    print(f"\n  v5b OOF: F1={overall_f1:.4f}, Acc={overall_acc:.4f} ({valid.sum()} samples)")
    print(f"  Saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="v9, v8, v11, v14, v5b, or all")
    args = parser.parse_args()

    if args.model == "v5b":
        generate_oof_v5b()
    elif args.model == "all":
        for name, cfg in MODELS.items():
            generate_oof(name, cfg)
        generate_oof_v5b()
    elif args.model in MODELS:
        generate_oof(args.model, MODELS[args.model])
    else:
        print(f"Unknown model: {args.model}. Options: {list(MODELS.keys())}, v5b, or 'all'")


if __name__ == "__main__":
    main()
