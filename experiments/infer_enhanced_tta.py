"""
Enhanced TTA Inference — test rotation + multi-scale TTA on existing checkpoints.

Applies to any trained model. Default: v9 ConvNeXt-V2 Tiny.

TTA modes:
  - hflip: horizontal flip only (baseline, what v9 already does)
  - rot4: 4 rotations (0, 90, 180, 270) — circles are rotation-invariant
  - rot4_hflip: 4 rotations x 2 (normal + hflip) = 8 views
  - multiscale: multiple resolutions averaged
  - full: rot4_hflip + multiscale = 8 views x N scales

"""

import argparse
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import timm
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent  # repo root

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ── Model (must match training) ──────────────────────────────────────────

class PenModel(nn.Module):
    def __init__(self, backbone_name, num_classes=8, drop_path_rate=0.2,
                 head_dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, num_classes=0,
            drop_path_rate=drop_path_rate,
        )
        embed_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# ── Dataset ───────────────────────────────────────────────────────────────

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


# ── TTA transforms ───────────────────────────────────────────────────────

class RotateTransform:
    """Picklable rotation transform for Windows multiprocessing."""
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img):
        return img.rotate(-self.angle, expand=False)


def make_transform(size, hflip=False, rotation=0):
    """Build a transform with optional hflip and rotation."""
    tfms = []
    tfms.append(transforms.Resize((size, size),
                                  interpolation=transforms.InterpolationMode.LANCZOS))
    if rotation != 0:
        tfms.append(RotateTransform(rotation))
    if hflip:
        tfms.append(transforms.RandomHorizontalFlip(p=1.0))
    tfms.append(transforms.ToTensor())
    tfms.append(transforms.Normalize(mean=MEAN, std=STD))
    return transforms.Compose(tfms)


def get_tta_transforms(tta_mode, base_size):
    """Returns list of (name, transform) pairs for the TTA mode."""
    if tta_mode == "hflip":
        return [
            ("clean", make_transform(base_size)),
            ("hflip", make_transform(base_size, hflip=True)),
        ]

    elif tta_mode == "rot4":
        views = []
        for rot in [0, 90, 180, 270]:
            views.append((f"rot{rot}", make_transform(base_size, rotation=rot)))
        return views

    elif tta_mode == "rot4_hflip":
        views = []
        for rot in [0, 90, 180, 270]:
            views.append((f"rot{rot}", make_transform(base_size, rotation=rot)))
            views.append((f"rot{rot}_hf", make_transform(base_size, hflip=True, rotation=rot)))
        return views

    elif tta_mode == "multiscale":
        scales = [base_size - 32, base_size, base_size + 32]
        views = []
        for s in scales:
            views.append((f"s{s}", make_transform(s)))
            views.append((f"s{s}_hf", make_transform(s, hflip=True)))
        return views

    elif tta_mode == "full":
        scales = [base_size - 32, base_size, base_size + 32]
        views = []
        for s in scales:
            for rot in [0, 90, 180, 270]:
                views.append((f"s{s}_r{rot}", make_transform(s, rotation=rot)))
                views.append((f"s{s}_r{rot}_hf", make_transform(s, hflip=True, rotation=rot)))
        return views

    else:
        raise ValueError(f"Unknown TTA mode: {tta_mode}")


# ── Inference ─────────────────────────────────────────────────────────────

def run_inference(args):
    data_dir = BASE_DIR / "icdar-2026-circleid-pen-classification"
    test_df = pd.read_csv(data_dir / "test.csv")
    train_df = pd.read_csv(data_dir / "train.csv")

    # Exclude W41/W50 for consistent pen mapping
    EXCLUDE_WRITERS = {"W41", "W50"}
    train_df = train_df[~train_df["writer_id"].isin(EXCLUDE_WRITERS)].reset_index(drop=True)
    pen_ids = sorted(train_df["pen_id"].unique())
    pen2idx = {p: i for i, p in enumerate(pen_ids)}
    idx2pen = {v: k for k, v in pen2idx.items()}
    num_pens = len(pen_ids)
    n_test = len(test_df)

    tta_views = get_tta_transforms(args.tta, args.image_size)
    print(f"TTA mode: {args.tta} ({len(tta_views)} views)")
    print(f"Views: {[name for name, _ in tta_views]}")
    print(f"Backbone: {args.backbone}")
    print(f"Checkpoint dir: {args.ckpt_dir}")
    print(f"Base image size: {args.image_size}")

    all_logits = np.zeros((n_test, num_pens), dtype=np.float64)
    folds_loaded = 0

    for fold in range(5):
        ckpt_path = Path(args.ckpt_dir) / f"fold{fold}" / "best_model.pt"
        if not ckpt_path.exists():
            print(f"  Fold {fold}: not found, skipping")
            continue

        ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
        model = PenModel(args.backbone, num_pens, args.drop_path, args.head_dropout)
        state = ckpt["model"]
        if "ema_shadow" in ckpt:
            for n in ckpt["ema_shadow"]:
                if n in state:
                    state[n] = ckpt["ema_shadow"][n]
        model.load_state_dict(state)
        model.to("cuda")
        model.eval()

        print(f"  Fold {fold}: val_f1={ckpt.get('val_f1', 0):.4f}")

        fold_logits = np.zeros((n_test, num_pens), dtype=np.float64)

        for view_name, view_tfm in tta_views:
            ds = PenTestDataset(test_df, data_dir, view_tfm)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=2, pin_memory=True)
            view_logits = []
            with torch.no_grad():
                for images, _ in loader:
                    images = images.to("cuda", non_blocking=True)
                    with torch.amp.autocast("cuda"):
                        logits = model(images)
                    view_logits.append(logits.cpu().float().numpy())
            fold_logits += np.concatenate(view_logits, axis=0)

        fold_logits /= len(tta_views)
        all_logits += fold_logits
        folds_loaded += 1
        del model
        torch.cuda.empty_cache()

    if folds_loaded == 0:
        print("ERROR: No checkpoints found!")
        return

    all_logits /= folds_loaded
    predictions = np.argmax(all_logits, axis=1)
    pen_preds = [idx2pen[p] for p in predictions]

    out_dir = BASE_DIR / f"outputs_tta_{args.tta}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = Path(args.ckpt_dir).name
    submission = pd.DataFrame({
        "image_id": test_df["image_id"].values,
        "pen_id": pen_preds,
    })
    sub_path = out_dir / f"submission_{model_name}_{args.tta}.csv"
    submission.to_csv(sub_path, index=False)

    logits_path = out_dir / f"test_logits_{model_name}_{args.tta}.npy"
    np.save(logits_path, all_logits)

    print(f"\n  Submission: {sub_path}")
    print(f"  Logits: {logits_path}")
    print(f"  Folds: {folds_loaded}, Views: {len(tta_views)}")
    print(f"\n  Pen distribution:")
    print(submission["pen_id"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tta", type=str, default="rot4_hflip",
                        choices=["hflip", "rot4", "rot4_hflip", "multiscale", "full"],
                        help="TTA mode")
    parser.add_argument("--ckpt_dir", type=str, default=str(BASE_DIR / "checkpoints_v9"),
                        help="Checkpoint directory")
    parser.add_argument("--backbone", type=str, default="convnextv2_tiny.fcmae_ft_in22k_in1k")
    parser.add_argument("--image_size", type=int, default=336)
    parser.add_argument("--drop_path", type=float, default=0.2)
    parser.add_argument("--head_dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    run_inference(args)
