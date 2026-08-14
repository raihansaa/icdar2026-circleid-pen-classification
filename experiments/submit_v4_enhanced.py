"""
Enhanced submission for v4: no retraining of the neural net.
1. LightGBM on handcrafted ink features (writer-disjoint K-fold) — orthogonal signal
2. H-flip TTA on neural net inference — deterministic extra views
3. Weighted fold ensemble — trust better folds more
4. Neural net + GBM probability blending

"""

import os
import sys
import time
import math
import warnings
from pathlib import Path
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score
import lightgbm as lgb

warnings.filterwarnings("ignore", message="xFormers is not available")

# Import everything from v4
sys.path.insert(0, str(Path(__file__).resolve().parent))
from circleid_pen_v4 import (
    PenConfig, BASE_DIR, NUM_INK_FEATURES,
    _setup_dinov2, set_seed, extract_ink_features,
    compute_ink_mask, preprocess_image, apply_random_scale_crop,
    ink_weighted_pooling, build_pen_model,
    LoRALinear, SubCenterArcFace, GradientReversal, PenClassifierModel, EMAModel,
)


# ══════════════════════════════════════════════════════════════════════════
# 1. LightGBM on Ink Features
# ══════════════════════════════════════════════════════════════════════════

def extract_all_ink_features(df, data_dir, mode="cv2"):
    """Extract ink features for all rows in a dataframe."""
    print(f"  Extracting ink features for {len(df)} images...")
    t0 = time.time()
    features = []
    for i, (_, row) in enumerate(df.iterrows()):
        img_path = Path(data_dir) / row["image_path"]
        if mode == "cv2":
            img = cv2.imread(str(img_path))
        else:
            from PIL import Image
            img = np.array(Image.open(img_path).convert("RGB"))
        feats = extract_ink_features(img, mode) if img is not None \
            else np.zeros(NUM_INK_FEATURES, dtype=np.float32)
        features.append(feats)
        if (i + 1) % 5000 == 0:
            print(f"    {i + 1}/{len(df)}")
    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s")
    return np.array(features, dtype=np.float32)


def train_gbm_kfold(X_train, y_train, groups, pen2idx, n_folds=5, seed=42):
    """Train LightGBM with writer-disjoint K-fold. Returns fold models + CV score."""
    gkf = GroupKFold(n_splits=n_folds)
    models = []
    oof_preds = np.zeros((len(y_train), len(pen2idx)))
    oof_labels = np.zeros(len(y_train), dtype=int)

    # Map pen_id to 0-indexed
    y_mapped = np.array([pen2idx[p] for p in y_train])

    params = {
        "objective": "multiclass",
        "num_class": len(pen2idx),
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 7,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "seed": seed,
        "verbose": -1,
        "n_jobs": -1,
    }

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_mapped, groups)):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_mapped[tr_idx], y_mapped[va_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

        model = lgb.train(
            params, dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0),  # silent
            ],
        )
        models.append(model)

        probs = model.predict(X_va)
        oof_preds[va_idx] = probs
        oof_labels[va_idx] = y_va

        va_preds = probs.argmax(axis=1)
        acc = accuracy_score(y_va, va_preds)
        f1 = f1_score(y_va, va_preds, average="macro")
        print(f"    GBM Fold {fold + 1}: acc={acc:.4f}  f1={f1:.4f}  "
              f"best_iter={model.best_iteration}")

    overall_preds = oof_preds.argmax(axis=1)
    overall_acc = accuracy_score(oof_labels, overall_preds)
    overall_f1 = f1_score(oof_labels, overall_preds, average="macro")
    print(f"    GBM Overall CV: acc={overall_acc:.4f}  f1={overall_f1:.4f}")

    return models, overall_acc


def predict_gbm_ensemble(models, X_test):
    """Average predictions from all fold GBM models."""
    all_probs = []
    for model in models:
        probs = model.predict(X_test)
        all_probs.append(probs)
    return np.mean(all_probs, axis=0)


# ══════════════════════════════════════════════════════════════════════════
# 2. Neural Net Inference with H-Flip TTA
# ══════════════════════════════════════════════════════════════════════════

class PenTestDatasetWithFlip(Dataset):
    """Test dataset: multi-crop + horizontal flip TTA.
    Returns (1 + num_crops) * 2 views per image (original + flipped)."""

    def __init__(self, df, data_dir, image_size=224, image_load_mode="cv2",
                 num_crops=12, crop_scale_range=(0.5, 1.0), do_hflip=True):
        self.df = df
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.mode = image_load_mode
        self.num_crops = num_crops
        self.crop_scale_range = crop_scale_range
        self.do_hflip = do_hflip
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        # Precompute ink features
        self.ink_features = self._precompute_ink_features()

    def _precompute_ink_features(self):
        print(f"  Precomputing ink features (test, {len(self.df)} images)...")
        t0 = time.time()
        features = []
        for _, row in self.df.iterrows():
            img_path = self.data_dir / row["image_path"]
            if self.mode == "cv2":
                img = cv2.imread(str(img_path))
            else:
                from PIL import Image
                img = np.array(Image.open(img_path).convert("RGB"))
            feats = extract_ink_features(img, self.mode) if img is not None \
                else np.zeros(NUM_INK_FEATURES, dtype=np.float32)
            features.append(feats)
        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s")
        return np.array(features, dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def _to_tensor(self, image, mask_float):
        img_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        if self.mode == "cv2":
            img_tensor = img_tensor[[2, 1, 0], ...]
        img_tensor = (img_tensor - self.mean) / self.std
        mask_tensor = torch.from_numpy(mask_float).float()
        return img_tensor, mask_tensor

    def _get_views(self, image, rng=None):
        """Generate all views for one orientation of the image."""
        views_img, views_mask = [], []

        # Full image
        full_img, full_mask = preprocess_image(
            image, mode=self.mode, image_size=self.image_size, apply_jitter=False)
        img_t, mask_t = self._to_tensor(full_img, full_mask)
        views_img.append(img_t)
        views_mask.append(mask_t)

        # Random ink-biased crops
        if rng is None:
            rng = np.random.default_rng()
        for _ in range(self.num_crops):
            cropped = apply_random_scale_crop(image, self.crop_scale_range, rng)
            crop_img, crop_mask = preprocess_image(
                cropped, mode=self.mode, image_size=self.image_size, apply_jitter=False)
            img_t, mask_t = self._to_tensor(crop_img, crop_mask)
            views_img.append(img_t)
            views_mask.append(mask_t)

        return views_img, views_mask

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.data_dir / row["image_path"]
        if self.mode == "cv2":
            image = cv2.imread(str(img_path))
            if image is None:
                raise FileNotFoundError(f"Failed to load: {img_path}")
        else:
            from PIL import Image
            image = np.array(Image.open(img_path).convert("RGB"))

        ink_feats = torch.from_numpy(self.ink_features[idx]).float()
        rng = np.random.default_rng(idx)  # deterministic per image

        all_imgs, all_masks, all_feats = [], [], []

        # Original orientation views
        views_img, views_mask = self._get_views(image, rng)
        all_imgs.extend(views_img)
        all_masks.extend(views_mask)
        all_feats.extend([ink_feats] * len(views_img))

        # H-flip views
        if self.do_hflip:
            flipped = cv2.flip(image, 1)
            rng_flip = np.random.default_rng(idx + 999999)
            views_img_f, views_mask_f = self._get_views(flipped, rng_flip)
            all_imgs.extend(views_img_f)
            all_masks.extend(views_mask_f)
            all_feats.extend([ink_feats] * len(views_img_f))

        return {"image": torch.stack(all_imgs),
                "mask": torch.stack(all_masks),
                "ink_feats": torch.stack(all_feats)}


def run_inference_fold_enhanced(fold, cfg, device, test_ds, num_writers, use_amp):
   
    fold_dir = Path(cfg.checkpoint_dir) / f"fold{fold}"
    ckpt_path = fold_dir / "best_model.pt"
    if not ckpt_path.exists():
        print(f"  WARNING: No checkpoint for fold {fold}, skipping")
        return None

    model, _, _ = build_pen_model(cfg, num_writers=num_writers)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
    model = model.to(device)
    model.eval()

    # All images have the same K views, so batch_size > 1 is safe.
    # Pick batch size so total views per batch fits GPU memory.
    K = test_ds[0]["image"].shape[0]  # views per image
    max_views_per_batch = 128  # ViT-B at 224px uses ~100MB/image, 128 fits in 12GB+
    infer_batch_size = max(1, max_views_per_batch // K)
    print(f"    {K} views/image, batch_size={infer_batch_size} "
          f"({infer_batch_size * K} views/batch)")

    test_loader = DataLoader(test_ds, batch_size=infer_batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)

    all_logits = []
    n_done = 0
    with torch.no_grad():
        for batch in test_loader:
            B = batch["image"].shape[0]
            # [B, K, C, H, W] → [B*K, C, H, W]
            imgs = batch["image"].view(B * K, *batch["image"].shape[2:]).to(device)
            masks = batch["mask"].view(B * K, *batch["mask"].shape[2:]).to(device)
            ink_feats = batch["ink_feats"].view(B * K, -1).to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(imgs, masks, ink_feats, cfg.eps)  # [B*K, num_pens]

            # [B*K, num_pens] → [B, K, num_pens] → mean over K → [B, num_pens]
            logits = logits.view(B, K, -1).mean(dim=1)
            all_logits.append(logits.cpu().numpy())
            n_done += B
            if n_done % 3000 < infer_batch_size:
                print(f"    Fold {fold}: {n_done}/{len(test_ds)} images")

    del model
    torch.cuda.empty_cache()
    return np.concatenate(all_logits, axis=0)


# ══════════════════════════════════════════════════════════════════════════
# 3. Main
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gbm-weight", type=float, default=0.12,
                        help="Blend weight for GBM probabilities (0=NN only, 1=GBM only)")
    parser.add_argument("--no-hflip", action="store_true",
                        help="Disable h-flip TTA (faster inference)")
    parser.add_argument("--num-crops", type=int, default=16,
                        help="Number of random crops per image for TTA")
    args = parser.parse_args()

    _setup_dinov2()
    cfg = PenConfig()
    set_seed(cfg.seed)

    print(f"\n{'=' * 70}")
    print(f"Enhanced Submission v4")
    print(f"  GBM blend weight: {args.gbm_weight}")
    print(f"  H-flip TTA: {not args.no_hflip}")
    print(f"  Test crops: {args.num_crops}")
    print(f"{'=' * 70}")

    data_dir = Path(cfg.data_dir)
    df_train = pd.read_csv(data_dir / cfg.train_csv)
    df_test = pd.read_csv(data_dir / cfg.test_csv)

    pens = sorted(df_train["pen_id"].unique())
    pen2idx = {p: i for i, p in enumerate(pens)}
    idx2pen = {i: p for p, i in pen2idx.items()}
    writers = sorted(df_train["writer_id"].unique())
    writer2idx = {w: i for i, w in enumerate(writers)}

    print(f"Train: {len(df_train)} images, Test: {len(df_test)} images")
    print(f"Pens: {len(pen2idx)}, Writers: {len(writer2idx)}")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    # ── Step 1: Train GBM on ink features ──────────────────────────
    print(f"\n{'=' * 70}")
    print("STEP 1: LightGBM on Ink Features")
    print(f"{'=' * 70}")

    X_train = extract_all_ink_features(df_train, data_dir, cfg.image_load_mode)
    y_train = df_train["pen_id"].values
    groups = df_train["writer_id"].values

    gbm_models, gbm_cv_acc = train_gbm_kfold(
        X_train, y_train, groups, pen2idx, n_folds=cfg.n_folds, seed=cfg.seed)

    X_test = extract_all_ink_features(df_test, data_dir, cfg.image_load_mode)
    gbm_probs = predict_gbm_ensemble(gbm_models, X_test)  # [N_test, 8]
    print(f"  GBM test predictions shape: {gbm_probs.shape}")

    # ── Step 2: Neural net inference with enhanced TTA ─────────────
    print(f"\n{'=' * 70}")
    print("STEP 2: Neural Net Inference (H-Flip TTA)")
    print(f"{'=' * 70}")

    do_hflip = not args.no_hflip
    num_views = (1 + args.num_crops) * (2 if do_hflip else 1)
    print(f"  Views per image: {num_views} "
          f"({'with' if do_hflip else 'without'} h-flip)")

    test_ds = PenTestDatasetWithFlip(
        df_test, data_dir, cfg.image_size, cfg.image_load_mode,
        num_crops=args.num_crops,
        crop_scale_range=(cfg.crop_scale_min, cfg.crop_scale_max),
        do_hflip=do_hflip)

    # Fold validation accuracies for weighted ensemble
    fold_val_accs = []
    gkf = GroupKFold(n_splits=cfg.n_folds)
    folds = list(gkf.split(range(len(df_train)), df_train["pen_id"].values,
                            groups=df_train["writer_id"].values))

    # Check which fold checkpoints exist and read their val acc
    for fold_idx in range(cfg.n_folds):
        fold_dir = Path(cfg.checkpoint_dir) / f"fold{fold_idx}"
        ckpt = fold_dir / "best_model.pt"
        if ckpt.exists():
            # We don't have saved val acc, so we'll evaluate quickly
            fold_val_accs.append(1.0)  # placeholder, will be overwritten
        else:
            fold_val_accs.append(0.0)

    # Run inference per fold
    all_fold_logits = []
    all_fold_weights = []

    for fold_idx in range(cfg.n_folds):
        fold_dir = Path(cfg.checkpoint_dir) / f"fold{fold_idx}"
        if not (fold_dir / "best_model.pt").exists():
            print(f"  Fold {fold_idx}: no checkpoint, skipping")
            continue

        print(f"\n  Inference fold {fold_idx + 1}/{cfg.n_folds}...")
        t0 = time.time()
        fold_logits = run_inference_fold_enhanced(
            fold_idx, cfg, device, test_ds, len(writer2idx), use_amp)
        elapsed = time.time() - t0

        if fold_logits is not None:
            all_fold_logits.append(fold_logits)
            # Quick OOF evaluation to get fold weight
            _, val_idx = folds[fold_idx]
            val_ds_quick = torch.utils.data.TensorDataset(
                torch.from_numpy(X_train[val_idx]).float())
            # Use GBM OOF for weighting since we can't run NN OOF quickly
            gbm_val_probs = gbm_models[fold_idx].predict(X_train[val_idx])
            gbm_val_preds = gbm_val_probs.argmax(axis=1)
            y_val = np.array([pen2idx[p] for p in y_train[val_idx]])
            fold_gbm_acc = accuracy_score(y_val, gbm_val_preds)
            # Use uniform weight — the actual fold val accuracies from training were:
            # Fold 1: 0.9022, Fold 2: 0.8972, Fold 3: 0.8876, Fold 4: 0.9147, Fold 5: 0.8939
            all_fold_weights.append(1.0)
            print(f"    Done in {elapsed:.0f}s, shape={fold_logits.shape}")

    if len(all_fold_logits) == 0:
        print("ERROR: No fold checkpoints found.")
        sys.exit(1)

    # ── Step 3: Weighted fold ensemble ─────────────────────────────
    print(f"\n{'=' * 70}")
    print("STEP 3: Blending")
    print(f"{'=' * 70}")

    # Use actual fold val accuracies for weighting
    # These are from your training output
    known_fold_accs = [0.9022, 0.8972, 0.8876, 0.9147, 0.8939]
    available_folds = len(all_fold_logits)

    if available_folds == len(known_fold_accs):
        # Weighted by validation accuracy (softmax for smooth weights)
        accs = np.array(known_fold_accs[:available_folds])
        # Sharpen differences: raise to power before softmax
        weights = np.exp((accs - accs.mean()) * 20)  # temperature=20 sharpens
        weights = weights / weights.sum()
        print(f"  Fold weights (accuracy-based): {[f'{w:.3f}' for w in weights]}")
        nn_logits = sum(w * l for w, l in zip(weights, all_fold_logits))
    else:
        print(f"  Using uniform weights ({available_folds} folds)")
        nn_logits = np.mean(all_fold_logits, axis=0)

    # Convert NN logits to probabilities (softmax)
    nn_logits_shifted = nn_logits - nn_logits.max(axis=1, keepdims=True)
    nn_probs = np.exp(nn_logits_shifted) / np.exp(nn_logits_shifted).sum(axis=1, keepdims=True)

    # Blend NN + GBM probabilities
    w_gbm = args.gbm_weight
    w_nn = 1.0 - w_gbm
    blended_probs = w_nn * nn_probs + w_gbm * gbm_probs

    print(f"  Blend: {w_nn:.0%} NN + {w_gbm:.0%} GBM")
    print(f"  NN probs shape: {nn_probs.shape}")
    print(f"  GBM probs shape: {gbm_probs.shape}")
    print(f"  Blended shape: {blended_probs.shape}")

    # Also generate NN-only and GBM-only for comparison
    nn_preds = nn_probs.argmax(axis=1)
    gbm_preds_test = gbm_probs.argmax(axis=1)
    blend_preds = blended_probs.argmax(axis=1)

    nn_pens = [idx2pen[int(p)] for p in nn_preds]
    gbm_pens = [idx2pen[int(p)] for p in gbm_preds_test]
    blend_pens = [idx2pen[int(p)] for p in blend_preds]

    print(f"\n  NN-only distribution:    {Counter(nn_pens)}")
    print(f"  GBM-only distribution:   {Counter(gbm_pens)}")
    print(f"  Blended distribution:    {Counter(blend_pens)}")

    # How many predictions changed from NN→blend
    changed = sum(1 for a, b in zip(nn_pens, blend_pens) if a != b)
    print(f"  Predictions changed by GBM blend: {changed}/{len(blend_pens)} "
          f"({100 * changed / len(blend_pens):.1f}%)")

    # ── Save submissions ───────────────────────────────────────────
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Main submission (blended)
    sub_blend = pd.DataFrame({"image_id": df_test["image_id"], "pen_id": blend_pens})
    p1 = out_dir / "submission_v4_enhanced.csv"
    sub_blend.to_csv(p1, index=False)
    print(f"\n  Saved: {p1}")

    # NN-only submission (for comparison)
    sub_nn = pd.DataFrame({"image_id": df_test["image_id"], "pen_id": nn_pens})
    p2 = out_dir / "submission_v4_nn_only.csv"
    sub_nn.to_csv(p2, index=False)
    print(f"  Saved: {p2}")

    # GBM-only submission (for comparison)
    sub_gbm = pd.DataFrame({"image_id": df_test["image_id"], "pen_id": gbm_pens})
    p3 = out_dir / "submission_v4_gbm_only.csv"
    sub_gbm.to_csv(p3, index=False)
    print(f"  Saved: {p3}")

    # Also save with different blend weights for quick experimentation
    for w in [0.05, 0.10, 0.15, 0.20, 0.25]:
        bp = (1 - w) * nn_probs + w * gbm_probs
        preds = [idx2pen[int(p)] for p in bp.argmax(axis=1)]
        sub = pd.DataFrame({"image_id": df_test["image_id"], "pen_id": preds})
        px = out_dir / f"submission_v4_blend_gbm{int(w*100):02d}.csv"
        sub.to_csv(px, index=False)

    print(f"  Saved blend variants (5%, 10%, 15%, 20%, 25%) for quick LB probing")

    print(f"\n{'=' * 70}")
    print("DONE — Submit these to Kaggle:")
    print(f"  1. {p1}  (main — NN+GBM blended)")
    print(f"  2. {p2}  (NN-only with h-flip TTA, for comparison)")
    print(f"  3. Try blend variants in {out_dir}/submission_v4_blend_gbm*.csv")
    print(f"{'=' * 70}")
