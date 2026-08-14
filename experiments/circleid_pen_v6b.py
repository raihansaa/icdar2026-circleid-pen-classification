"""
CircleID — Pen Classification v6b (ABLATION: MLP LoRA only)
v5 baseline + expanded LoRA (QKV + attn.proj + MLP fc1/fc2) + lower lr_lora.
Everything else identical to v5.


"""

# ══════════════════════════════════════════════════════════════════════════
# 1. Setup & Config
# ══════════════════════════════════════════════════════════════════════════

import os
import sys
import warnings
import time
import math
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from collections import Counter

warnings.filterwarnings("ignore", message="xFormers is not available")

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent  
DINOV2_REPO_DIR = BASE_DIR / "dinov2_repo"
DINOV2_WEIGHTS_PATH = BASE_DIR / "dinov2_vitb14_reg4_pretrain.pth"

NUM_INK_FEATURES = 51


def _setup_dinov2():
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    if not DINOV2_REPO_DIR.exists() or not DINOV2_WEIGHTS_PATH.exists():
        print("Downloading DINOv2 model from torch.hub...")
        model_tmp = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        del model_tmp
        hub_cache = Path(torch.hub.get_dir())
        repo_src = None
        for d in hub_cache.iterdir():
            if d.is_dir() and "facebookresearch_dinov2" in d.name:
                repo_src = d
                break
        weights_src = hub_cache / "checkpoints" / "dinov2_vitb14_reg4_pretrain.pth"
        if repo_src and not DINOV2_REPO_DIR.exists():
            shutil.copytree(str(repo_src), str(DINOV2_REPO_DIR))
        if weights_src.exists() and not DINOV2_WEIGHTS_PATH.exists():
            shutil.copy(str(weights_src), str(DINOV2_WEIGHTS_PATH))
    else:
        print(f"DINOv2 found locally")


@dataclass
class PenConfig:
    data_dir: Path = BASE_DIR / "icdar-2026-circleid-pen-classification"
    train_csv: str = "train.csv"
    test_csv: str = "test.csv"
    output_dir: Path = BASE_DIR / "outputs_v6b"
    checkpoint_dir: Path = BASE_DIR / "checkpoints_v6b"
    dinov2_repo: str = str(DINOV2_REPO_DIR)
    dinov2_weights: str = str(DINOV2_WEIGHTS_PATH)
    num_pens: int = 8
    # ── Resolution: 336 = 24×14 patches ──────────────────────────────
    image_size: int = 336                  
    image_load_mode: str = "cv2"
    dinov2_variant: str = "dinov2_vitb14_reg"
    dinov2_embed_dim: int = 768
    # ── LoRA EXPANDED ────────────────────────────────────────────────
    lora_rank: int = 32                  
    lora_alpha: int = 64                  
    lora_last_n_blocks: int = 8           
    unfreeze_last_n_blocks: int = 4      
    # ── Architecture ─────────────────────────────────────────────────
    num_ink_features: int = NUM_INK_FEATURES
    ink_proj_dim: int = 128                
    projection_dim: int = 512
    pen_embed_dim: int = 128
    head_dropout: float = 0.15
    # ── ArcFace ──────────────────────────────────────────────────────
    arcface_s: float = 30.0
    arcface_m: float = 0.35
    arcface_k: int = 3
    # ── Training ─────────────────────────────────────────────────────
    epochs: int = 50                      
    batch_size: int = 32                   
    grad_accum_steps: int = 4             
    lr_lora: float = 7e-5                   
    lr_heads: float = 5e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    label_smoothing: float = 0.05
    # ── Same-Pen Mixup ──────────────────────────────────────────────
    mixup_prob: float = 0.3              
    mixup_alpha: float = 0.4             
    # ── K-Fold ───────────────────────────────────────────────────────
    n_folds: int = 5
    # ── Writer Adversarial ───────────────────────────────────────────
    lambda_adv: float = 0.3
    adv_ramp_end: int = 15
    # ── Contrastive ──────────────────────────────────────────────────
    lambda_contrast: float = 0.15
    contrast_tau: float = 0.07
    contrast_warmup_end: int = 5
    contrast_ramp_end: int = 15
    # ── EMA ──────────────────────────────────────────────────────────
    use_ema: bool = True
    ema_decay: float = 0.999
    ema_start_epoch: int = 5
    # ── Augmentation ─────────────────────────────────────────────────
    jitter_brightness: float = 0.03
    jitter_contrast: float = 0.05
    random_rotation_deg: float = 15.0
    hflip_prob: float = 0.5
    crop_scale_min: float = 0.5
    crop_scale_max: float = 1.0
    num_test_crops: int = 12
    # NOTE: random erasing REMOVED — it destroys ink signal
    eps: float = 1e-8
    # ── System ───────────────────────────────────────────────────────
    num_workers: int = 2
    pin_memory: bool = True
    device: str = "cuda"
    seed: int = 42


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════════════
# 2. Rich Ink Feature Extraction (51 features)
# ══════════════════════════════════════════════════════════════════════════

def _compute_skewness(x):
    m = x.mean()
    s = x.std()
    if s < 1e-8:
        return 0.0
    return float(((x - m) ** 3).mean() / (s ** 3))


def _compute_kurtosis(x):
    m = x.mean()
    s = x.std()
    if s < 1e-8:
        return 0.0
    return float(((x - m) ** 4).mean() / (s ** 4) - 3.0)


def _compute_lbp(gray):
    """Simple 8-neighbor LBP using cv2 (no skimage dependency)."""
    h, w = gray.shape
    padded = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    center = padded[1:h + 1, 1:w + 1].astype(np.int16)
    lbp = np.zeros((h, w), dtype=np.uint8)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for bit, (dy, dx) in enumerate(offsets):
        neighbor = padded[1 + dy:h + 1 + dy, 1 + dx:w + 1 + dx].astype(np.int16)
        lbp |= ((neighbor >= center).astype(np.uint8) << bit)
    return lbp


def _compute_gabor_features(gray, mask):
    """4 orientations × 2 frequencies = 8 Gabor features."""
    feats = []
    gray_f = gray.astype(np.float32)
    for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
        for sigma in [1.0, 3.0]:
            kernel = cv2.getGaborKernel((9, 9), sigma, theta, 5.0, 0.5, 0,
                                         ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray_f, cv2.CV_32F, kernel)
            ink_vals = filtered[mask > 0]
            feats.append(float(ink_vals.mean()) if len(ink_vals) > 0 else 0.0)
    return feats


def extract_ink_features(image, mode="cv2"):
    """Extract 51 handcrafted features from raw image.

    Groups:
      1-7:   Intensity distribution (mean, std, p10-p90)
      8-13:  RGB channel means + stds on ink
      14-16: Color differences (R-G, B-G, R-G std)
      17-20: HSV (H mean/std, S mean/std)
      21-24: Stroke width (DT mean/std/max/median)
      25:    Ink coverage
      26:    Edge gradient mean
      27:    Ink range (p90-p10)
      28:    Core uniformity
      29:    Intensity skewness
      30:    Intensity kurtosis
      31:    Border pixel ratio
      32-39: Gabor filter responses (4 orient × 2 freq)
      40-47: LBP histogram (8 bins)
      48-50: Connected components (count, mean area, max area)
      51:    FFT low-frequency concentration
    """
    if mode == "cv2":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        rgb = image[:, :, ::-1].copy()
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        rgb = image

    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if mask.mean() < 5 or mask.mean() > 250:
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
    ink_px = gray[mask > 0]

    if len(ink_px) < 10:
        return np.zeros(NUM_INK_FEATURES, dtype=np.float32)

    feats = []

    # 1-7: Intensity distribution
    feats.extend([ink_px.mean(), ink_px.std(),
                  np.percentile(ink_px, 10), np.percentile(ink_px, 25),
                  np.percentile(ink_px, 50), np.percentile(ink_px, 75),
                  np.percentile(ink_px, 90)])

    # 8-13: RGB channel stats
    r = rgb[:, :, 0][mask > 0].astype(np.float32)
    g = rgb[:, :, 1][mask > 0].astype(np.float32)
    b = rgb[:, :, 2][mask > 0].astype(np.float32)
    feats.extend([r.mean(), r.std(), g.mean(), g.std(), b.mean(), b.std()])

    # 14-16: Color differences
    feats.extend([(r - g).mean(), (b - g).mean(), (r - g).std()])

    # 17-20: HSV
    h_ch = hsv[:, :, 0][mask > 0].astype(np.float32)
    s_ch = hsv[:, :, 1][mask > 0].astype(np.float32)
    feats.extend([h_ch.mean(), h_ch.std(), s_ch.mean(), s_ch.std()])

    # 21-24: Stroke width
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dt_ink = dt[mask > 0]
    feats.extend([dt_ink.mean(), dt_ink.std(), dt_ink.max(), np.median(dt_ink)])

    # 25: Ink coverage
    feats.append(mask.mean() / 255.0)

    # 26: Edge gradient
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gmag = np.sqrt(gx ** 2 + gy ** 2)
    feats.append(gmag[mask > 0].mean())

    # 27: Ink range
    feats.append(np.percentile(ink_px, 90) - np.percentile(ink_px, 10))

    # 28: Core uniformity
    kernel3 = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask, kernel3, iterations=1)
    core_px = gray[eroded > 0]
    feats.append(core_px.std() if len(core_px) > 5 else ink_px.std())

    # 29: Intensity skewness (pen 3 vs 7 separator)
    feats.append(_compute_skewness(ink_px.astype(np.float64)))

    # 30: Intensity kurtosis (pen 3: 0.695 vs pen 7: 0.893)
    feats.append(_compute_kurtosis(ink_px.astype(np.float64)))

    # 31: Border pixel ratio
    dilated = cv2.dilate(mask, kernel3, iterations=1)
    boundary = dilated - eroded
    border_count = (boundary > 0).sum()
    ink_count = (mask > 0).sum()
    feats.append(border_count / (ink_count + 1e-8))

    # 32-39: Gabor features (4 orientations × 2 frequencies)
    feats.extend(_compute_gabor_features(gray, mask))

    # 40-47: LBP histogram (8 bins)
    lbp = _compute_lbp(gray)
    lbp_ink = lbp[mask > 0]
    hist, _ = np.histogram(lbp_ink, bins=8, range=(0, 256))
    hist = hist.astype(np.float32)
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist = hist / hist_sum
    feats.extend(hist.tolist())

    # 48-50: Connected components
    n_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
        feats.extend([float(n_labels - 1), areas.mean(), areas.max()])
    else:
        feats.extend([0.0, 0.0, 0.0])

    # 51: FFT low-frequency concentration
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    fh, fw = magnitude.shape
    cy, cx = fh // 2, fw // 2
    r = max(1, min(fh, fw) // 8)
    low_freq = magnitude[cy - r:cy + r, cx - r:cx + r].mean()
    total_freq = magnitude.mean() + 1e-8
    feats.append(low_freq / total_freq)

    assert len(feats) == NUM_INK_FEATURES, f"Expected {NUM_INK_FEATURES}, got {len(feats)}"
    return np.array(feats, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════
# 3. Preprocessing
# ══════════════════════════════════════════════════════════════════════════

def to_gray(image, mode="cv2"):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if mode == "cv2" else cv2.COLOR_RGB2GRAY)


def compute_ink_mask(image, mode="cv2"):
    gray = to_gray(image, mode)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if mask.mean() < 5 or mask.mean() > 250:
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
    return mask.astype(np.float32) / 255.0


def apply_intensity_jitter(image, brightness_range=0.03, contrast_range=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    bf = 1.0 + rng.uniform(-brightness_range, brightness_range)
    cf = 1.0 + rng.uniform(-contrast_range, contrast_range)
    img = image.astype(np.float32)
    mean = img.mean()
    img = ((img - mean) * cf + mean) * bf
    return np.clip(img, 0, 255).astype(np.uint8)


def apply_random_rotation(image, max_deg=15.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    angle = rng.uniform(-max_deg, max_deg)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(255, 255, 255))


def apply_random_scale_crop(image, scale_range=(0.5, 1.0), rng=None):
    if rng is None:
        rng = np.random.default_rng()
    h, w = image.shape[:2]
    scale = rng.uniform(*scale_range)
    crop_h, crop_w = int(h * scale), int(w * scale)
    if crop_h >= h or crop_w >= w:
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    ink_mask = (gray < 200).astype(np.float32)
    ink_coords = np.where(ink_mask > 0)
    if len(ink_coords[0]) > 50:
        idx = rng.integers(0, len(ink_coords[0]))
        cy, cx = ink_coords[0][idx], ink_coords[1][idx]
        top = max(0, min(cy - crop_h // 2, h - crop_h))
        left = max(0, min(cx - crop_w // 2, w - crop_w))
    else:
        top = rng.integers(0, max(1, h - crop_h + 1))
        left = rng.integers(0, max(1, w - crop_w + 1))
    cropped = image[top:top + crop_h, left:left + crop_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)


def preprocess_image(image, mode="cv2", image_size=336,
                     apply_jitter=False, jitter_brightness=0.03, jitter_contrast=0.05,
                     rotation_deg=0.0, random_scale_crop=False,
                     crop_scale_range=(0.5, 1.0), hflip=False, rng=None):
    if hflip:
        image = cv2.flip(image, 1)
    if random_scale_crop:
        image = apply_random_scale_crop(image, scale_range=crop_scale_range, rng=rng)
    resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LANCZOS4)
    if apply_jitter and rotation_deg > 0:
        resized = apply_random_rotation(resized, rotation_deg, rng)
    if apply_jitter:
        resized = apply_intensity_jitter(resized, jitter_brightness, jitter_contrast, rng)
    mask = compute_ink_mask(resized, mode)
    return resized, mask


def ink_weighted_pooling(patch_tokens, mask_float, eps=1e-8):
    num_patches = patch_tokens.shape[1]
    grid_size = int(num_patches ** 0.5)
    mask_patches = F.adaptive_avg_pool2d(
        mask_float.unsqueeze(1), (grid_size, grid_size)).flatten(1)
    mask_sum = mask_patches.sum(dim=1, keepdim=True)
    uniform = torch.ones_like(mask_patches) / mask_patches.shape[1]
    use_uniform = (mask_sum < eps).float()
    mask_patches = (1 - use_uniform) * mask_patches + use_uniform * uniform
    weights = mask_patches / (mask_patches.sum(dim=1, keepdim=True) + eps)
    pooled = (patch_tokens * weights.unsqueeze(-1)).sum(dim=1)
    return pooled


# ══════════════════════════════════════════════════════════════════════════
# 4. Dataset (NO random erasing — ink pixels are signal)
# ══════════════════════════════════════════════════════════════════════════

class PenDataset(Dataset):
    def __init__(self, df, indices, data_dir, pen2idx, writer2idx,
                 image_size=336, image_load_mode="cv2", is_training=False,
                 jitter_brightness=0.03, jitter_contrast=0.05, rotation_deg=0.0,
                 hflip_prob=0.5, crop_scale_range=(0.5, 1.0)):
        self.df = df
        self.indices = indices
        self.data_dir = Path(data_dir)
        self.pen2idx = pen2idx
        self.writer2idx = writer2idx
        self.image_size = image_size
        self.mode = image_load_mode
        self.is_training = is_training
        self.jitter_brightness = jitter_brightness
        self.jitter_contrast = jitter_contrast
        self.rotation_deg = rotation_deg
        self.hflip_prob = hflip_prob
        self.crop_scale_range = crop_scale_range
        self.rng = np.random.default_rng()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.ink_features = self._precompute_ink_features()

    def _precompute_ink_features(self):
        tag = "train" if self.is_training else "val"
        print(f"  Precomputing ink features ({tag}, {len(self.indices)} images)...")
        t0 = time.time()
        features = []
        for global_idx in self.indices:
            row = self.df.iloc[global_idx]
            img_path = self.data_dir / row["image_path"]
            img = cv2.imread(str(img_path)) if self.mode == "cv2" else \
                np.array(__import__('PIL').Image.open(img_path).convert("RGB"))
            feats = extract_ink_features(img, self.mode) if img is not None \
                else np.zeros(NUM_INK_FEATURES, dtype=np.float32)
            features.append(feats)
        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s")
        return np.array(features, dtype=np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row = self.df.iloc[self.indices[idx]]
        img_path = self.data_dir / row["image_path"]
        if self.mode == "cv2":
            image = cv2.imread(str(img_path))
            if image is None:
                raise FileNotFoundError(f"Failed to load: {img_path}")
        else:
            from PIL import Image
            image = np.array(Image.open(img_path).convert("RGB"))

        do_hflip = self.is_training and self.rng.random() < self.hflip_prob

        final_image, mask_float = preprocess_image(
            image, mode=self.mode, image_size=self.image_size,
            apply_jitter=self.is_training,
            jitter_brightness=self.jitter_brightness,
            jitter_contrast=self.jitter_contrast,
            rotation_deg=self.rotation_deg if self.is_training else 0.0,
            random_scale_crop=self.is_training,
            crop_scale_range=self.crop_scale_range,
            hflip=do_hflip, rng=self.rng)

        img_tensor = torch.from_numpy(final_image).float().permute(2, 0, 1) / 255.0
        if self.mode == "cv2":
            img_tensor = img_tensor[[2, 1, 0], ...]
        img_tensor = (img_tensor - self.mean) / self.std
        # NO random erasing — ink pixels are the signal we're classifying

        mask_tensor = torch.from_numpy(mask_float).float()
        ink_feats = torch.from_numpy(self.ink_features[idx]).float()
        pen_label = self.pen2idx[row["pen_id"]]
        writer_id = self.writer2idx[row["writer_id"]]
        return {"image": img_tensor, "mask": mask_tensor, "ink_feats": ink_feats,
                "pen_label": pen_label, "writer_id": writer_id}


class PenTestDataset(Dataset):
    def __init__(self, df, data_dir, image_size=336, image_load_mode="cv2",
                 num_crops=0, crop_scale_range=(0.5, 1.0)):
        self.df = df
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.mode = image_load_mode
        self.num_crops = num_crops
        self.crop_scale_range = crop_scale_range
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.ink_features = self._precompute_ink_features()

    def _precompute_ink_features(self):
        print(f"  Precomputing ink features (test, {len(self.df)} images)...")
        t0 = time.time()
        features = []
        for _, row in self.df.iterrows():
            img_path = self.data_dir / row["image_path"]
            img = cv2.imread(str(img_path)) if self.mode == "cv2" else \
                np.array(__import__('PIL').Image.open(img_path).convert("RGB"))
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
        return img_tensor, torch.from_numpy(mask_float).float()

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

        if self.num_crops == 0:
            final_image, mask_float = preprocess_image(
                image, mode=self.mode, image_size=self.image_size, apply_jitter=False)
            img_t, mask_t = self._to_tensor(final_image, mask_float)
            return {"image": img_t, "mask": mask_t, "ink_feats": ink_feats}

        rng = np.random.default_rng(idx)
        imgs, masks, feats = [], [], []
        full_img, full_mask = preprocess_image(
            image, mode=self.mode, image_size=self.image_size, apply_jitter=False)
        img_t, mask_t = self._to_tensor(full_img, full_mask)
        imgs.append(img_t)
        masks.append(mask_t)
        feats.append(ink_feats)
        for _ in range(self.num_crops):
            cropped = apply_random_scale_crop(image, self.crop_scale_range, rng)
            crop_img, crop_mask = preprocess_image(
                cropped, mode=self.mode, image_size=self.image_size, apply_jitter=False)
            img_t, mask_t = self._to_tensor(crop_img, crop_mask)
            imgs.append(img_t)
            masks.append(mask_t)
            feats.append(ink_feats)
        return {"image": torch.stack(imgs), "mask": torch.stack(masks),
                "ink_feats": torch.stack(feats)}


# ══════════════════════════════════════════════════════════════════════════
# 5. Balanced Pen-Writer Batch Sampler
# ══════════════════════════════════════════════════════════════════════════

class BalancedPenWriterSampler(Sampler):
    def __init__(self, df, indices, pen2idx, batch_size=16, seed=42):
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.samples_per_pen = max(1, batch_size // len(pen2idx))
        self.pen_writer_indices = {}
        self.pen_writers = {}
        for i, global_idx in enumerate(indices):
            row = df.iloc[global_idx]
            p = pen2idx[row["pen_id"]]
            w = row["writer_id"]
            self.pen_writer_indices.setdefault((p, w), []).append(i)
            self.pen_writers.setdefault(p, set()).add(w)
        self.pen_writers = {p: list(ws) for p, ws in self.pen_writers.items()}
        self.all_pens = sorted(self.pen_writers.keys())
        self.num_batches = len(indices) // batch_size
        avg_w = np.mean([len(ws) for ws in self.pen_writers.values()])
        print(f"  BalancedSampler: {len(self.all_pens)} pens, "
              f"avg {avg_w:.1f} writers/pen, {self.samples_per_pen}/pen/batch")

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            for pen in self.all_pens:
                writers = self.pen_writers[pen]
                n = min(self.samples_per_pen, len(writers))
                sel_writers = self.rng.choice(writers, n, replace=False)
                for w in sel_writers:
                    batch.append(self.rng.choice(self.pen_writer_indices[(pen, w)]))
            while len(batch) < self.batch_size:
                pen = self.rng.choice(self.all_pens)
                w = self.rng.choice(self.pen_writers[pen])
                batch.append(self.rng.choice(self.pen_writer_indices[(pen, w)]))
            self.rng.shuffle(batch)
            yield batch[:self.batch_size]

    def __len__(self):
        return self.num_batches


# ══════════════════════════════════════════════════════════════════════════
# 6. Losses
# ══════════════════════════════════════════════════════════════════════════

class SubCenterArcFace(nn.Module):
    def __init__(self, embed_dim, num_classes, K=3, s=30.0, m=0.35):
        super().__init__()
        self.K = K
        self.s = s
        self.m = m
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.FloatTensor(num_classes * K, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels=None):
        W = F.normalize(self.weight, dim=1)
        cosine_all = F.linear(embeddings, W).view(-1, self.num_classes, self.K)
        cosine = cosine_all.max(dim=2).values
        if labels is not None and self.training:
            theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
            one_hot = F.one_hot(labels, self.num_classes).bool()
            theta = torch.where(one_hot, theta + self.m, theta)
            cosine = torch.cos(theta)
        return cosine * self.s


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def masked_logsumexp(sim, mask, dim=1):
    neg_inf = torch.finfo(sim.dtype).min
    return torch.logsumexp(sim.masked_fill(~mask, neg_inf), dim=dim)


def pen_contrastive_loss(pen_emb, pen_ids, writer_ids, tau=0.07):
    B = pen_emb.shape[0]
    device = pen_emb.device
    eye = torch.eye(B, dtype=torch.bool, device=device)
    same_p = pen_ids.unsqueeze(1) == pen_ids.unsqueeze(0)
    same_w = writer_ids.unsqueeze(1) == writer_ids.unsqueeze(0)
    pos_mask = same_p & ~same_w & ~eye
    all_mask = ~eye
    sim = torch.mm(pen_emb, pen_emb.t()) / tau
    has_pos = pos_mask.any(dim=1)
    if has_pos.sum() > 0:
        num = masked_logsumexp(sim[has_pos], pos_mask[has_pos])
        den = masked_logsumexp(sim[has_pos], all_mask[has_pos])
        return -(num - den).mean()
    return torch.tensor(0.0, device=device)


def get_ramp(epoch, target, warmup_end, ramp_end):
    if epoch < warmup_end:
        return 0.0
    elif epoch < ramp_end:
        return target * (epoch - warmup_end) / (ramp_end - warmup_end)
    return target


# ══════════════════════════════════════════════════════════════════════════
# 7. Model
# ══════════════════════════════════════════════════════════════════════════

class LoRALinear(nn.Module):
    def __init__(self, original, rank=32, alpha=64):
        super().__init__()
        self.original = original
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(original.in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, original.out_features))
        original.weight.requires_grad = False
        if original.bias is not None:
            original.bias.requires_grad = False

    def forward(self, x):
        return self.original(x) + (x @ self.lora_A @ self.lora_B) * self.scaling


def apply_lora_to_dinov2(model, rank=32, alpha=64, last_n_blocks=8):
    """Apply LoRA to QKV + attn.proj + MLP fc1/fc2 in last N blocks."""
    blocks = model.blocks
    n = len(blocks)
    lora_params = []

    for i in range(n - last_n_blocks, n):
        block = blocks[i]

        # Attention QKV (same as v5)
        lora_qkv = LoRALinear(block.attn.qkv, rank, alpha)
        block.attn.qkv = lora_qkv
        lora_params.extend([lora_qkv.lora_A, lora_qkv.lora_B])

        # Attention output projection
        lora_proj = LoRALinear(block.attn.proj, rank, alpha)
        block.attn.proj = lora_proj
        lora_params.extend([lora_proj.lora_A, lora_proj.lora_B])

        # MLP fc1/fc2 — half rank since these layers are already 4x wider
        mlp_rank = max(8, rank // 2)
        mlp_alpha = max(16, alpha // 2)

        lora_fc1 = LoRALinear(block.mlp.fc1, mlp_rank, mlp_alpha)
        block.mlp.fc1 = lora_fc1
        lora_params.extend([lora_fc1.lora_A, lora_fc1.lora_B])

        lora_fc2 = LoRALinear(block.mlp.fc2, mlp_rank, mlp_alpha)
        block.mlp.fc2 = lora_fc2
        lora_params.extend([lora_fc2.lora_A, lora_fc2.lora_B])

    return model, lora_params


class PenClassifierModel(nn.Module):
    def __init__(self, backbone, num_pens=8, num_writers=44, dinov2_dim=768,
                 projection_dim=512, pen_embed_dim=128,
                 num_ink_feats=NUM_INK_FEATURES, ink_proj_dim=128,
                 dropout=0.15, arcface_s=30.0, arcface_m=0.35, arcface_k=3):
        super().__init__()
        self.backbone = backbone
        self.ink_proj = nn.Sequential(
            nn.BatchNorm1d(num_ink_feats),
            nn.Linear(num_ink_feats, ink_proj_dim),
            nn.GELU(),
            nn.Linear(ink_proj_dim, ink_proj_dim),
            nn.GELU())
        self.projection = nn.Sequential(
            nn.Linear(dinov2_dim * 2 + ink_proj_dim, projection_dim),
            nn.LayerNorm(projection_dim), nn.GELU())
        self.pen_proj = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim), nn.GELU(),
            nn.Linear(projection_dim, pen_embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.arcface = SubCenterArcFace(pen_embed_dim, num_pens, arcface_k,
                                         arcface_s, arcface_m)
        self.writer_adversary = nn.Sequential(
            nn.Linear(projection_dim, projection_dim // 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(projection_dim // 2, num_writers))

    def forward(self, images, masks, ink_feats, eps=1e-8,
                labels=None, return_emb=False, adv_alpha=0.0):
        with torch.set_grad_enabled(self.training):
            out = self.backbone.forward_features(images)
        cls_token = out["x_norm_clstoken"]
        patch_tokens = out["x_norm_patchtokens"]
        patch_avg = ink_weighted_pooling(patch_tokens, masks, eps)
        ink_proj = self.ink_proj(ink_feats)
        combined = torch.cat([cls_token, patch_avg, ink_proj], dim=1)
        projected = self.projection(combined)
        pen_raw = self.pen_proj(projected)
        if self.training:
            pen_raw = self.dropout(pen_raw)
        pen_emb = F.normalize(pen_raw, p=2, dim=1)
        logits = self.arcface(pen_emb, labels if self.training else None)
        if return_emb:
            reversed_feat = GradientReversal.apply(projected, adv_alpha)
            writer_logits = self.writer_adversary(reversed_feat)
            return logits, pen_emb, writer_logits
        return logits


class EMAModel:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone()
                       for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    def reinit_shadow(self, model):
        self.shadow = {n: p.data.clone()
                       for n, p in model.named_parameters() if p.requires_grad}

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


def build_pen_model(cfg, num_writers=44):
    backbone = torch.hub.load(cfg.dinov2_repo, cfg.dinov2_variant,
                               source="local", pretrained=False)
    state = torch.load(cfg.dinov2_weights, map_location="cpu", weights_only=False)
    backbone.load_state_dict(state, strict=True)
    backbone.eval()
    with torch.no_grad():
        out = backbone.forward_features(torch.randn(1, 3, cfg.image_size, cfg.image_size))
    assert "x_norm_clstoken" in out and "x_norm_patchtokens" in out
    print(f"  DINOv2 @ {cfg.image_size}px: patches={out['x_norm_patchtokens'].shape[1]}")

    for p in backbone.parameters():
        p.requires_grad = False
    backbone, lora_params = apply_lora_to_dinov2(
        backbone, cfg.lora_rank, cfg.lora_alpha, cfg.lora_last_n_blocks)
    print(f"  LoRA: rank={cfg.lora_rank}, {sum(p.numel() for p in lora_params):,} params "
          f"in last {cfg.lora_last_n_blocks} blocks")

    # Unfreeze last N blocks (configurable, was hardcoded to 2 in v4)
    lora_ids = {id(p) for p in lora_params}
    unfreeze_params = []
    n_blocks = len(backbone.blocks)
    for i in range(n_blocks - cfg.unfreeze_last_n_blocks, n_blocks):
        for p in backbone.blocks[i].parameters():
            p.requires_grad = True
            if id(p) not in lora_ids:
                unfreeze_params.append(p)
    print(f"  Unfrozen last {cfg.unfreeze_last_n_blocks} blocks: "
          f"{sum(p.numel() for p in unfreeze_params):,} extra params")

    model = PenClassifierModel(
        backbone, cfg.num_pens, num_writers, cfg.dinov2_embed_dim,
        cfg.projection_dim, cfg.pen_embed_dim,
        cfg.num_ink_features, cfg.ink_proj_dim, cfg.head_dropout,
        cfg.arcface_s, cfg.arcface_m, cfg.arcface_k)

    all_backbone_ids = {id(p) for p in lora_params} | {id(p) for p in unfreeze_params}
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and id(p) not in all_backbone_ids]
    lora_params = lora_params + unfreeze_params

    total_t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_a = sum(p.numel() for p in model.parameters())
    print(f"  Model: {total_t:,} trainable / {total_a:,} total ({100 * total_t / total_a:.2f}%)")
    return model, lora_params, head_params


# ══════════════════════════════════════════════════════════════════════════
# 8. Training (with same-pen mixup)
# ══════════════════════════════════════════════════════════════════════════

def build_optimizer_scheduler(lora_params, head_params, cfg, steps_per_epoch):
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": cfg.lr_lora},
        {"params": head_params, "lr": cfg.lr_heads},
    ], weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])
    return optimizer, scheduler


def apply_same_pen_mixup(imgs, masks, ink_feats, labels, wids, alpha=0.4):
    """Mix images of the SAME pen, DIFFERENT writer.
    Teaches: pen properties are invariant, writer style is noise.
    Labels stay the same since both images have the same pen."""
    B = imgs.size(0)
    device = imgs.device
    mixed_imgs = imgs.clone()
    mixed_masks = masks.clone()
    mixed_feats = ink_feats.clone()

    for i in range(B):
        # Find candidates: same pen, different writer
        candidates = ((labels == labels[i]) & (wids != wids[i])).nonzero(as_tuple=True)[0]
        if len(candidates) == 0:
            continue
        j = candidates[torch.randint(len(candidates), (1,)).item()]
        lam = np.random.default_rng().beta(alpha, alpha)
        mixed_imgs[i] = lam * imgs[i] + (1 - lam) * imgs[j]
        mixed_masks[i] = lam * masks[i] + (1 - lam) * masks[j]
        mixed_feats[i] = lam * ink_feats[i] + (1 - lam) * ink_feats[j]
        # pen_label stays the same (both have same pen)

    return mixed_imgs, mixed_masks, mixed_feats


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, cfg,
                    ema=None, scaler=None):
    model.train()
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    lam_con = get_ramp(epoch, cfg.lambda_contrast,
                       cfg.contrast_warmup_end, cfg.contrast_ramp_end)
    adv_alpha = get_ramp(epoch, 1.0, cfg.contrast_warmup_end, cfg.adv_ramp_end)

    tot_loss = tot_arc = tot_con = tot_adv = c_p = c_w = n = n_mixup = 0
    optimizer.zero_grad()
    num_batches = len(loader)

    for step, batch in enumerate(loader):
        imgs = batch["image"].to(device)
        masks = batch["mask"].to(device)
        ink_feats = batch["ink_feats"].to(device)
        labels = batch["pen_label"].to(device)
        wids = batch["writer_id"].to(device)

        # Same-pen mixup (after warmup, with probability cfg.mixup_prob)
        if epoch >= cfg.warmup_epochs and np.random.random() < cfg.mixup_prob:
            imgs, masks, ink_feats = apply_same_pen_mixup(
                imgs, masks, ink_feats, labels, wids, cfg.mixup_alpha)
            n_mixup += 1

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            logits, pen_emb, writer_logits = model(
                imgs, masks, ink_feats, cfg.eps,
                labels=labels, return_emb=True, adv_alpha=adv_alpha)
            loss_arc = ce_criterion(logits, labels)
            if lam_con > 0:
                loss_con = pen_contrastive_loss(pen_emb, labels, wids, cfg.contrast_tau)
            else:
                loss_con = torch.tensor(0.0, device=device)
            loss_adv = F.cross_entropy(writer_logits, wids)
            loss = loss_arc + lam_con * loss_con + cfg.lambda_adv * loss_adv

        scaled_loss = loss / cfg.grad_accum_steps
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        do_step = ((step + 1) % cfg.grad_accum_steps == 0) or ((step + 1) == num_batches)
        if do_step:
            if scaler is not None:
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if ema and epoch >= cfg.ema_start_epoch:
                ema.update(model)

        B = imgs.size(0)
        tot_loss += loss.item() * B
        tot_arc += loss_arc.item() * B
        tot_con += loss_con.item() * B
        tot_adv += loss_adv.item() * B
        c_p += (logits.argmax(1) == labels).sum().item()
        c_w += (writer_logits.argmax(1) == wids).sum().item()
        n += B

    return {"loss": tot_loss / n, "arc": tot_arc / n, "con": tot_con / n,
            "adv": tot_adv / n, "pen_acc": c_p / n, "writer_acc": c_w / n,
            "lam_con": lam_con, "adv_alpha": adv_alpha, "n_mixup": n_mixup}


@torch.no_grad()
def evaluate(model, loader, device, eps=1e-8, use_amp=False):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(batch["image"].to(device),
                           batch["mask"].to(device),
                           batch["ink_feats"].to(device), eps)
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(batch["pen_label"].numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return {"acc": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="macro")}


# ══════════════════════════════════════════════════════════════════════════
# 9. Fold Training & Inference
# ══════════════════════════════════════════════════════════════════════════

def train_fold(fold, train_idx, val_idx, df, data_dir, pen2idx, writer2idx,
               cfg, device):
    print(f"\n{'=' * 70}")
    print(f"FOLD {fold + 1}/{cfg.n_folds}")
    print(f"{'=' * 70}")

    train_writers = set(df.iloc[train_idx]["writer_id"].unique())
    val_writers = set(df.iloc[val_idx]["writer_id"].unique())
    print(f"  Train: {len(train_idx)} ({len(train_writers)} writers), "
          f"Val: {len(val_idx)} ({len(val_writers)} writers), "
          f"Overlap: {len(train_writers & val_writers)}")

    kw = dict(num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    train_ds = PenDataset(
        df, list(train_idx), data_dir, pen2idx, writer2idx,
        cfg.image_size, image_load_mode=cfg.image_load_mode, is_training=True,
        jitter_brightness=cfg.jitter_brightness, jitter_contrast=cfg.jitter_contrast,
        rotation_deg=cfg.random_rotation_deg, hflip_prob=cfg.hflip_prob,
        crop_scale_range=(cfg.crop_scale_min, cfg.crop_scale_max))
    val_ds = PenDataset(
        df, list(val_idx), data_dir, pen2idx, writer2idx,
        cfg.image_size, image_load_mode=cfg.image_load_mode, is_training=False)

    train_sampler = BalancedPenWriterSampler(
        df, list(train_idx), pen2idx, batch_size=cfg.batch_size, seed=cfg.seed + fold)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, **kw)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, **kw)

    model, lora_params, head_params = build_pen_model(cfg, num_writers=len(writer2idx))
    model = model.to(device)
    steps_per_epoch = math.ceil(len(train_sampler) / cfg.grad_accum_steps)
    optimizer, scheduler = build_optimizer_scheduler(
        lora_params, head_params, cfg, steps_per_epoch)
    ema = EMAModel(model, cfg.ema_decay) if cfg.use_ema else None
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    use_amp = scaler is not None

    fold_dir = Path(cfg.checkpoint_dir) / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    best_f1 = 0.0
    patience_counter = 0
    patience_limit = 15

    ema_initialized = False
    for epoch in range(cfg.epochs):
        if ema and epoch == cfg.ema_start_epoch and not ema_initialized:
            ema.reinit_shadow(model)
            ema_initialized = True
            print(f"  EMA shadow re-initialized")

        t0 = time.time()
        tm = train_one_epoch(model, train_loader, optimizer, scheduler,
                             device, epoch, cfg, ema, scaler)

        if ema and epoch >= cfg.ema_start_epoch:
            ema.apply_shadow(model)
        val_metrics = evaluate(model, val_loader, device, cfg.eps, use_amp)
        if ema and epoch >= cfg.ema_start_epoch:
            ema.restore(model)

        elapsed = time.time() - t0
        ema_tag = " (EMA)" if ema and epoch >= cfg.ema_start_epoch else ""
        is_best = val_metrics["acc"] > best_acc
        mixup_tag = f" mix={tm['n_mixup']}" if tm['n_mixup'] > 0 else ""
        print(f"  E{epoch + 1:02d} ({elapsed:.0f}s) "
              f"L={tm['loss']:.4f} Arc={tm['arc']:.4f} "
              f"Con={tm['con']:.4f}(l={tm['lam_con']:.2f}) "
              f"Adv={tm['adv']:.4f}(a={tm['adv_alpha']:.2f}) "
              f"P={tm['pen_acc']:.3f} W={tm['writer_acc']:.3f}{mixup_tag} | "
              f"val_P={val_metrics['acc']:.3f} F1={val_metrics['f1']:.3f}"
              f"{ema_tag}{' *' if is_best else ''}")

        if is_best:
            best_acc = val_metrics["acc"]
            best_f1 = val_metrics["f1"]
            patience_counter = 0
            if ema and epoch >= cfg.ema_start_epoch:
                ema.apply_shadow(model)
                torch.save(model.state_dict(), fold_dir / "best_model.pt")
                ema.restore(model)
            else:
                torch.save(model.state_dict(), fold_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit and epoch >= 20:
                print(f"  Early stopping at epoch {epoch + 1} (no improvement for {patience_limit} epochs)")
                break

    print(f"  Fold {fold + 1} best: acc={best_acc:.4f} f1={best_f1:.4f}")
    del model, optimizer, scheduler, ema, scaler, train_loader, val_loader, train_ds, val_ds
    torch.cuda.empty_cache()
    return best_acc, best_f1


def run_inference_fold(fold, cfg, device, test_ds, num_writers, use_amp):
    fold_dir = Path(cfg.checkpoint_dir) / f"fold{fold}"
    ckpt_path = fold_dir / "best_model.pt"
    if not ckpt_path.exists():
        print(f"  WARNING: No checkpoint for fold {fold}, skipping")
        return None

    model, _, _ = build_pen_model(cfg, num_writers=num_writers)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
    model = model.to(device)
    model.eval()

    K = test_ds[0]["image"].shape[0]
    max_views = 64  # conservative for 336px
    infer_bs = max(1, max_views // K)
    test_loader = DataLoader(test_ds, batch_size=infer_bs, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)

    all_logits = []
    n_done = 0
    with torch.no_grad():
        for batch in test_loader:
            B = batch["image"].shape[0]
            imgs = batch["image"].view(B * K, *batch["image"].shape[2:]).to(device)
            masks = batch["mask"].view(B * K, *batch["mask"].shape[2:]).to(device)
            ink_feats = batch["ink_feats"].view(B * K, -1).to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(imgs, masks, ink_feats, cfg.eps)
            logits = logits.view(B, K, -1).mean(dim=1)
            all_logits.append(logits.cpu().numpy())
            n_done += B
            if n_done % 3000 < infer_bs:
                print(f"    Fold {fold}: {n_done}/{len(test_ds)}")

    del model
    torch.cuda.empty_cache()
    return np.concatenate(all_logits, axis=0)


# ══════════════════════════════════════════════════════════════════════════
# 10. Main
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--infer-only", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    _setup_dinov2()
    cfg = PenConfig()
    if args.seed is not None:
        cfg.seed = args.seed
    set_seed(cfg.seed)

    print(f"\n{'=' * 70}")
    print(f"CircleID Pen v6b — v5+MLPLoRA — {cfg.n_folds}-Fold")
    print(f"{'=' * 70}")
    print(f"  Resolution: {cfg.image_size}px, Epochs: {cfg.epochs}, Seed: {cfg.seed}")
    print(f"  LoRA: rank={cfg.lora_rank}, alpha={cfg.lora_alpha}, "
          f"blocks={cfg.lora_last_n_blocks}, unfreeze={cfg.unfreeze_last_n_blocks}")
    print(f"  ArcFace: s={cfg.arcface_s}, m={cfg.arcface_m}, K={cfg.arcface_k}")
    print(f"  Mixup: prob={cfg.mixup_prob}, alpha={cfg.mixup_alpha}")
    print(f"  Jitter: b={cfg.jitter_brightness}, c={cfg.jitter_contrast}")
    print(f"  Ink features: {cfg.num_ink_features}D")
    print(f"  Batch: {cfg.batch_size} × accum {cfg.grad_accum_steps} = "
          f"{cfg.batch_size * cfg.grad_accum_steps} effective")

    data_dir = Path(cfg.data_dir)
    df = pd.read_csv(data_dir / cfg.train_csv)
    df_test = pd.read_csv(data_dir / cfg.test_csv)
    print(f"\nTrain: {len(df)}, Test: {len(df_test)}, "
          f"Writers: {df['writer_id'].nunique()}, Pens: {df['pen_id'].nunique()}")

    pens = sorted(df["pen_id"].unique())
    pen2idx = {p: i for i, p in enumerate(pens)}
    idx2pen = {i: p for p, i in pen2idx.items()}
    writers = sorted(df["writer_id"].unique())
    writer2idx = {w: i for i, w in enumerate(writers)}

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}")

    gkf = GroupKFold(n_splits=cfg.n_folds)
    folds = list(gkf.split(range(len(df)), df["pen_id"].values,
                            groups=df["writer_id"].values))

    # ── Training ────────────────────────────────────────────────────
    if not args.infer_only:
        fold_results = []
        folds_to_train = [args.fold] if args.fold is not None else range(cfg.n_folds)
        for fold_idx in folds_to_train:
            train_idx, val_idx = folds[fold_idx]
            acc, f1 = train_fold(fold_idx, train_idx, val_idx,
                                  df, data_dir, pen2idx, writer2idx, cfg, device)
            fold_results.append((fold_idx, acc, f1))

        print(f"\n{'=' * 70}")
        print("FOLD SUMMARY")
        print(f"{'=' * 70}")
        for fi, acc, f1 in fold_results:
            print(f"  Fold {fi + 1}: acc={acc:.4f}  f1={f1:.4f}")
        accs = [a for _, a, _ in fold_results]
        f1s = [f for _, _, f in fold_results]
        print(f"  Mean:   acc={np.mean(accs):.4f}  f1={np.mean(f1s):.4f}")

    # ── Submission ──────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("GENERATING SUBMISSION")
    print(f"{'=' * 70}")
    print(f"Test: {len(df_test)}, Crops: 1+{cfg.num_test_crops}")

    test_ds = PenTestDataset(df_test, data_dir, cfg.image_size, cfg.image_load_mode,
                              num_crops=cfg.num_test_crops,
                              crop_scale_range=(cfg.crop_scale_min, cfg.crop_scale_max))

    all_fold_logits = []
    for fold_idx in range(cfg.n_folds):
        print(f"\nInference fold {fold_idx + 1}/{cfg.n_folds}...")
        fold_logits = run_inference_fold(
            fold_idx, cfg, device, test_ds, len(writer2idx), use_amp)
        if fold_logits is not None:
            all_fold_logits.append(fold_logits)
            print(f"  Done, shape={fold_logits.shape}")

    if not all_fold_logits:
        print("ERROR: No checkpoints found.")
        sys.exit(1)

    final_logits = np.mean(all_fold_logits, axis=0)
    pen_preds = final_logits.argmax(axis=1)
    p_orig = [idx2pen[int(p)] for p in pen_preds]

    print(f"\nEnsembled {len(all_fold_logits)} folds")
    print(f"Distribution: {Counter(p_orig)}")

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({"image_id": df_test["image_id"], "pen_id": p_orig})
    out_path = Path(cfg.output_dir) / "submission_pen_v6b.csv"
    sub.to_csv(out_path, index=False)
    print(sub.head(10))
    print(f"Saved {out_path}")
    print("\nDone.")
