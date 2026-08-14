"""
CircleID — Pen Classification v3
DINOv2 + LoRA (expanded) + Simple Resize + Pen Contrastive + Pen-Axis Sampler

"""

# ══════════════════════════════════════════════════════════════════════════
# 1. Setup & Config
# ══════════════════════════════════════════════════════════════════════════

import os
import sys
import time
import math
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = Path(__file__).resolve().parent.parent  
DINOV2_REPO_DIR = BASE_DIR / "dinov2_repo"
DINOV2_WEIGHTS_PATH = BASE_DIR / "dinov2_vitb14_reg4_pretrain.pth"


def _setup_dinov2():
    """Download DINOv2 and cache locally (only called from main process)."""
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
            print(f"Copied DINOv2 repo to {DINOV2_REPO_DIR}")

        if weights_src.exists() and not DINOV2_WEIGHTS_PATH.exists():
            shutil.copy(str(weights_src), str(DINOV2_WEIGHTS_PATH))
            print(f"Copied DINOv2 weights to {DINOV2_WEIGHTS_PATH}")
    else:
        print(f"DINOv2 repo found at {DINOV2_REPO_DIR}")
        print(f"DINOv2 weights found at {DINOV2_WEIGHTS_PATH}")


@dataclass
class PenConfig:
    data_dir: Path = BASE_DIR / "icdar-2026-circleid-pen-classification"
    train_csv: str = "train.csv"
    test_csv: str = "test.csv"
    output_dir: Path = BASE_DIR / "outputs"
    checkpoint_dir: Path = BASE_DIR / "checkpoints_contrastive"
    dinov2_repo: str = str(DINOV2_REPO_DIR)
    dinov2_weights: str = str(DINOV2_WEIGHTS_PATH)
    num_pens: int = 8
    image_size: int = 224
    image_load_mode: str = "cv2"
    dinov2_variant: str = "dinov2_vitb14_reg"
    dinov2_embed_dim: int = 768
    # ── LoRA EXPANDED ───────────────────────────────────────────────
    lora_rank: int = 16               # was 8
    lora_alpha: int = 32              # was 16
    lora_last_n_blocks: int = 6       # was 4
    # ── Architecture ────────────────────────────────────────────────
    projection_dim: int = 512
    pen_embed_dim: int = 128
    head_dropout: float = 0.15
    # ── Training ────────────────────────────────────────────────────
    epochs: int = 50
    batch_size: int = 32
    grad_accum_steps: int = 4
    lr_lora: float = 1e-4
    lr_heads: float = 5e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    label_smoothing: float = 0.0
    val_ratio: float = 0.1
    # ── Pen Contrastive Loss ────────────────────────────────────────
    lambda_contrast: float = 0.5
    contrast_tau: float = 0.07
    contrast_warmup_end: int = 5
    contrast_ramp_end: int = 15
    # ── Writer Adversarial (Gradient Reversal) ─────────────────────
    lambda_adv: float = 0.1
    adv_ramp_end: int = 20
    # ── EMA ──────────────────────────────────────────────────────────
    use_ema: bool = True
    ema_decay: float = 0.999
    ema_start_epoch: int = 5
    # ── TTA ──────────────────────────────────────────────────────────
    tta_variants: int = 3
    # ── Augmentation ─────────────────────────────────────────────────
    jitter_brightness: float = 0.3
    jitter_contrast: float = 0.3
    random_rotation_deg: float = 15.0
    crop_scale_min: float = 0.5           
    crop_scale_max: float = 1.0
    num_test_crops: int = 16             
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
# 2. Preprocessing (simplified)
# ══════════════════════════════════════════════════════════════════════════

def to_gray(image, mode="cv2"):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if mode == "cv2" else cv2.COLOR_RGB2GRAY)


def compute_ink_mask(image, mode="cv2"):
    """Otsu + adaptive fallback. Returns mask_float [H,W] in [0,1]."""
    gray = to_gray(image, mode)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if mask.mean() < 5 or mask.mean() > 250:
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
    return mask.astype(np.float32) / 255.0


def apply_intensity_jitter(image, brightness_range=0.3, contrast_range=0.3, rng=None):
    if rng is None: rng = np.random.default_rng()
    bf = 1.0 + rng.uniform(-brightness_range, brightness_range)
    cf = 1.0 + rng.uniform(-contrast_range, contrast_range)
    img = image.astype(np.float32)
    mean = img.mean()
    img = ((img - mean) * cf + mean) * bf
    return np.clip(img, 0, 255).astype(np.uint8)


def apply_random_rotation(image, max_deg=15.0, rng=None):
    if rng is None: rng = np.random.default_rng()
    angle = rng.uniform(-max_deg, max_deg)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(255, 255, 255))


def apply_random_scale_crop(image, scale_range=(0.15, 0.60), rng=None):
    
    if rng is None: rng = np.random.default_rng()
    h, w = image.shape[:2]
    scale = rng.uniform(*scale_range)
    crop_h, crop_w = int(h * scale), int(w * scale)
    if crop_h >= h or crop_w >= w:
        return image

    # Bias crop toward ink regions (dark pixels)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    ink_mask = (gray < 200).astype(np.float32) 

   
    ink_coords = np.where(ink_mask > 0)
    if len(ink_coords[0]) > 50:
        
        idx = rng.integers(0, len(ink_coords[0]))
        cy, cx = ink_coords[0][idx], ink_coords[1][idx]
        top = max(0, min(cy - crop_h // 2, h - crop_h))
        left = max(0, min(cx - crop_w // 2, w - crop_w))
    else:
        
        top = rng.integers(0, h - crop_h + 1)
        left = rng.integers(0, w - crop_w + 1)

    cropped = image[top:top+crop_h, left:left+crop_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)


def apply_random_erasing(img_tensor, p=0.5, scale_range=(0.02, 0.2), ratio_range=(0.3, 3.3), rng=None):
    """Random erasing on tensor: fills a random rectangle with mean pixel value.
    Destroys circle/character shape, forces texture learning from remaining area."""
    if rng is None: rng = np.random.default_rng()
    if rng.random() > p:
        return img_tensor
    c, h, w = img_tensor.shape
    area = h * w
    for _ in range(10):
        target_area = rng.uniform(*scale_range) * area
        aspect = rng.uniform(*ratio_range)
        eh = int(round((target_area * aspect) ** 0.5))
        ew = int(round((target_area / aspect) ** 0.5))
        if eh < h and ew < w:
            top = rng.integers(0, h - eh)
            left = rng.integers(0, w - ew)
            img_tensor[:, top:top+eh, left:left+ew] = img_tensor.mean()
            break
    return img_tensor


def preprocess_image(image, mode="cv2", image_size=224,
                     apply_jitter=False, jitter_brightness=0.3, jitter_contrast=0.3,
                     rotation_deg=0.0, random_scale_crop=False,
                     crop_scale_range=(0.5, 1.0), rng=None):
    """
    Simplified pipeline: (scale_crop) -> resize -> (rotate) -> (jitter) -> compute mask.
    NO ink crop. Full image preserved.
    """
    if random_scale_crop:
        image = apply_random_scale_crop(image, scale_range=crop_scale_range, rng=rng)

    # Images are ~100px natively — always upscaling, so use LANCZOS4
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
    mask_patches = F.adaptive_avg_pool2d(mask_float.unsqueeze(1), (grid_size, grid_size)).flatten(1)
    mask_sum = mask_patches.sum(dim=1, keepdim=True)
    uniform = torch.ones_like(mask_patches) / mask_patches.shape[1]
    use_uniform = (mask_sum < eps).float()
    mask_patches = (1 - use_uniform) * mask_patches + use_uniform * uniform
    weights = mask_patches / (mask_patches.sum(dim=1, keepdim=True) + eps)
    pooled = (patch_tokens * weights.unsqueeze(-1)).sum(dim=1)
    return pooled


# ══════════════════════════════════════════════════════════════════════════
# 3. Dataset
# ══════════════════════════════════════════════════════════════════════════

class PenDataset(Dataset):
    def __init__(self, df, indices, data_dir, pen2idx, writer2idx,
                 image_size=224, image_load_mode="cv2", is_training=False,
                 jitter_brightness=0.3, jitter_contrast=0.3, rotation_deg=0.0,
                 crop_scale_range=(0.5, 1.0)):
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
        self.crop_scale_range = crop_scale_range
        self.rng = np.random.default_rng()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        row = self.df.iloc[self.indices[idx]]
        img_path = self.data_dir / row["image_path"]
        if self.mode == "cv2":
            image = cv2.imread(str(img_path))
            if image is None: raise FileNotFoundError(f"Failed to load: {img_path}")
        else:
            from PIL import Image
            image = np.array(Image.open(img_path).convert("RGB"))

        final_image, mask_float = preprocess_image(
            image, mode=self.mode, image_size=self.image_size,
            apply_jitter=self.is_training,
            jitter_brightness=self.jitter_brightness,
            jitter_contrast=self.jitter_contrast,
            rotation_deg=self.rotation_deg if self.is_training else 0.0,
            random_scale_crop=self.is_training,
            crop_scale_range=self.crop_scale_range,
            rng=self.rng)

        img_tensor = torch.from_numpy(final_image).float().permute(2, 0, 1) / 255.0
        if self.mode == "cv2":
            img_tensor = img_tensor[[2, 1, 0], ...]
        img_tensor = (img_tensor - self.mean) / self.std
        if self.is_training:
            img_tensor = apply_random_erasing(img_tensor, p=0.5, rng=self.rng)
        mask_tensor = torch.from_numpy(mask_float).float()
        pen_label = self.pen2idx[row["pen_id"]]
        writer_id = self.writer2idx[row["writer_id"]]
        return {"image": img_tensor, "mask": mask_tensor,
                "pen_label": pen_label, "writer_id": writer_id}


class PenTestDataset(Dataset):
    """Test dataset with multi-crop support.
    num_crops=0 returns full image (original behavior).
    num_crops>0 returns N random ink-biased crops + 1 full image = num_crops+1 views."""
    def __init__(self, df, data_dir, image_size=224, image_load_mode="cv2",
                 num_crops=0, crop_scale_range=(0.5, 1.0)):
        self.df = df
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.mode = image_load_mode
        self.num_crops = num_crops
        self.crop_scale_range = crop_scale_range
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self): return len(self.df)

    def _to_tensor(self, image, mask_float):
        img_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        if self.mode == "cv2":
            img_tensor = img_tensor[[2, 1, 0], ...]
        img_tensor = (img_tensor - self.mean) / self.std
        mask_tensor = torch.from_numpy(mask_float).float()
        return img_tensor, mask_tensor

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.data_dir / row["image_path"]
        if self.mode == "cv2":
            image = cv2.imread(str(img_path))
            if image is None: raise FileNotFoundError(f"Failed to load: {img_path}")
        else:
            from PIL import Image
            image = np.array(Image.open(img_path).convert("RGB"))

        if self.num_crops == 0:
            final_image, mask_float = preprocess_image(
                image, mode=self.mode, image_size=self.image_size, apply_jitter=False)
            img_t, mask_t = self._to_tensor(final_image, mask_float)
            return {"image": img_t, "mask": mask_t}

        # Multi-crop: full image + N random ink crops
        rng = np.random.default_rng()
        imgs, masks = [], []

        # Full image view
        full_img, full_mask = preprocess_image(
            image, mode=self.mode, image_size=self.image_size, apply_jitter=False)
        img_t, mask_t = self._to_tensor(full_img, full_mask)
        imgs.append(img_t)
        masks.append(mask_t)

        # Random ink-biased crops
        for _ in range(self.num_crops):
            cropped = apply_random_scale_crop(image, self.crop_scale_range, rng)
            crop_img, crop_mask = preprocess_image(
                cropped, mode=self.mode, image_size=self.image_size, apply_jitter=False)
            img_t, mask_t = self._to_tensor(crop_img, crop_mask)
            imgs.append(img_t)
            masks.append(mask_t)

        return {"image": torch.stack(imgs), "mask": torch.stack(masks)}


# ══════════════════════════════════════════════════════════════════════════
# 4. Pen-Axis Batch Sampler
# ══════════════════════════════════════════════════════════════════════════

class PenAxisBatchSampler(Sampler):
    def __init__(self, df, indices, pen2idx, batch_size=32, seed=42):
        self.indices = indices
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.pen_to_writers = {}
        self.pw_to_idx = {}
        for i, global_idx in enumerate(indices):
            row = df.iloc[global_idx]
            p = pen2idx[row["pen_id"]]
            w = row["writer_id"]
            self.pen_to_writers.setdefault(p, set()).add(w)
            self.pw_to_idx.setdefault((p, w), []).append(i)
        self.all_pens = list(self.pen_to_writers.keys())
        avg_w = np.mean([len(ws) for ws in self.pen_to_writers.values()])
        print(f"PenAxisSampler: {len(self.all_pens)} pens, avg {avg_w:.1f} writers/pen")

    def __iter__(self):
        n_batches = len(self.indices) // self.batch_size
        for _ in range(n_batches):
            batch = []
            used = set()
            n_seed = min(3, len(self.all_pens))
            seed_pens = self.rng.choice(self.all_pens, n_seed, replace=False)
            for pen in seed_pens:
                writers = list(self.pen_to_writers[pen])
                n_w = min(len(writers), self.batch_size // n_seed)
                if n_w == 0: continue
                sel_writers = self.rng.choice(writers, n_w, replace=False)
                for w in sel_writers:
                    if (pen, w) in self.pw_to_idx and (pen, w) not in used:
                        idx = self.rng.choice(self.pw_to_idx[(pen, w)])
                        batch.append(idx)
                        used.add((pen, w))
                    if len(batch) >= self.batch_size: break
                if len(batch) >= self.batch_size: break
            while len(batch) < self.batch_size:
                batch.append(self.rng.integers(0, len(self.indices)))
            yield batch[:self.batch_size]

    def __len__(self):
        return len(self.indices) // self.batch_size


# ══════════════════════════════════════════════════════════════════════════
# 5. Pen Contrastive Loss
# ══════════════════════════════════════════════════════════════════════════

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
        loss = -(num - den).mean()
    else:
        loss = torch.tensor(0.0, device=device)
    return loss, {"pen_positives": pos_mask.sum().item(), "pen_orphans": (~has_pos).sum().item()}


def get_lambda_contrast(epoch, target=0.5, warmup_end=5, ramp_end=15):
    if epoch < warmup_end: return 0.0
    elif epoch < ramp_end: return target * (epoch - warmup_end) / (ramp_end - warmup_end)
    else: return target


# ══════════════════════════════════════════════════════════════════════════
# 5b. Gradient Reversal Layer (Writer Adversarial)
# ══════════════════════════════════════════════════════════════════════════

class GradientReversal(torch.autograd.Function):
    """Reverses gradient flow — backbone learns to NOT encode writer identity."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


# ══════════════════════════════════════════════════════════════════════════
# 6. Model
# ══════════════════════════════════════════════════════════════════════════

class LoRALinear(nn.Module):
    def __init__(self, original, rank=16, alpha=32):
        super().__init__()
        self.original = original
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(original.in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, original.out_features))
        original.weight.requires_grad = False
        if original.bias is not None: original.bias.requires_grad = False

    def forward(self, x):
        return self.original(x) + (x @ self.lora_A @ self.lora_B) * self.scaling


def apply_lora_to_dinov2(model, rank=16, alpha=32, last_n_blocks=6):
    blocks = model.blocks
    n = len(blocks)
    lora_params = []
    for i in range(n - last_n_blocks, n):
        attn = blocks[i].attn
        lora_qkv = LoRALinear(attn.qkv, rank, alpha)
        attn.qkv = lora_qkv
        lora_params.extend([lora_qkv.lora_A, lora_qkv.lora_B])
    return model, lora_params


class PenClassifierModel(nn.Module):
    def __init__(self, backbone, num_pens=8, num_writers=44, dinov2_dim=768,
                 projection_dim=512, pen_embed_dim=128, dropout=0.15):
        super().__init__()
        self.backbone = backbone
        self.projection = nn.Sequential(
            nn.Linear(dinov2_dim * 2, projection_dim),
            nn.LayerNorm(projection_dim), nn.GELU())
        self.pen_proj = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim), nn.GELU(),
            nn.Linear(projection_dim, pen_embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.pen_classifier = nn.Linear(pen_embed_dim, num_pens)
        # Writer adversarial head — GRL reverses gradients to backbone
        self.writer_adversary = nn.Sequential(
            nn.Linear(projection_dim, projection_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim // 2, num_writers))

    def forward(self, images, masks, eps=1e-8, return_emb=False, adv_alpha=0.0):
        with torch.set_grad_enabled(self.training):
            out = self.backbone.forward_features(images)
        cls_token = out["x_norm_clstoken"]
        patch_tokens = out["x_norm_patchtokens"]
        patch_avg = ink_weighted_pooling(patch_tokens, masks, eps)
        combined = torch.cat([cls_token, patch_avg], dim=1)
        projected = self.projection(combined)
        pen_emb = F.normalize(self.pen_proj(projected), p=2, dim=1)
        logits = self.pen_classifier(self.dropout(pen_emb))
        if return_emb:
            reversed_feat = GradientReversal.apply(projected, adv_alpha)
            writer_logits = self.writer_adversary(reversed_feat)
            return logits, pen_emb, writer_logits
        return logits


class EMAModel:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

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
            if n in self.backup: p.data.copy_(self.backup[n])
        self.backup = {}


def build_pen_model(cfg, num_writers=44):
    backbone = torch.hub.load(cfg.dinov2_repo, cfg.dinov2_variant, source="local", pretrained=False)
    state = torch.load(cfg.dinov2_weights, map_location="cpu", weights_only=False)
    backbone.load_state_dict(state, strict=True)
    backbone.eval()
    with torch.no_grad():
        out = backbone.forward_features(torch.randn(1, 3, cfg.image_size, cfg.image_size))
    assert "x_norm_clstoken" in out and "x_norm_patchtokens" in out
    print(f"DINOv2: CLS {out['x_norm_clstoken'].shape}, patches {out['x_norm_patchtokens'].shape}")
    for p in backbone.parameters(): p.requires_grad = False
    backbone, lora_params = apply_lora_to_dinov2(backbone, cfg.lora_rank, cfg.lora_alpha, cfg.lora_last_n_blocks)
    print(f"LoRA: {sum(p.numel() for p in lora_params):,} params in last {cfg.lora_last_n_blocks} blocks")
    # Unfreeze last 2 blocks entirely for more capacity (skip LoRA params to avoid duplicates)
    lora_ids = {id(p) for p in lora_params}
    unfreeze_params = []
    for i in range(len(backbone.blocks) - 2, len(backbone.blocks)):
        for p in backbone.blocks[i].parameters():
            p.requires_grad = True
            if id(p) not in lora_ids:
                unfreeze_params.append(p)
    print(f"Unfrozen last 2 blocks: {sum(p.numel() for p in unfreeze_params):,} extra params (excl LoRA)")
    model = PenClassifierModel(backbone, cfg.num_pens, num_writers, cfg.dinov2_embed_dim,
                                cfg.projection_dim, cfg.pen_embed_dim, cfg.head_dropout)
    all_backbone_ids = {id(p) for p in lora_params} | {id(p) for p in unfreeze_params}
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and id(p) not in all_backbone_ids]
    lora_params = lora_params + unfreeze_params
    total_t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_a = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_t:,} trainable / {total_a:,} total ({100*total_t/total_a:.2f}%)")
    return model, lora_params, head_params


# ══════════════════════════════════════════════════════════════════════════
# 7. Training
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


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, cfg, ema=None, scaler=None):
    model.train()
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    lam = get_lambda_contrast(epoch, cfg.lambda_contrast, cfg.contrast_warmup_end, cfg.contrast_ramp_end)
    tot_loss = tot_ce = tot_con = tot_adv = c_p = c_w = n = 0
    n_orphan_batches = 0
    adv_alpha = get_lambda_contrast(epoch, target=1.0, warmup_end=cfg.contrast_warmup_end, ramp_end=cfg.adv_ramp_end)
    optimizer.zero_grad()
    num_batches = len(loader)
    for step, batch in enumerate(loader):
        imgs   = batch["image"].to(device)
        masks  = batch["mask"].to(device)
        labels = batch["pen_label"].to(device)
        wids   = batch["writer_id"].to(device)
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            logits, pen_emb, writer_logits = model(imgs, masks, cfg.eps, return_emb=True, adv_alpha=adv_alpha)
            loss_ce = ce_criterion(logits, labels)
            if lam > 0:
                loss_con, diag = pen_contrastive_loss(pen_emb, labels, wids, cfg.contrast_tau)
                if diag["pen_positives"] == 0: n_orphan_batches += 1
            else:
                loss_con = torch.tensor(0.0, device=device)
            loss_adv = F.cross_entropy(writer_logits, wids)
            loss = loss_ce + lam * loss_con + cfg.lambda_adv * loss_adv
        scaled_loss = loss / cfg.grad_accum_steps
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        do_step = ((step + 1) % cfg.grad_accum_steps == 0) or ((step + 1) == num_batches)
        if do_step:
            if scaler is not None:
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if ema and epoch >= cfg.ema_start_epoch: ema.update(model)
        B = imgs.size(0)
        tot_loss += loss.item() * B
        tot_ce += loss_ce.item() * B
        tot_con += loss_con.item() * B
        tot_adv += loss_adv.item() * B
        c_p += (logits.argmax(1) == labels).sum().item()
        c_w += (writer_logits.argmax(1) == wids).sum().item()
        n += B
    return {"loss": tot_loss/n, "ce": tot_ce/n, "con": tot_con/n, "adv": tot_adv/n,
            "pen_acc": c_p/n, "writer_acc": c_w/n, "lam": lam, "adv_alpha": adv_alpha,
            "orphan_batches": n_orphan_batches}


@torch.no_grad()
def evaluate(model, loader, device, eps=1e-8, use_amp=False):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(batch["image"].to(device), batch["mask"].to(device), eps)
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(batch["pen_label"].numpy())
    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return {"acc": accuracy_score(labels, preds), "f1": f1_score(labels, preds, average="macro")}


# ══════════════════════════════════════════════════════════════════════════
# 8. Main — Data Loading & Training
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="Override seed for ensemble training")
    args = parser.parse_args()

    _setup_dinov2()

    cfg = PenConfig()
    if args.seed is not None:
        cfg.seed = args.seed
        cfg.checkpoint_dir = BASE_DIR / f"checkpoints_seed{cfg.seed}"
        cfg.output_dir = BASE_DIR / f"outputs_seed{cfg.seed}"
    print(f"Config: {cfg.dinov2_variant}, epochs={cfg.epochs}, num_pens={cfg.num_pens}, seed={cfg.seed}")
    print(f"  LoRA: rank={cfg.lora_rank}, alpha={cfg.lora_alpha}, blocks={cfg.lora_last_n_blocks}")
    print(f"  Augmentation: jitter={cfg.jitter_brightness}, rotation={cfg.random_rotation_deg}deg")
    print(f"  Contrastive: lambda={cfg.lambda_contrast}, tau={cfg.contrast_tau}")
    print(f"  Writer Adversarial: lambda_adv={cfg.lambda_adv}, ramp_end={cfg.adv_ramp_end}")
    set_seed(cfg.seed)

    data_dir = Path(cfg.data_dir)
    df = pd.read_csv(data_dir / cfg.train_csv)
    print(f"Dataset: {df.shape[0]} images, Writers: {df['writer_id'].nunique()}, Pens: {df['pen_id'].nunique()}")

    pens = sorted(df["pen_id"].unique())
    pen2idx = {p: i for i, p in enumerate(pens)}
    idx2pen = {i: p for p, i in pen2idx.items()}

    writers = sorted(df["writer_id"].unique())
    writer2idx = {w: i for i, w in enumerate(writers)}
    print(f"Pen map: {pen2idx}")
    print(f"Writers: {len(writer2idx)}")

    print("\nPen distribution:")
    for pen, cnt in df["pen_id"].value_counts().sort_index().items():
        print(f"  pen {pen}: {cnt} ({100*cnt/len(df):.1f}%)")

    # ── Train/Val Split (WRITER-DISJOINT) ────────────────────────────────
    set_seed(cfg.seed)
    all_idx = list(range(len(df)))
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.val_ratio, random_state=cfg.seed)
    train_rel, val_rel = next(gss.split(all_idx, df["pen_id"].values, groups=df["writer_id"].values))
    train_writers = set(df.iloc[train_rel]["writer_id"].unique())
    val_writers = set(df.iloc[val_rel]["writer_id"].unique())
    print(f"Train: {len(train_rel)} ({len(train_writers)} writers), Val: {len(val_rel)} ({len(val_writers)} writers)")
    print(f"Writer overlap: {len(train_writers & val_writers)} (should be 0)")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    kw = dict(num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    train_ds = PenDataset(df, list(train_rel), data_dir, pen2idx, writer2idx,
                           cfg.image_size, image_load_mode=cfg.image_load_mode,
                           is_training=True,
                           jitter_brightness=cfg.jitter_brightness,
                           jitter_contrast=cfg.jitter_contrast,
                           rotation_deg=cfg.random_rotation_deg,
                           crop_scale_range=(cfg.crop_scale_min, cfg.crop_scale_max))
    val_ds = PenDataset(df, list(val_rel), data_dir, pen2idx, writer2idx,
                         cfg.image_size, image_load_mode=cfg.image_load_mode,
                         is_training=False)

    train_sampler = PenAxisBatchSampler(df, list(train_rel), pen2idx,
                                         batch_size=cfg.batch_size, seed=cfg.seed)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, **kw)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, **kw)

    model, lora_params, head_params = build_pen_model(cfg, num_writers=len(writer2idx))
    model = model.to(device)
    steps_per_epoch = math.ceil(len(train_sampler) / cfg.grad_accum_steps)
    optimizer, scheduler = build_optimizer_scheduler(lora_params, head_params, cfg, steps_per_epoch)
    ema = EMAModel(model, cfg.ema_decay) if cfg.use_ema else None
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    use_amp = scaler is not None
    print(f"Mixed precision (AMP): {'ON' if use_amp else 'OFF'}")

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(cfg.epochs):
        t0 = time.time()
        tm = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch, cfg, ema, scaler)
        # Evaluate with EMA weights when available
        if ema and epoch >= cfg.ema_start_epoch:
            ema.apply_shadow(model)
        val_metrics = evaluate(model, val_loader, device, cfg.eps, use_amp)
        if ema and epoch >= cfg.ema_start_epoch:
            ema.restore(model)
        elapsed = time.time() - t0
        ema_tag = " (EMA)" if ema and epoch >= cfg.ema_start_epoch else ""
        print(f"  E{epoch+1:02d} ({elapsed:.0f}s) L={tm['loss']:.4f} CE={tm['ce']:.4f} "
              f"Con={tm['con']:.4f}(lam={tm['lam']:.2f}) "
              f"Adv={tm['adv']:.4f}(a={tm['adv_alpha']:.2f}) "
              f"P={tm['pen_acc']:.3f} W={tm['writer_acc']:.3f} | "
              f"val_P={val_metrics['acc']:.3f} F1={val_metrics['f1']:.3f}"
              f"{ema_tag}{' *' if val_metrics['acc'] > best_acc else ''}")
        if tm['orphan_batches'] > 0:
            print(f"    warning: {tm['orphan_batches']} batches had no contrastive positives")
        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            if ema and epoch >= cfg.ema_start_epoch:
                ema.apply_shadow(model)
                torch.save(model.state_dict(), Path(cfg.checkpoint_dir) / "best_pen_model.pt")
                ema.restore(model)
            else:
                torch.save(model.state_dict(), Path(cfg.checkpoint_dir) / "best_pen_model.pt")

    print(f"\nBest val acc: {best_acc:.4f}")

    # ── 9. Submission ───────────────────────────────────────────────────
    set_seed(cfg.seed)
    full_ds = PenDataset(df, list(range(len(df))), data_dir, pen2idx, writer2idx,
                          cfg.image_size, image_load_mode=cfg.image_load_mode,
                          is_training=True,
                          jitter_brightness=cfg.jitter_brightness,
                          jitter_contrast=cfg.jitter_contrast,
                          rotation_deg=cfg.random_rotation_deg,
                          crop_scale_range=(cfg.crop_scale_min, cfg.crop_scale_max))
    full_sampler = PenAxisBatchSampler(df, list(range(len(df))), pen2idx,
                                        batch_size=cfg.batch_size, seed=cfg.seed)
    full_loader = DataLoader(full_ds, batch_sampler=full_sampler,
                              num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)

    model_final, lora_f, head_f = build_pen_model(cfg, num_writers=len(writer2idx))
    ckpt_path = Path(cfg.checkpoint_dir) / "best_pen_ema.pt"
    if not ckpt_path.exists():
        ckpt_path = Path(cfg.checkpoint_dir) / "best_pen_model.pt"
    model_final.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
    model_final = model_final.to(device)
    print(f"Loaded checkpoint: {ckpt_path}")

    finetune_epochs = 5
    steps_f = math.ceil(len(full_sampler) / cfg.grad_accum_steps)
    optimizer_f = torch.optim.AdamW([
        {"params": lora_f, "lr": cfg.lr_lora * 0.1},
        {"params": head_f, "lr": cfg.lr_heads * 0.1},
    ], weight_decay=cfg.weight_decay)
    cosine_f = CosineAnnealingLR(optimizer_f, T_max=finetune_epochs * steps_f)
    ema_f = EMAModel(model_final, cfg.ema_decay)
    scaler_f = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    ce_criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    for epoch in range(finetune_epochs):
        model_final.train()
        tot_loss = c_p = n = 0
        optimizer_f.zero_grad()
        num_ft_batches = len(full_loader)
        for step, batch in enumerate(full_loader):
            imgs   = batch["image"].to(device)
            masks  = batch["mask"].to(device)
            labels = batch["pen_label"].to(device)
            wids   = batch["writer_id"].to(device)
            with torch.amp.autocast("cuda", enabled=scaler_f is not None):
                logits, pen_emb, writer_logits = model_final(imgs, masks, cfg.eps, return_emb=True, adv_alpha=1.0)
                loss_ce = ce_criterion(logits, labels)
                loss_con, _ = pen_contrastive_loss(pen_emb, labels, wids, cfg.contrast_tau)
                loss_adv = F.cross_entropy(writer_logits, wids)
                loss = loss_ce + cfg.lambda_contrast * loss_con + cfg.lambda_adv * loss_adv
            scaled_loss = loss / cfg.grad_accum_steps
            if scaler_f is not None:
                scaler_f.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            do_step = ((step + 1) % cfg.grad_accum_steps == 0) or ((step + 1) == num_ft_batches)
            if do_step:
                if scaler_f is not None:
                    scaler_f.unscale_(optimizer_f)
                nn.utils.clip_grad_norm_([p for p in model_final.parameters() if p.requires_grad], cfg.grad_clip)
                if scaler_f is not None:
                    scaler_f.step(optimizer_f)
                    scaler_f.update()
                else:
                    optimizer_f.step()
                cosine_f.step()
                optimizer_f.zero_grad()
                ema_f.update(model_final)
            B = imgs.size(0)
            tot_loss += loss.item() * B
            c_p += (logits.argmax(1) == labels).sum().item()
            n += B
        print(f"  FT E{epoch+1}/{finetune_epochs} L={tot_loss/n:.4f} P={c_p/n:.3f}")

    ema_f.apply_shadow(model_final)
    model_final.eval()

    df_test = pd.read_csv(data_dir / cfg.test_csv)
    print(f"Test size: {len(df_test)}")

   
    print(f"Multi-crop inference: 1 full + {cfg.num_test_crops} crops per image")
    test_ds = PenTestDataset(df_test, data_dir, cfg.image_size, cfg.image_load_mode,
                              num_crops=cfg.num_test_crops,
                              crop_scale_range=(cfg.crop_scale_min, cfg.crop_scale_max))
   
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)

    all_logits = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # batch["image"] shape: [1, num_crops+1, C, H, W]
            imgs = batch["image"].squeeze(0).to(device)    # [K, C, H, W]
            masks = batch["mask"].squeeze(0).to(device)     # [K, H, W]
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model_final(imgs, masks, cfg.eps)  # [K, num_pens]
            # Average across all crops
            avg_logits = logits.mean(dim=0, keepdim=True)   # [1, num_pens]
            all_logits.append(avg_logits.cpu().numpy())
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(df_test)} images done")
    print(f"  Multi-crop inference done ({cfg.num_test_crops + 1} views/image)")

    final_logits = np.concatenate(all_logits, axis=0)
    pen_preds = final_logits.argmax(axis=1)
    p_orig = [idx2pen[int(p)] for p in pen_preds]

    print(f"\nPredicted pen distribution: {Counter(p_orig)}")

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({"image_id": df_test["image_id"], "pen_id": p_orig})
    submission.to_csv(Path(cfg.output_dir) / "submission_pen.csv", index=False)
    print(submission.head(10))
    print("Saved submission_pen.csv")
