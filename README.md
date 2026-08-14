# ICDAR 2026 CircleID — Pen Classification

Solution for the [**ICDAR 2026 CircleID Pen Classification**](https://www.kaggle.com/competitions/icdar-2026-circleid-pen-classification)
challenge: given a small image of a hand-drawn circle, identify which of 8 pens drew it.

**Final result: 9th place.** Private LB 0.91948, public LB 0.94202 — a rank-averaged ensemble
of 5 CNN/transformer backbones. 4th on the public leaderboard before the private split was
revealed.

The 0.023 drop from public to private is the most instructive number in this repo. The
out-of-fold CV of the submitted ensemble was **0.9202**, and the private LB came in at
**0.91948** — within 0.001. Cross-validation predicted the real score almost exactly; the public
leaderboard, on 30% of the test set, was the outlier. See
[what the private LB revealed](#what-the-private-lb-revealed).

---

## The problem in one picture

Each image is a ~100×100 px hand-drawn circle. Two pens can be told apart only by ink colour,
stroke width and micro-texture — not by shape. Five of the eight pens separate cleanly on ink
intensity alone. **Pens 3 and 7 are visually near-identical** and account for essentially all
remaining error:

| Pen | Mean intensity | Stroke width | R−G diff | Saturation |
|-----|---------------|--------------|----------|------------|
| 6 | 64 (darkest) | 2.21 | — | — |
| 2 | 83 | 2.11 | — | — |
| 1 | 112 | 2.16 | — | — |
| **3** | **131.5** | **2.00** | **10.48** | **21.17** |
| **7** | **131.3** | **2.00** | **10.01** | **20.30** |
| **8** | **132.2** | **1.78** | **7.64** | **16.16** |
| 4 | 148 | 1.73 | — | — |
| 5 | 177 (lightest) | 1.69 | — | — |

Final per-pen accuracy: pens 1, 2, 4, 5 ≈ 100%, pen 6 ≈ 98%, pen 8 ≈ 95%,
**pen 3 ≈ 68%, pen 7 ≈ 70%**. See [`docs/BOTTLENECK.md`](docs/BOTTLENECK.md).

---

## Results

### Individual models (5-fold writer-disjoint CV)

| Tag | Backbone | Params | Loss | OOF acc | OOF macro-F1 | Public LB |
|-----|----------|--------|------|---------|--------------|-----------|
| `v14` | ConvNeXt-V2 Base | 89M | Focal γ=2 | 0.9099 | 0.9113 | 0.9303 |
| `v9`  | ConvNeXt-V2 Tiny | 28M | Focal γ=2 | 0.9094 | 0.9105 | 0.9281 |
| `v11` | Swin-Base | 87M | CE + LS | 0.9070 | 0.9085 | 0.9231 |
| `v8`  | CAFormer-S36 | 37M | Focal γ=2 | 0.9102 | 0.9117 | 0.9197 |
| `v5b` | DINOv2 ViT-B + LoRA + 75 ink features | 86M | ArcFace | **0.9131** | **0.9155** | 0.9025 |

Note how CV and LB rank the models differently — `v5b` has the best CV and the worst LB, yet it
carries the largest ensemble weight. It is the only model with a genuinely different feature
space (handcrafted ink statistics), so it corrects the CNNs where they agree with each other and
are wrong.

### Ensemble

| Method | Weights (v14 / v11 / v8 / v9 / v5b) | Public LB | Private LB |
|--------|-------------------------------------|-----------|------------|
| Probability blend, LB-tuned | 50 / 25 / 15 / — / 10 | 0.93589 | — |
| Probability blend, CV-tuned | 20 / 20 / 30 / 10 / 20 | 0.93143 | — |
| LightGBM stacking on OOF | — | 0.91973 | — |
| Rank average, hand weights | 25 / 20 / 20 / 20 / 15 | 0.94091 | — |
| **Rank average, OOF-optimized** | **15 / 15 / 20 / 20 / 30** | **0.94202** | **0.91948** |

Only the submitted ensemble was scored on the private split; the rest were never selected, so
their private scores are unknown.

**Rank averaging was worth +0.006 LB over the best probability blend.** The five models are
miscalibrated relative to each other, so probability-space averaging lets one overconfident
model dominate. Converting each model's logits to per-class ranks throws away magnitude and
keeps only ordering.

Stacking had the *best* OOF (0.9268) and the *worst* LB (0.91973) — a textbook meta-learner
overfit on 22,850 OOF rows.

### What the private LB revealed

The public leaderboard scored 30% of the test set; the private leaderboard, released at the end,
scored the other 70%.

| | OOF (CV) | Public LB (30%) | Private LB (70%) |
|---|---|---|---|
| Submitted ensemble | 0.9202 | 0.94202 | **0.91948** |
| | | 4th place | **9th place** |

**CV was honest and the public LB was not.** OOF accuracy landed within 0.001 of the private
score, while the public LB sat 0.023 above it. Five places on the final leaderboard were lost to
a 30% sample that flattered every team that tuned against it.

The warning signs were visible before the private split was revealed, and are preserved in
[`docs/RESULTS.md`](docs/RESULTS.md): the weight vector with the best public LB (`v14:0.50,
v11:0.25, v8:0.15, v5b:0.10`) had the *worst* CV of every candidate, at 0.9171 against 0.9224
for the CV-optimal weights. That inversion was correctly read at the time as public-split
overfitting, and the final submission used OOF-optimized weights instead of the LB-optimized
ones. That decision was right — it just was not enough to hold rank.

The honest conclusion is that the +0.006 gained from rank averaging and the +0.006 gained from
ensembling were real, but the last stretch of public-LB tuning above roughly 0.93 was measuring
noise on 1,772 samples rather than progress on the task.

---

## Reproduce the final submission

The per-model OOF and test logits are committed under [`artifacts/`](artifacts/), so the whole
ensemble stage runs **on CPU, in seconds, with no dataset and no checkpoints**:

```bash
pip install -r requirements.txt
python src/ensemble_rank_average.py
```

```
  v5b   acc=0.9131  macro-F1=0.9155
  v8    acc=0.9102  macro-F1=0.9117
  v9    acc=0.9094  macro-F1=0.9105
  v11   acc=0.9070  macro-F1=0.9085
  v14   acc=0.9099  macro-F1=0.9113

Rank-averaged ensemble  weights={'v14': 0.15, 'v11': 0.15, 'v8': 0.2, 'v9': 0.2, 'v5b': 0.3}
  OOF acc=0.9202  macro-F1=0.9216
  test predictions: 5905
  pen distribution: {1: 861, 2: 772, 3: 719, 4: 659, 5: 726, 6: 721, 7: 727, 8: 720}
```

That distribution is the fingerprint of the 0.94202 submission — if you match it, you have
reproduced it exactly. To also write the CSV, point the script at the extracted dataset:

```bash
python src/ensemble_rank_average.py --data-dir icdar-2026-circleid-pen-classification
```

To re-derive the weights instead of trusting them (grid search over the weight simplex, ~10k
combinations, a couple of minutes on CPU):

```bash
python src/ensemble_rank_average.py --optimize
```

---

## Reproduce from scratch

### 1. Data

Download the [competition data](https://www.kaggle.com/competitions/icdar-2026-circleid-pen-classification/data)
from Kaggle and extract it into the repo root:

```
icdar-2026-circleid-pen-classification/
├── images/                  # 46,155 PNGs
├── train.csv                # 23,850 rows — image_id, image_path, writer_id, pen_id
├── additional_train.csv     # 16,400 rows — the original v1 test set, re-released with labels
├── test.csv                 #  5,905 rows — image_id, image_path
└── sample_submission.csv
```

Two things about this dataset that cost real LB points to discover:

- **Use `additional_train.csv`.** It is the *v1 test set* released with labels after the test
  set was swapped on 5 Mar 2026 (v2 filenames carry a `v2_` prefix). It is 16,400 extra labelled
  images — a 69% increase in training data. Every script here folds it into the training split
  only, never into validation.
- **Writers `W41` and `W50` have wrong pen-ID labels.** The organisers announced this on
  9 Mar 2026. That is ~1,450 samples, 3.6% of the data, poisoning the exact decision boundary
  that is hardest (pens 3/7/8). Every script hard-excludes them via `EXCLUDE_WRITERS = {"W41", "W50"}`.

The dataset is not redistributed here.

### 2. Train the five models

Each script trains 5 writer-disjoint `GroupKFold` folds and writes checkpoints, per-fold OOF
predictions, and hflip-TTA test logits. Expect **8–20 hours per model** on one RTX 5060 (8 GB).

```bash
python src/train_v14_convnext_base.py --seed 42   # ConvNeXt-V2 Base  — strongest solo
python src/train_v9_convnext_tiny.py              # ConvNeXt-V2 Tiny
python src/train_v11_swin_base.py                 # Swin-Base
python src/train_v8_caformer.py                   # CAFormer-S36
python src/train_v5b_dinov2.py                    # DINOv2 + LoRA + ink features
```

Each accepts `--fold N` to train a single fold and `--infer-only` to regenerate test logits from
existing checkpoints. Train one fold first to sanity-check VRAM before committing to a full run.

`train_v5b_dinov2.py` additionally downloads the DINOv2 ViT-B/14 repo and weights
(`dinov2_vitb14_reg4_pretrain.pth`, ~346 MB) into the repo root on first run via `torch.hub`.

### 3. OOF predictions

`v14` writes OOF during training. For models trained before that was added, regenerate from the
saved checkpoints without retraining:

```bash
python src/generate_oof.py --model all
```

### 4. Ensemble

```bash
python src/ensemble_rank_average.py --optimize    # find weights on OOF, apply to test
```

Also available: `ensemble_sweep.py` (probability-space weight sweep with pairwise model
agreement stats), `optimize_cv_weights.py` (CV-optimal probability weights),
`ensemble_final_tricks.py` (rank averaging, LogReg/LightGBM stacking, confidence tie-breaking —
all the variants that were tried at the end).

---

## Method

**Shared recipe across all four timm models** (`v8`, `v9`, `v11`, `v14`):

| | |
|---|---|
| Input | 336 px (≈3× the native image size — upscaling genuinely helps here) |
| Split | 5-fold `GroupKFold` grouped on `writer_id`, so no writer appears in both train and val |
| Head | Linear on pooled features, dropout 0.3 |
| Loss | Focal (γ=2) + label smoothing 0.1; `v11` plain CE + LS |
| Optimizer | AdamW, discriminative LR — backbone 1.0–1.5e-5, head 5e-4 |
| Schedule | 3 warmup epochs → cosine, 40 epochs max, early stop on val macro-F1 |
| Batch | effective 60–72 via gradient accumulation, FP16 |
| EMA | decay 0.999, starting epoch 3 |
| TTA | horizontal flip only |

**`v5b`** is deliberately the odd one out: DINOv2 ViT-B/14 with rank-32 LoRA on the last 8
blocks, SubCenter-ArcFace, same-pen mixup, and a 75-dimensional handcrafted ink-feature vector
(colour statistics, stroke width, LBP texture histogram, Gabor responses, FFT energy, connected
components, border sharpness) projected to 128-d and concatenated with the CLS token. Weakest
solo model, largest ensemble weight — diversity beats individual strength.

### Augmentation

The single highest-leverage decision in the whole project. Standard ImageNet augmentation
recipes are actively harmful here, because they are designed to make features *invariant* to
exactly the signal being classified.

| | Setting | Why |
|---|---|---|
| Colour jitter | brightness/contrast **0.10**, saturation 0.05, hue 0.02 | 0.3 destroys the ink signal entirely; 0.03 overfits to absolute colour. 0.10 is the measured sweet spot. |
| Vertical flip | **removed** | People draw circles in a consistent direction; ink deposition differs top vs bottom. |
| RandomErasing | **removed** | Encourages shape bias. Here texture *is* the signal. |
| Random crop | scale 0.7–1.0 | Below ~0.7 the upscaled crop destroys stroke texture. |
| Gaussian blur | p=0.3, σ 0.3–1.0 | Improves robustness without erasing texture. |
| Rotation | ±15° | Mild rotation in training is fine; rotation *TTA* is not (see below). |

### What was tried and did not work

Full write-up in [`experiments/README.md`](experiments/README.md). The short version:

- **Rotation and multi-scale TTA hurt.** hflip-only 0.9281 → multi-scale 0.9242 → 4×rotation 0.9197.
  Circles are *not* rotation-invariant — stroke direction and writing angle are real signal.
- **Every targeted attack on the pen 3/7 confusion failed**: focal loss, a dedicated 448 px pen-3/7
  binary specialist, confidence-based tie-breaking, pseudo-labelling. The specialist was worse than
  the ensemble it was meant to fix (0.9236 vs 0.9337).
- **Training tricks did not transfer**: SAM optimizer −0.017, GeM pooling −0.005, progressive
  resizing +0.001. Model scale and ensemble diversity were the only levers that moved.
- **Several architectures were simply too weak** for this task: EfficientNetV2-M, MaxViT-Tiny,
  BEiT-Base (pen 3 at 46%), EVA02. ConvNeXt-V2 and Swin were the only families that worked.
- **Changing several things at once** produced v6, which scored 0.015 *below* v5 with no way to
  attribute the regression. Everything after that was ablated one change at a time.

---

## Repository layout

```
src/                             the five final models + ensemble tooling
  train_v5b_dinov2.py            DINOv2 ViT-B + LoRA + 75 ink features
  train_v8_caformer.py           CAFormer-S36
  train_v9_convnext_tiny.py      ConvNeXt-V2 Tiny
  train_v11_swin_base.py         Swin-Base
  train_v14_convnext_base.py     ConvNeXt-V2 Base
  generate_oof.py                OOF predictions from existing checkpoints, no retraining
  ensemble_rank_average.py       final submission — rank averaging + weight search
  ensemble_sweep.py              probability-space weight sweep + model agreement stats
  optimize_cv_weights.py         CV-optimal probability weights
  ensemble_final_tricks.py       rank avg / stacking / tie-breaking variants

experiments/                     the full research log — 20 scripts, mostly negative results
artifacts/                       committed OOF + test logits for the 5 final models (~7 MB)
docs/                            results log and the pen 3/7 bottleneck analysis
```

Scripts are named by their historical version tag (`v5b`, `v8`, `v9`, `v11`, `v14`) because
that is how the artifacts, checkpoint directories and ensemble weights refer to them. Original
filenames map as follows:

| Original | Here |
|---|---|
| `circleid_pen_v5.py` | `src/train_v5b_dinov2.py` |
| `circleid_pen_v8.py` | `src/train_v8_caformer.py` |
| `circleid_pen_v9.py` | `src/train_v9_convnext_tiny.py` |
| `circleid_pen_v11_eva02.py` | `src/train_v11_swin_base.py` (it is Swin-Base, not EVA02 — the file kept the name of an abandoned experiment) |
| `circleid_pen_v14.py` | `src/train_v14_convnext_base.py` |
| `ensemble_v5_v7.py` | `src/ensemble_sweep.py` |

## Requirements

Python 3.10+, PyTorch 2.x with CUDA. Trained on a single RTX 5060 (8 GB) — `v14` needs gradient
checkpointing to fit, `v11` runs at batch size 10. `pip install -r requirements.txt`.

The ensemble scripts need only numpy / pandas / scipy / scikit-learn and run fine on CPU.

## Documentation

- [`docs/RESULTS.md`](docs/RESULTS.md) — every submission and its LB score, chronologically
- [`docs/BOTTLENECK.md`](docs/BOTTLENECK.md) — the pen 3/7 problem in detail
- [`experiments/README.md`](experiments/README.md) — the failed-experiment archive

## License

MIT — see [LICENSE](LICENSE). The competition dataset is not included and is subject to the
ICDAR 2026 CircleID competition terms.
