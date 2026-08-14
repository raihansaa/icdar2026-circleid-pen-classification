# Experiment archive

Every script that did not make it into the final ensemble, kept because the negative results are
the more useful half of the record. These are preserved as they were run — they are not
maintained, and several depend on checkpoint directories that are not in this repo.

Scripts here resolve paths relative to the repo root, same as `src/`.

---

## Model versions

### `circleid_pen_v3.py` — DINOv2 + LoRA baseline
Never submitted. `jitter_brightness=0.3` in the augmentation pipeline destroyed the ink colour
signal, which is the entire basis for the task. This one bug set the direction for everything
after it: **augmentation is the dominant design choice here, not architecture.**

### `circleid_pen_v4.py` — CV 0.8991, LB 0.89491
First working model. Colour jitter cut to 0.03/0.05, 28 handcrafted ink features, SubCenter-ArcFace,
writer-disjoint 5-fold `GroupKFold`, pen-balanced batch sampling, EMA.
Per-fold F1: 0.9022 / 0.8972 / 0.8876 / 0.9147 / 0.8939.

`submit_v4_enhanced.py` (LB 0.89532) bolted a LightGBM model on the 28 ink features and blended
it with the network. GBM alone scored CV 0.8211; the blend gained **+0.00041**. Not worth it.

### `circleid_pen_v5.py` → shipped as `src/train_v5b_dinov2.py`
The v5 *submission* (LB 0.90630) was the first past 0.90, but it was invalidated by the W41/W50
annotation fix. The script was then overhauled into **v5b** — 336 px, LoRA rank 32, 75 ink
features, additional training data, W41/W50 excluded — and that version is in the final ensemble.

### `circleid_pen_v6.py` / `v6a.py` / `v6b.py` — LB 0.89126, a regression
Three changes landed at once: rotation 15°→0° with `crop_scale_min` 0.5→0.85, LoRA extended to
the MLP blocks, and hflip TTA. Result was **0.015 worse than v5** with no way to attribute the
loss. `crop_scale_min=0.85` was the likely culprit — too conservative to act as augmentation.

**This is the most expensive lesson in the project.** Everything afterwards changed one thing at
a time.

### `circleid_pen_v7.py` — CV 0.8978, LB 0.89855
EfficientNetV2-M full finetune, plain CE + label smoothing, ink features dropped entirely. For a
while the only submission that survived the W41/W50 reevaluation. Per-fold F1 ranged 0.8897–0.9180
— high variance. Dropping the ink features hurt.

This script is also what produced the **ConvNeXt-V2 Tiny** result (LB 0.9214, CV 0.9066) via
`--backbone convnextv2_tiny.fcmae_ft_in22k_in1k`. That run was the turning point: a plain CNN beat
DINOv2-plus-ink-features by two full points on LB despite a *lower* CV. It became `v9`.

### `circleid_pen_v10.py`, `circleid_pen_v12.py` — superseded
Intermediate ConvNeXt/hybrid variants, replaced by v9 and v14.

### `circleid_pen_v13_beit.py` — worst result of the project
BEiT-Base. Pen 3 accuracy **46%**, below every other model tried.

### `circleid_pen_v15a/b/c.py` — training tricks, all rejected
Ablated on fold 0 against the v9 baseline (F1 = 0.9070):

| Variant | Fold-0 F1 | Δ vs v9 | Verdict |
|---|---|---|---|
| `v15b` SAM optimizer | 0.8899 | −0.017 | Clearly worse; pen 3 fell to 54.4% |
| `v15a` GeM pooling | 0.9020 | −0.005 | Worse |
| `v15c` progressive resize 224→336→448 | 0.9084 | +0.001 | Inside noise, not worth the complexity |

Conclusion: on a Tiny backbone the optimizer/pooling/curriculum knobs are exhausted. The
remaining gains had to come from **model scale and ensemble diversity** — which is what `v14`
and the 5-model blend delivered.

### `circleid_pen_v16.py` — architecture search, all too weak
- EfficientNetV2-M (54M): fold-0 F1 0.8962, early-stopped at epoch 19
- MaxViT-Tiny (31M): fold-0 F1 0.884 at epoch 10
- MaxViT-Small (68M): 7.6 GB VRAM, would not fit alongside anything else

Together with v13, this closes the architecture search: **ConvNeXt-V2 and Swin are the only
families that work on this task.** Pure EfficientNet-style CNNs and small ViTs underperform.

### `kaggle_convnext_base.py` — ConvNeXt-V2 Base for 16 GB Kaggle GPUs
Same model as `v14` but sized for a T4/P100 instead of an 8 GB local card. `v14` reached the same
place locally using gradient checkpointing.

---

## Attacks on the pen 3/7 confusion — all failed

Pen 3 sits at ~68% accuracy and pen 7 at ~70%; every other pen is 95–100%. 1,363 test samples
have both pens in their top-2, and 226 have a top-2 gap below 0.20 — effectively coin flips.
This one confusion is the whole remaining gap to 1st place, so it got the most attention and
produced the least.

### `circleid_pen37_specialist.py` — LB 0.9236 (worse than the 0.9337 ensemble it was fixing)
A dedicated binary pen-3-vs-7 ConvNeXt at 448 px, applied as an override when the main ensemble
was uncertain. Mean CV 73.5%, with a heavy bias toward pen 7: it predicted 1,018 pen 7 against
487 pen 3, where the ensemble predicted 905 / 600. Even restricting overrides to the 47 samples
with a top-2 gap below 0.10 still hurt the LB.

### Focal loss (γ=2), in `v8`, `v9`, `v14`
Kept in the final models because it did not hurt, but it never produced the intended pen 3/7
improvement. The confusion is not a gradient-allocation problem.

### `pseudo_label.py` — flat CV
4,670 test samples pseudo-labelled at ≥0.85 confidence and folded back into training. The added
labels were noisy precisely where the model was already wrong, so it reinforced its own errors.

### `stacking.py` — best OOF, worst LB
Logistic regression and LightGBM meta-learners on the OOF probability matrix.
LightGBM: **OOF 0.9268 (best of anything tried) → LB 0.91973 (worst of anything submitted).**
A clean demonstration of a meta-learner overfitting 22,850 OOF rows.

### Confidence tie-breaking (in `src/ensemble_final_tricks.py`)
Deferring low-confidence predictions to `v5b`, swept over thresholds 0.4–0.7. No threshold helped.

**Conclusion:** the pen 3/7 distinction may not be recoverable from these pixels at this model
scale. Higher optical resolution of the source scans would probably be needed, not a better loss.

---

## TTA — `infer_enhanced_tta.py`

Tested on `v9` (LB 0.9281 with hflip only):

| TTA | Views | LB |
|---|---|---|
| **hflip** | 2 | **0.9281** |
| multi-scale (304 / 336 / 368 + hflip) | 6 | 0.9242 |
| 4 rotations × hflip | 8 | 0.9197 |

More TTA was monotonically worse. Circles are **not** rotation-invariant — stroke direction and
writing angle carry real pen signal, and rotating destroys it. Multi-scale adds noise because
the models were trained at a fixed 336 px. Every final model uses hflip only.

## `ensemble_adaptive.py`
Per-sample adaptive model weighting based on prediction confidence. Did not beat fixed weights.
