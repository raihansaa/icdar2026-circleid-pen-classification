# Results log

Public LB is 30% of the test set (1,772 images), private LB the remaining 70% (4,133 images).

## Final outcome

| | Score | Rank |
|---|---|---|
| Public LB | 0.94202 | 4th |
| **Private LB** | **0.91948** | **9th** |
| OOF cross-validation | 0.9202 | — |

Submission: `submission_rank_avg_oof_optimized.csv`, rank averaging with OOF-optimized weights
`v5b:0.30, v8:0.20, v9:0.20, v11:0.15, v14:0.15`.

**The CV predicted the private score to within 0.001.** The public LB overstated it by 0.023.
Everything below was tracked against the public leaderboard as it happened, so read those
numbers knowing that the last ~0.01 of public-LB movement above 0.93 did not survive the
private split.

## Timeline

| Date | Model / ensemble | CV | Public LB |
|------|------------------|-----|-----------|
| — | v3 DINOv2 + LoRA | — | not submitted (colour-jitter bug) |
| — | v4 DINOv2 + 28 ink features + ArcFace | 0.8991 | 0.89491 |
| — | v4 + LightGBM blend | — | 0.89532 |
| — | v5 (336 px, 51 ink features) | — | 0.90630 *(invalidated by W41/W50 fix)* |
| — | v6 (three changes at once) | — | 0.89126 |
| 8 Mar | v7 EfficientNetV2-M | 0.8978 | 0.89855 |
| 18 Mar | v5b DINOv2 + 75 ink features + additional data | 0.9155 | 0.9025 |
| 20 Mar | ConvNeXt-V2 Tiny (via v7 script) | 0.9066 | 0.9214 |
| 20 Mar | ConvNeXt 75% + v5b 25% | — | 0.9253 |
| 22 Mar | v9 ConvNeXt-V2 Tiny + focal | 0.9105 | 0.9281 |
| 23 Mar | v8 CAFormer-S36 + focal | 0.9117 | 0.9197 |
| 25 Mar | v11 Swin-Base | 0.9085 | 0.9231 |
| 31 Mar | v14 ConvNeXt-V2 Base | 0.9113 | 0.9303 |
| 1 Apr | 4-model probability blend (LB-tuned) | 0.9171 | 0.93589 |
| 3 Apr | 5-model probability blend (CV-tuned) | 0.9224 | 0.93143 |
| 3 Apr | LightGBM stacking | 0.9268 | 0.91973 |
| 3 Apr | Rank average, hand weights | — | 0.94091 |
| **3 Apr** | **Rank average, OOF-optimized** | **0.9202** | **0.94202** → private **0.91948** |

Public-LB 1st place at the time was 0.950.

## Final submissions

Two submissions were selected, deliberately chosen to disagree with each other.

**Sub A — `rank_avg_oof_optimized`, public LB 0.94202, private LB 0.91948 — 9th place**
Rank averaging. Weights `v5b:0.30, v8:0.20, v9:0.20, v11:0.15, v14:0.15`, found by grid search
over the weight simplex against OOF accuracy.

**Sub B — 5-model probability blend, public LB 0.93143**
Weights `v8:0.30, v14:0.20, v11:0.20, v5b:0.20, v9:0.10`, CV-optimized.

Sub B was not the second-best submission by public LB. It was picked because it was the only
candidate whose *disagreements* with Sub A favoured it: on rows where the two disagreed,
Sub B was right 48.1% of the time against Sub A's 42.9%. The alternatives were rejected on the
same test:

| Candidate | Public LB | Agreement with Sub A | Net on disagreements |
|---|---|---|---|
| `rank_5model` | 0.94091 | 98.3% | −65 |
| `prob_best_LB` | 0.93589 | — | −105 |
| `cv_opt_4m_prob` | 0.92530 | — | −1 |
| **`cv_opt_5m_prob`** | **0.93143** | — | **+15** |

A near-duplicate of your best submission provides no insurance against a private-LB shift. The
selected pair trades 0.011 of public LB for genuine decorrelation.

## Individual model CV vs LB

| Model | OOF acc | OOF macro-F1 | Public LB | Ensemble weight |
|---|---|---|---|---|
| v5b DINOv2 + ink | **0.9131** | **0.9155** | 0.9025 (worst) | **0.30** (largest) |
| v8 CAFormer-S36 | 0.9102 | 0.9117 | 0.9197 | 0.20 |
| v9 ConvNeXt-V2 Tiny | 0.9094 | 0.9105 | 0.9281 | 0.20 |
| v11 Swin-Base | 0.9070 | 0.9085 | 0.9231 | 0.15 |
| v14 ConvNeXt-V2 Base | 0.9099 | 0.9113 | **0.9303** (best) | 0.15 |

The inversion is the interesting part: **the best-CV / worst-LB model carries the largest
ensemble weight, and the best-LB model carries the smallest.** `v5b` is the only model built on a
different feature space (handcrafted ink statistics rather than learned convolutional features),
so it is wrong in different places than the four backbone models, which is exactly what an
ensemble needs.

## CV-optimized vs LB-optimized weights

| Model | LB-optimized | 4-model CV-opt | 5-model CV-opt |
|---|---|---|---|
| v14 | 0.50 | 0.30 | 0.20 |
| v8 | 0.15 | 0.25 | 0.30 |
| v11 | 0.25 | 0.15 | 0.20 |
| v5b | 0.10 | 0.30 | 0.20 |
| v9 | — | — | 0.10 |
| **CV F1** | 0.9171 | 0.9219 | **0.9224** |

The LB-optimized weights have the *worst* CV of the three. v14 is overweighted at 0.50 against an
optimal 0.20–0.30, and v5b is starved at 0.10 against an optimal 0.20–0.30 — the signature of
fitting the 30% public split rather than the task.

**Confirmed by the private LB.** This table was written before the private split was released and
was the reason the final submission used OOF-optimized rather than LB-optimized weights. The
private score of 0.91948 against an OOF of 0.9202 shows the diagnosis was right: the public
leaderboard was measuring the sample, not the model. Trusting CV over the public LB was the
correct call — it just could not recover the ground that public-LB tuning had never really held.

## Per-pen accuracy (ConvNeXt-V2 Tiny, per fold)

| Fold | F1 | Pen 3 | Pen 7 | Pen 8 |
|---|---|---|---|---|
| 0 | 0.8996 | 0.537 | 0.776 | 0.940 |
| 1 | 0.9048 | 0.623 | 0.714 | 0.956 |
| 2 | 0.9199 | 0.676 | 0.803 | 0.953 |
| 3 | 0.9087 | 0.670 | 0.645 | 0.973 |

Pens 1, 2, 4, 5 sit at ~100% and pen 6 at ~98% in every fold. Pen 3 is the worst at 54–68% and
pen 7 the second worst at 65–80%. Fold-to-fold variance on those two pens is larger than the
difference between most of the architectures tried.

## What actually moved the needle

| Change | Gain |
|---|---|
| Fixing colour augmentation (0.3 → 0.10 jitter) | the difference between working and not |
| Switching to ConvNeXt-V2 from DINOv2+LoRA | +0.019 LB |
| Adding `additional_train.csv` (+16,400 images) | large, folded in with other changes |
| Excluding mislabelled writers W41/W50 | required for a valid score |
| Scaling Tiny → Base | +0.002 LB solo |
| 5-model ensemble over best single model | +0.006 LB |
| Rank averaging instead of probability blending | +0.006 LB |

All gains in that table are measured on the public LB, which the private split later showed to be
inflated by ~0.023. The early changes — the augmentation fix, the backbone switch, the extra
training data, the label cleanup — are large enough to be real regardless. The late ensemble
gains of +0.006 each are near the resolution of a 1,772-sample leaderboard and should be read as
plausible rather than proven.

## If there were a next time

1. **Select on CV, not on public LB.** Here CV tracked the private score to within 0.001 across
   22,850 writer-disjoint OOF samples. The public LB, at 1,772 samples, did not.
2. **Stop tuning when public-LB deltas drop below the noise floor.** Everything past ~0.93 public
   was chasing a sample. That effort would have been better spent on the pen 3/7 problem, which
   was the only real remaining error mass.
3. **Spend the second submission slot on genuine decorrelation.** That was done here — Sub B was
   chosen for disagreement rather than for public score — and it remains the right instinct.
4. **The pen 3/7 confusion was, and still is, the whole ballgame.** Pens 3 and 7 sit at ~68-70%
   accuracy while every other pen is at 95-100%. No amount of ensembling addressed that, and
   nothing else was ever going to matter as much.
