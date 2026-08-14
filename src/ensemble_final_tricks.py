"""
Final ensemble tricks — generate all submissions at once.

Methods:
  1. Untested weight combos (prob-space)
  2. Rank averaging
  3. Stacking (LightGBM meta-learner on OOF)
  4. Confidence-based tie-breaking

"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import softmax
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parent.parent  # repo root
DATA_DIR = BASE_DIR / "icdar-2026-circleid-pen-classification"
OUTPUT_DIR = BASE_DIR / "outputs_ensemble"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
pen_list = sorted(train_df["pen_id"].unique())

# Load test logits
TEST_LOGITS = {
    "v5b": np.load(BASE_DIR / "outputs_v5b" / "test_logits_v5b.npy"),
    "v8":  np.load(BASE_DIR / "outputs_v8" / "test_logits_v8.npy"),
    "v9":  np.load(BASE_DIR / "outputs_v9" / "test_logits_v9.npy"),
    "v11": np.load(BASE_DIR / "outputs_v11" / "test_logits_v11.npy"),
    "v14": np.load(BASE_DIR / "outputs_v14_seed42" / "test_logits_v14.npy"),
}

# Load OOF logits and labels
OOF_LOGITS = {
    "v5b": np.load(BASE_DIR / "checkpoints_v5b" / "oof_logits_all.npy"),
    "v8":  np.load(BASE_DIR / "checkpoints_v8" / "oof_logits_all.npy"),
    "v9":  np.load(BASE_DIR / "checkpoints_v9" / "oof_logits_all.npy"),
    "v11": np.load(BASE_DIR / "checkpoints_v11" / "oof_logits_all.npy"),
    "v14": np.load(BASE_DIR / "checkpoints_v14_seed42" / "oof_logits_all.npy"),
}
OOF_LABELS = np.load(BASE_DIR / "checkpoints_v14_seed42" / "oof_labels_all.npy")


def save_submission(preds, tag):
    pen_preds = [pen_list[p] for p in preds]
    sub = pd.DataFrame({"image_id": test_df["image_id"], "pen_id": pen_preds})
    out_path = OUTPUT_DIR / f"submission_{tag}.csv"
    sub.to_csv(out_path, index=False)
    dist = sub["pen_id"].value_counts().sort_index()
    print(f"  Saved: {out_path.name}")
    print(f"  Pen dist: {dict(dist)}")
    return sub


def prob_blend(weights):
    blended = np.zeros_like(TEST_LOGITS["v14"], dtype=np.float64)
    for name, w in weights.items():
        blended += w * softmax(TEST_LOGITS[name], axis=1)
    return blended.argmax(axis=1)


def oof_prob_blend(weights):
    blended = np.zeros_like(OOF_LOGITS["v14"], dtype=np.float64)
    for name, w in weights.items():
        blended += w * softmax(OOF_LOGITS[name], axis=1)
    return blended.argmax(axis=1)


# ============================================================
# 1. Untested weight combos
# ============================================================
print("=" * 60)
print("1. UNTESTED WEIGHT COMBOS")
print("=" * 60)

combos = {
    "v11w30_v14w50_v5bw10_v8w10_prob": {"v14": 0.50, "v8": 0.10, "v11": 0.30, "v5b": 0.10},
    "v11w30_v14w50_v5bw05_v8w15_prob": {"v14": 0.50, "v8": 0.15, "v11": 0.30, "v5b": 0.05},
}

for tag, weights in combos.items():
    # OOF accuracy
    oof_preds = oof_prob_blend(weights)
    oof_acc = accuracy_score(OOF_LABELS, oof_preds)
    print(f"\n{tag}")
    print(f"  OOF accuracy: {oof_acc:.4f}")
    preds = prob_blend(weights)
    save_submission(preds, tag)


# ============================================================
# 2. Rank averaging
# ============================================================
print("\n" + "=" * 60)
print("2. RANK AVERAGING")
print("=" * 60)

def rank_average(logits_dict, weights):
    """Rank-average: convert logits to ranks per class, then weight-average."""
    n_samples = next(iter(logits_dict.values())).shape[0]
    n_classes = next(iter(logits_dict.values())).shape[1]
    blended = np.zeros((n_samples, n_classes), dtype=np.float64)
    for name, w in weights.items():
        logits = logits_dict[name]
        # Rank each column (higher logit = higher rank)
        ranked = np.zeros_like(logits, dtype=np.float64)
        for c in range(n_classes):
            ranked[:, c] = rankdata(logits[:, c])
        blended += w * ranked
    return blended

# Best LB weights with rank averaging
rank_weights = {"v14": 0.50, "v8": 0.15, "v11": 0.25, "v5b": 0.10}
print(f"\nRank avg with best LB weights: {rank_weights}")
oof_blended = rank_average(OOF_LOGITS, rank_weights)
oof_acc = accuracy_score(OOF_LABELS, oof_blended.argmax(axis=1))
print(f"  OOF accuracy: {oof_acc:.4f}")
preds = rank_average(TEST_LOGITS, rank_weights).argmax(axis=1)
save_submission(preds, "rank_avg_best")

# Rank averaging with 5 models equal weight
rank_weights_5 = {"v14": 0.25, "v8": 0.20, "v11": 0.20, "v5b": 0.15, "v9": 0.20}
print(f"\nRank avg 5-model: {rank_weights_5}")
oof_blended = rank_average(OOF_LOGITS, rank_weights_5)
oof_acc = accuracy_score(OOF_LABELS, oof_blended.argmax(axis=1))
print(f"  OOF accuracy: {oof_acc:.4f}")
preds = rank_average(TEST_LOGITS, rank_weights_5).argmax(axis=1)
save_submission(preds, "rank_avg_5model")


# ============================================================
# 3. Stacking (meta-learner)
# ============================================================
print("\n" + "=" * 60)
print("3. STACKING (Meta-learner)")
print("=" * 60)

MODEL_NAMES = ["v5b", "v8", "v9", "v11", "v14"]

# Build OOF feature matrix: softmax probs from each model concatenated
oof_features = np.hstack([softmax(OOF_LOGITS[n], axis=1) for n in MODEL_NAMES])
test_features = np.hstack([softmax(TEST_LOGITS[n], axis=1) for n in MODEL_NAMES])
print(f"OOF features: {oof_features.shape}, Test features: {test_features.shape}")

# 3a. Logistic Regression stacking
print("\n3a. Logistic Regression stacking")
lr = LogisticRegression(C=1.0, max_iter=1000, multi_class="multinomial", solver="lbfgs")
lr.fit(oof_features, OOF_LABELS)
oof_preds = lr.predict(oof_features)
print(f"  OOF accuracy (train): {accuracy_score(OOF_LABELS, oof_preds):.4f}")
test_preds = lr.predict(test_features)
save_submission(test_preds, "stack_lr")

# 3b. Logistic Regression with regularization sweep
print("\n3b. LR regularization sweep")
best_c, best_acc = 1.0, 0.0
for c in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
    lr_c = LogisticRegression(C=c, max_iter=1000, multi_class="multinomial", solver="lbfgs")
    # Simple 5-fold CV on OOF
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(lr_c, oof_features, OOF_LABELS, cv=5, scoring="accuracy")
    mean_acc = scores.mean()
    print(f"  C={c}: CV accuracy = {mean_acc:.4f} (+/- {scores.std():.4f})")
    if mean_acc > best_acc:
        best_acc = mean_acc
        best_c = c

print(f"  Best C={best_c}, CV acc={best_acc:.4f}")
lr_best = LogisticRegression(C=best_c, max_iter=1000, multi_class="multinomial", solver="lbfgs")
lr_best.fit(oof_features, OOF_LABELS)
test_preds = lr_best.predict(test_features)
save_submission(test_preds, f"stack_lr_C{best_c}")

# 3c. Try LightGBM if available
try:
    import lightgbm as lgb
    print("\n3c. LightGBM stacking")

    from sklearn.model_selection import StratifiedKFold

    # Use 5-fold stacking to avoid overfitting
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lgb_test_preds = np.zeros((len(test_features), 8))
    lgb_oof_preds = np.zeros(len(OOF_LABELS))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(oof_features, OOF_LABELS)):
        dtrain = lgb.Dataset(oof_features[tr_idx], OOF_LABELS[tr_idx])
        dval = lgb.Dataset(oof_features[val_idx], OOF_LABELS[val_idx])

        params = {
            "objective": "multiclass",
            "num_class": 8,
            "metric": "multi_logloss",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbose": -1,
            "seed": 42,
        }

        model = lgb.train(params, dtrain, num_boost_round=500,
                         valid_sets=[dval], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

        val_pred = model.predict(oof_features[val_idx])
        lgb_oof_preds[val_idx] = val_pred.argmax(axis=1)
        lgb_test_preds += model.predict(test_features) / 5

    print(f"  OOF accuracy: {accuracy_score(OOF_LABELS, lgb_oof_preds):.4f}")
    test_preds = lgb_test_preds.argmax(axis=1)
    save_submission(test_preds, "stack_lgb")

except ImportError:
    print("\n3c. LightGBM not installed, skipping")


# ============================================================
# 4. Confidence-based tie-breaking
# ============================================================
print("\n" + "=" * 60)
print("4. CONFIDENCE-BASED TIE-BREAKING")
print("=" * 60)

# Use best ensemble as primary, but for low-confidence predictions,
# defer to v5b (different feature space — DINOv2 + ink features)
primary_weights = {"v14": 0.50, "v8": 0.15, "v11": 0.25, "v5b": 0.10}
primary_probs = np.zeros_like(TEST_LOGITS["v14"], dtype=np.float64)
for name, w in primary_weights.items():
    primary_probs += w * softmax(TEST_LOGITS[name], axis=1)

primary_preds = primary_probs.argmax(axis=1)
primary_conf = primary_probs.max(axis=1)

# For low-confidence samples, use v5b as tiebreaker
v5b_probs = softmax(TEST_LOGITS["v5b"], axis=1)
v5b_preds = v5b_probs.argmax(axis=1)

for threshold in [0.4, 0.5, 0.6, 0.7]:
    preds = primary_preds.copy()
    low_conf_mask = primary_conf < threshold
    n_replaced = low_conf_mask.sum()

    # For low confidence, average primary with heavier v5b
    fallback_probs = 0.5 * primary_probs[low_conf_mask] + 0.5 * v5b_probs[low_conf_mask]
    preds[low_conf_mask] = fallback_probs.argmax(axis=1)

    # OOF version
    oof_primary_probs = np.zeros_like(OOF_LOGITS["v14"], dtype=np.float64)
    for name, w in primary_weights.items():
        oof_primary_probs += w * softmax(OOF_LOGITS[name], axis=1)
    oof_primary_preds = oof_primary_probs.argmax(axis=1)
    oof_primary_conf = oof_primary_probs.max(axis=1)
    oof_v5b_probs = softmax(OOF_LOGITS["v5b"], axis=1)

    oof_preds = oof_primary_preds.copy()
    oof_low = oof_primary_conf < threshold
    oof_fallback = 0.5 * oof_primary_probs[oof_low] + 0.5 * oof_v5b_probs[oof_low]
    oof_preds[oof_low] = oof_fallback.argmax(axis=1)
    oof_acc = accuracy_score(OOF_LABELS, oof_preds)

    print(f"\n  Threshold={threshold}: {n_replaced} samples replaced, OOF acc={oof_acc:.4f}")
    save_submission(preds, f"conf_tiebreak_t{int(threshold*100)}")


# ============================================================
# 5. Summary — OOF accuracy comparison
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY — OOF Accuracy for all methods")
print("=" * 60)

# Best LB baseline
oof_preds_baseline = oof_prob_blend({"v14": 0.50, "v8": 0.15, "v11": 0.25, "v5b": 0.10})
print(f"  Baseline (best LB):    {accuracy_score(OOF_LABELS, oof_preds_baseline):.4f}")
