"""
Adaptive Ensemble — when models disagree, trust the most confident one.

Instead of fixed-weight averaging, this uses:
1. Standard weighted average as base
2. For samples where top models disagree, boost the most confident model's vote

"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import softmax

BASE_DIR = Path(__file__).resolve().parent.parent  # repo root
DATA_DIR = BASE_DIR / "icdar-2026-circleid-pen-classification"
OUTPUT_DIR = BASE_DIR / "outputs_adaptive"

LOGIT_SOURCES = {
    "v14": BASE_DIR / "outputs_v14_seed42" / "test_logits_v14.npy",
    "v8":  BASE_DIR / "outputs_v8" / "test_logits_v8.npy",
    "v11": BASE_DIR / "outputs_v11" / "test_logits_v11.npy",
    "v5b": BASE_DIR / "outputs_v5b" / "test_logits_v5b.npy",
}

BASE_WEIGHTS = {"v14": 0.50, "v8": 0.15, "v11": 0.25, "v5b": 0.10}


def load_logits():
    logits = {}
    for name, path in LOGIT_SOURCES.items():
        if path.exists():
            logits[name] = np.load(path)
            print(f"  Loaded {name}: {logits[name].shape}")
        else:
            print(f"  WARNING: {name} not found at {path}")
    return logits


def method_confidence_weighted(logits_dict, use_softmax=True):
    """Weight each model by its confidence (max probability) per sample."""
    names = list(logits_dict.keys())
    n_samples = logits_dict[names[0]].shape[0]
    n_classes = logits_dict[names[0]].shape[1]

    blended = np.zeros((n_samples, n_classes), dtype=np.float64)

    for i in range(n_samples):
        total_conf = 0.0
        sample_blend = np.zeros(n_classes, dtype=np.float64)

        for name in names:
            if use_softmax:
                probs = softmax(logits_dict[name][i])
            else:
                probs = logits_dict[name][i]
            confidence = probs.max()
            weight = BASE_WEIGHTS[name] * confidence
            sample_blend += weight * probs
            total_conf += weight

        blended[i] = sample_blend / (total_conf + 1e-12)

    return blended


def method_max_confidence(logits_dict, use_softmax=True):
    """For each sample, use the prediction of the most confident model."""
    names = list(logits_dict.keys())
    n_samples = logits_dict[names[0]].shape[0]
    n_classes = logits_dict[names[0]].shape[1]

    blended = np.zeros((n_samples, n_classes), dtype=np.float64)

    for i in range(n_samples):
        best_conf = -1
        best_probs = None

        for name in names:
            if use_softmax:
                probs = softmax(logits_dict[name][i])
            else:
                probs = logits_dict[name][i]
            confidence = probs.max()
            if confidence > best_conf:
                best_conf = confidence
                best_probs = probs

        blended[i] = best_probs

    return blended


def method_disagree_boost(logits_dict, use_softmax=True):
    """Standard weighted avg, but when models disagree, boost the most confident."""
    names = list(logits_dict.keys())
    n_samples = logits_dict[names[0]].shape[0]
    n_classes = logits_dict[names[0]].shape[1]

    # Get per-model predictions and probs
    all_probs = {}
    all_preds = {}
    for name in names:
        if use_softmax:
            all_probs[name] = softmax(logits_dict[name], axis=1)
        else:
            all_probs[name] = logits_dict[name]
        all_preds[name] = all_probs[name].argmax(axis=1)

    # Standard weighted blend
    standard_blend = np.zeros((n_samples, n_classes), dtype=np.float64)
    for name in names:
        standard_blend += BASE_WEIGHTS[name] * all_probs[name]

    # Find disagreements
    pred_matrix = np.stack([all_preds[name] for name in names], axis=1)  # (N, M)
    n_disagree = 0

    blended = standard_blend.copy()
    for i in range(n_samples):
        preds_i = pred_matrix[i]
        if len(set(preds_i)) > 1:  # models disagree
            n_disagree += 1
            # Find most confident model
            best_conf = -1
            best_name = None
            for name in names:
                conf = all_probs[name][i].max()
                if conf > best_conf:
                    best_conf = conf
                    best_name = name

            # Blend: 60% standard avg + 40% most confident model
            blended[i] = 0.6 * standard_blend[i] + 0.4 * all_probs[best_name][i]

    print(f"  Disagreements: {n_disagree}/{n_samples} ({n_disagree/n_samples*100:.1f}%)")
    return blended


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logit-space", action="store_true")
    args = parser.parse_args()

    use_softmax = not args.logit_space
    mode = "probability" if use_softmax else "logit"
    print(f"Adaptive Ensemble ({mode}-space)")

    logits = load_logits()
    if len(logits) < 2:
        print("ERROR: Need at least 2 logit files")
        return

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    pen_list = sorted(train_df["pen_id"].unique())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    methods = {
        "conf_weighted": method_confidence_weighted,
        "max_confidence": method_max_confidence,
        "disagree_boost": method_disagree_boost,
    }

    for method_name, method_fn in methods.items():
        print(f"\n--- Method: {method_name} ---")
        blended = method_fn(logits, use_softmax=use_softmax)
        preds = blended.argmax(axis=1)
        pen_preds = [pen_list[p] for p in preds]

        sub = pd.DataFrame({"image_id": test_df["image_id"], "pen_id": pen_preds})
        sub_path = OUTPUT_DIR / f"submission_{method_name}_{mode}.csv"
        sub.to_csv(sub_path, index=False)
        print(f"  Saved: {sub_path}")
        print(f"  Pen distribution:")
        print(f"  {sub['pen_id'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
