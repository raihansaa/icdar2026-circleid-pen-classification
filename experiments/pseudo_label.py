"""
Pseudo-labeling pipeline for CircleID Pen Classification

Takes the best ensemble's predictions on test set, filters high-confidence samples,
and creates a pseudo-labeled CSV that can be used as additional training data.

"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import softmax

BASE_DIR = Path(__file__).resolve().parent.parent  # repo root
DATA_DIR = BASE_DIR / "icdar-2026-circleid-pen-classification"


def generate_pseudo_labels(confidence_threshold=0.95):
    """Generate pseudo-labels from best ensemble logits."""
    print("=" * 60)
    print("GENERATING PSEUDO-LABELS")
    print("=" * 60)

    # Load all available logits
    logit_files = {
        "v9": BASE_DIR / "outputs_v9" / "test_logits_v9.npy",
        "v8": BASE_DIR / "outputs_v8" / "test_logits_v8.npy",
        "v5b": BASE_DIR / "outputs_v5b" / "test_logits_v5b.npy",
    }

    available = {}
    for name, path in logit_files.items():
        if path.exists():
            available[name] = np.load(path)
            print(f"  Loaded {name}: {path}")
        else:
            print(f"  Not found: {path}")

    if not available:
        print("ERROR: No logits found!")
        return

    # Best ensemble weights (from sweep)
    weights = {"v9": 0.55, "v8": 0.30, "v5b": 0.15}
    weights = {k: v for k, v in weights.items() if k in available}
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}
    print(f"\n  Ensemble weights: {weights}")

    # Blend in probability space
    blended = np.zeros_like(next(iter(available.values())), dtype=np.float64)
    for name, w in weights.items():
        probs = softmax(available[name], axis=1)
        blended += w * probs

    # Load test CSV
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    pen_list = sorted(train_df["pen_id"].unique())

    # Get predictions and confidences
    predictions = blended.argmax(axis=1)
    confidences = blended.max(axis=1)

    print(f"\n  Test samples: {len(test_df)}")
    print(f"  Confidence threshold: {confidence_threshold}")
    print(f"\n  Confidence distribution:")
    for thresh in [0.99, 0.95, 0.90, 0.85, 0.80]:
        n = (confidences >= thresh).sum()
        print(f"    >= {thresh}: {n} samples ({n/len(test_df)*100:.1f}%)")

    # Filter high-confidence samples
    mask = confidences >= confidence_threshold
    high_conf_indices = np.where(mask)[0]

    pseudo_df = pd.DataFrame({
        "image_id": test_df.iloc[high_conf_indices]["image_id"].values,
        "image_path": test_df.iloc[high_conf_indices]["image_path"].values,
        "pen_id": [pen_list[predictions[i]] for i in high_conf_indices],
        "confidence": confidences[high_conf_indices],
        "writer_id": "PSEUDO",  # mark as pseudo-labeled
    })

    # Per-pen distribution
    print(f"\n  Selected {len(pseudo_df)} pseudo-labeled samples ({len(pseudo_df)/len(test_df)*100:.1f}%)")
    print(f"\n  Pen distribution:")
    for pen in pen_list:
        n = (pseudo_df["pen_id"] == pen).sum()
        print(f"    Pen {pen}: {n}")

    # Save
    out_path = BASE_DIR / "pseudo_labels.csv"
    pseudo_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    return pseudo_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="Generate pseudo-labels from ensemble")
    parser.add_argument("--threshold", type=float, default=0.95, help="Confidence threshold (default: 0.95)")
    args = parser.parse_args()

    if args.generate:
        generate_pseudo_labels(args.threshold)
    else:
        print("Usage:")
        print("  python pseudo_label.py --generate              # generate pseudo-labels")
        print("  python pseudo_label.py --generate --threshold 0.90  # lower threshold")


if __name__ == "__main__":
    main()
