"""
Find optimal ensemble weights by maximizing CV accuracy on OOF predictions.
These weights are tuned on your full training data (not public LB),
so they're more likely to generalize to private LB.

"""

import numpy as np
from pathlib import Path
from scipy.special import softmax
from sklearn.metrics import f1_score, accuracy_score
from itertools import product

BASE_DIR = Path(__file__).resolve().parent.parent  # repo root

OOF_SOURCES = {
    "v14": BASE_DIR / "checkpoints_v14_seed42",
    "v8":  BASE_DIR / "checkpoints_v8",
    "v11": BASE_DIR / "checkpoints_v11",
    "v9":  BASE_DIR / "checkpoints_v9",
    "v5b": BASE_DIR / "checkpoints_v5b",
}


def load_oof(name, path):
    logits = np.load(path / "oof_logits_all.npy")
    labels = np.load(path / "oof_labels_all.npy")
    mask = np.load(path / "oof_mask_all.npy")
    print(f"  {name}: {mask.sum()} OOF samples")
    return logits, labels, mask


def blend_oof(oof_dict, weights, use_softmax=True):
    names = list(weights.keys())
    ref = oof_dict[names[0]]
    blended = np.zeros_like(ref[0], dtype=np.float64)

    for name in names:
        logits, labels, mask = oof_dict[name]
        if use_softmax:
            probs = softmax(logits, axis=1)
        else:
            probs = logits
        blended += weights[name] * probs

    return blended, ref[1], ref[2]  # return labels and mask from first model


def main():
    print("Loading OOF predictions...")
    oof_dict = {}
    for name, path in OOF_SOURCES.items():
        try:
            oof_dict[name] = load_oof(name, path)
        except FileNotFoundError:
            print(f"  {name}: OOF not found, skipping")

    if len(oof_dict) < 2:
        print("ERROR: Need at least 2 models with OOF predictions")
        return

    names = sorted(oof_dict.keys())
    print(f"\nModels: {names}")

    # Verify all have same mask
    ref_mask = oof_dict[names[0]][2]
    for name in names[1:]:
        assert np.array_equal(ref_mask, oof_dict[name][2]), f"Mask mismatch for {name}"

    labels = oof_dict[names[0]][1][ref_mask]

    # Individual model scores
    print(f"\n{'='*60}")
    print("Individual model CV scores:")
    print(f"{'='*60}")
    for name in names:
        logits = oof_dict[name][0][ref_mask]
        preds = logits.argmax(axis=1)
        f1 = f1_score(labels, preds, average="macro")
        acc = accuracy_score(labels, preds)
        print(f"  {name}: F1={f1:.4f}, Acc={acc:.4f}")

    # Grid search over weights
    print(f"\n{'='*60}")
    print("Searching optimal weights (probability-space)...")
    print(f"{'='*60}")

    best_f1 = 0
    best_weights = None
    best_acc = 0

    # Search with 5% increments
    steps = [i / 20 for i in range(21)]  # 0.00, 0.05, ..., 1.00

    # Precompute softmax probs for speed
    all_probs = {}
    for name in names:
        all_probs[name] = softmax(oof_dict[name][0][ref_mask], axis=1)

    if len(names) == 4:
        count = 0
        for w0 in steps:
            for w1 in steps:
                for w2 in steps:
                    w3 = 1.0 - w0 - w1 - w2
                    if w3 < -0.001 or w3 > 1.001:
                        continue
                    w3 = max(0, w3)
                    ws = [w0, w1, w2, w3]

                    blended = sum(ws[i] * all_probs[names[i]] for i in range(4))
                    preds = blended.argmax(axis=1)
                    f1 = f1_score(labels, preds, average="macro")

                    if f1 > best_f1:
                        best_f1 = f1
                        best_acc = accuracy_score(labels, preds)
                        best_weights = {names[i]: ws[i] for i in range(4)}
                        count += 1

    elif len(names) == 5:
        # Use 10% steps for 5 models (keeps search tractable)
        steps_5 = [i / 10 for i in range(11)]  # 0.0, 0.1, ..., 1.0
        count = 0
        for w0 in steps_5:
            for w1 in steps_5:
                for w2 in steps_5:
                    for w3 in steps_5:
                        w4 = 1.0 - w0 - w1 - w2 - w3
                        if w4 < -0.001 or w4 > 1.001:
                            continue
                        w4 = max(0, w4)
                        ws = [w0, w1, w2, w3, w4]

                        blended = sum(ws[i] * all_probs[names[i]] for i in range(5))
                        preds = blended.argmax(axis=1)
                        f1 = f1_score(labels, preds, average="macro")

                        if f1 > best_f1:
                            best_f1 = f1
                            best_acc = accuracy_score(labels, preds)
                            best_weights = {names[i]: ws[i] for i in range(5)}
                            count += 1

    if best_weights:
        print(f"\nBest CV weights (prob-space, {len(names)} models):")
        for name, w in sorted(best_weights.items()):
            print(f"  {name}: {w:.2f}")
        print(f"  CV F1: {best_f1:.4f}, CV Acc: {best_acc:.4f}")
    else:
        print("\n  No valid weight combination found.")

    # Also search without v9
    names_no_v9 = [n for n in names if n != "v9"]
    if len(names_no_v9) >= 3:
        print(f"\n{'='*60}")
        print(f"Searching without v9 ({names_no_v9})...")
        print(f"{'='*60}")

        best_f1_no9 = 0
        best_weights_no9 = None

        if len(names_no_v9) == 3:
            for w0 in steps:
                for w1 in steps:
                    w2 = 1.0 - w0 - w1
                    if w2 < -0.001 or w2 > 1.001:
                        continue
                    w2 = max(0, w2)
                    weights = {names_no_v9[0]: w0, names_no_v9[1]: w1, names_no_v9[2]: w2}

                    blended = np.zeros_like(oof_dict[names_no_v9[0]][0][ref_mask], dtype=np.float64)
                    for name in names_no_v9:
                        probs = softmax(oof_dict[name][0][ref_mask], axis=1)
                        blended += weights[name] * probs

                    preds = blended.argmax(axis=1)
                    f1 = f1_score(labels, preds, average="macro")

                    if f1 > best_f1_no9:
                        best_f1_no9 = f1
                        best_weights_no9 = weights.copy()

        elif len(names_no_v9) == 4:
            for w0 in steps:
                for w1 in steps:
                    for w2 in steps:
                        w3 = 1.0 - w0 - w1 - w2
                        if w3 < -0.001 or w3 > 1.001:
                            continue
                        w3 = max(0, w3)
                        weights = {names_no_v9[0]: w0, names_no_v9[1]: w1,
                                   names_no_v9[2]: w2, names_no_v9[3]: w3}

                        blended = np.zeros_like(oof_dict[names_no_v9[0]][0][ref_mask], dtype=np.float64)
                        for name in names_no_v9:
                            probs = softmax(oof_dict[name][0][ref_mask], axis=1)
                            blended += weights[name] * probs

                        preds = blended.argmax(axis=1)
                        f1 = f1_score(labels, preds, average="macro")

                        if f1 > best_f1_no9:
                            best_f1_no9 = f1
                            best_weights_no9 = weights.copy()

        print(f"\nBest CV weights without v9:")
        for name, w in sorted(best_weights_no9.items()):
            print(f"  {name}: {w:.2f}")
        blended = np.zeros_like(oof_dict[names_no_v9[0]][0][ref_mask], dtype=np.float64)
        for name in names_no_v9:
            probs = softmax(oof_dict[name][0][ref_mask], axis=1)
            blended += best_weights_no9[name] * probs
        preds = blended.argmax(axis=1)
        acc_no9 = accuracy_score(labels, preds)
        print(f"  CV F1: {best_f1_no9:.4f}, CV Acc: {acc_no9:.4f}")

    # Compare with LB-optimized weights
    print(f"\n{'='*60}")
    print("LB-optimized weights for comparison:")
    print(f"{'='*60}")
    lb_weights = {"v14": 0.50, "v8": 0.15, "v11": 0.25, "v5b": 0.10}
    blended = np.zeros_like(oof_dict[names[0]][0][ref_mask], dtype=np.float64)
    total_w = 0
    for name in lb_weights:
        if name in oof_dict:
            probs = softmax(oof_dict[name][0][ref_mask], axis=1)
            blended += lb_weights[name] * probs
            total_w += lb_weights[name]
    if total_w > 0:
        blended /= total_w
    preds = blended.argmax(axis=1)
    lb_f1 = f1_score(labels, preds, average="macro")
    lb_acc = accuracy_score(labels, preds)
    print(f"  v14:0.50, v8:0.15, v11:0.25 (normalized)")
    print(f"  CV F1: {lb_f1:.4f}, CV Acc: {lb_acc:.4f}")


if __name__ == "__main__":
    main()
