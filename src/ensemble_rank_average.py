"""
Rank-averaged ensemble — reproduces the final competition submission (public LB 0.94202).

Reads the pre-computed logits shipped in artifacts/, so this runs on CPU in seconds
without the dataset, the checkpoints, or a GPU.

"""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = Path(__file__).resolve().parent.parent  # repo root
OOF_DIR = BASE_DIR / "artifacts" / "oof"
TEST_DIR = BASE_DIR / "artifacts" / "test_logits"

MODELS = ["v5b", "v8", "v9", "v11", "v14"]

FINAL_WEIGHTS = {"v14": 0.15, "v11": 0.15, "v8": 0.20, "v9": 0.20, "v5b": 0.30}


def to_ranks(logits):
    """Per-class rank transform. Column c becomes the rank of each sample's logit for pen c."""
    ranked = np.empty(logits.shape, dtype=np.float64)
    for c in range(logits.shape[1]):
        ranked[:, c] = rankdata(logits[:, c])
    return ranked


def blend(ranks, weights):
    out = np.zeros(next(iter(ranks.values())).shape, dtype=np.float64)
    for name, w in weights.items():
        out += w * ranks[name]
    return out


def optimize(oof_ranks, labels, step=0.05):
    """Grid search the weight simplex for max OOF accuracy. This is how FINAL_WEIGHTS was found.

    At step=0.05 the maximum is a tie between two weight vectors; FINAL_WEIGHTS is one of them.
    """
    n_steps = int(round(1.0 / step))
    results = []
    # Every way to split n_steps units across len(MODELS) models (stars and bars).
    for cuts in combinations(range(n_steps + len(MODELS) - 1), len(MODELS) - 1):
        parts, prev = [], -1
        for c in cuts:
            parts.append(c - prev - 1)
            prev = c
        parts.append(n_steps + len(MODELS) - 2 - prev)
        weights = {m: round(p * step, 4) for m, p in zip(MODELS, parts)}
        acc = accuracy_score(labels, blend(oof_ranks, weights).argmax(axis=1))
        results.append((acc, weights))
    results.sort(key=lambda r: -r[0])
    best_acc = results[0][0]
    ties = [w for a, w in results if a == best_acc]
    print(f"  searched {len(results)} weight combinations at step {step}")
    print(f"  best OOF acc={best_acc:.6f}, reached by {len(ties)} weight vector(s):")
    for w in ties:
        flag = "  <- FINAL_WEIGHTS (the submitted one)" if w == {k: FINAL_WEIGHTS[k] for k in MODELS} else ""
        print(f"    {w}{flag}")
    return best_acc, results[0][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default=None,
                    help="e.g. v14:0.15,v11:0.15,v8:0.20,v9:0.20,v5b:0.30")
    ap.add_argument("--optimize", action="store_true", help="grid search weights on OOF")
    ap.add_argument("--step", type=float, default=0.05, help="grid resolution for --optimize")
    ap.add_argument("--data-dir", type=Path, default=BASE_DIR / "icdar-2026-circleid-pen-classification",
                    help="extracted competition data - needed only to write the submission CSV")
    ap.add_argument("--out", type=Path, default=BASE_DIR / "submission_rank_avg.csv")
    args = ap.parse_args()

    oof = {m: np.load(OOF_DIR / f"oof_logits_{m}.npy").astype(np.float64) for m in MODELS}
    test = {m: np.load(TEST_DIR / f"test_logits_{m}.npy").astype(np.float64) for m in MODELS}
    labels = np.load(OOF_DIR / "oof_labels.npy")
    mask = np.load(OOF_DIR / "oof_mask.npy")

    print("=" * 62)
    print("Individual model OOF scores")
    print("=" * 62)
    for m in MODELS:
        p = oof[m].argmax(axis=1)
        print(f"  {m:4s}  acc={accuracy_score(labels[mask], p[mask]):.4f}"
              f"  macro-F1={f1_score(labels[mask], p[mask], average='macro'):.4f}")

    oof_ranks = {m: to_ranks(oof[m]) for m in MODELS}
    test_ranks = {m: to_ranks(test[m]) for m in MODELS}

    if args.optimize:
        print("\n" + "=" * 62)
        print(f"Grid searching weights (step={args.step})")
        print("=" * 62)
        _, weights = optimize(oof_ranks, labels, args.step)
    elif args.weights:
        weights = {}
        for part in args.weights.split(","):
            name, w = part.split(":")
            weights[name.strip()] = float(w)
        missing = set(weights) - set(MODELS)
        if missing:
            raise SystemExit(f"unknown model(s): {sorted(missing)} - available: {MODELS}")
    else:
        weights = FINAL_WEIGHTS

    print("\n" + "=" * 62)
    print(f"Rank-averaged ensemble  weights={weights}")
    print("=" * 62)
    oof_preds = blend(oof_ranks, weights).argmax(axis=1)
    print(f"  OOF acc={accuracy_score(labels[mask], oof_preds[mask]):.4f}"
          f"  macro-F1={f1_score(labels[mask], oof_preds[mask], average='macro'):.4f}")

    test_preds = blend(test_ranks, weights).argmax(axis=1)
    pen_ids = test_preds + 1  # class index 0..7 -> pen_id 1..8
    dist = {int(p): int((pen_ids == p).sum()) for p in range(1, 9)}
    print(f"  test predictions: {len(pen_ids)}")
    print(f"  pen distribution: {dist}")

    test_csv = args.data_dir / "test.csv"
    if test_csv.exists():
        test_df = pd.read_csv(test_csv)
        if len(test_df) != len(pen_ids):
            raise SystemExit(f"test.csv has {len(test_df)} rows but logits have {len(pen_ids)} - "
                             "make sure you have the v2 dataset (test set was replaced 5 Mar 2026)")
        pd.DataFrame({"image_id": test_df["image_id"], "pen_id": pen_ids}).to_csv(args.out, index=False)
        print(f"\n  submission written: {args.out}")
    else:
        print(f"\n  {test_csv} not found - no CSV written.")
        print("  Point --data-dir at the extracted dataset to emit a submission file.")
        print("  (The scores and distribution above are already fully reproduced.)")


if __name__ == "__main__":
    main()
