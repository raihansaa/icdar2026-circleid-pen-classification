"""
Ensemble v5b + v7 logits for submission.

Supports loading multiple logit files (v5, v5b, v7) and blending them.

"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from scipy.special import softmax

BASE_DIR = Path(__file__).resolve().parent.parent  # repo root
DATA_DIR = BASE_DIR / "icdar-2026-circleid-pen-classification"
OUTPUT_DIR = BASE_DIR / "outputs_ensemble"

# All possible logit sources 
LOGIT_SOURCES = {
    "v5b":            BASE_DIR / "outputs_v5b" / "test_logits_v5b.npy",
    "v7":             BASE_DIR / "outputs_v7"  / "test_logits_v7.npy",
    "convnext_tiny":  BASE_DIR / "outputs_convnextv2-tiny" / "test_logits_convnextv2-tiny.npy",
    "v8":             BASE_DIR / "outputs_v8" / "test_logits_v8.npy",
    "v9":             BASE_DIR / "outputs_v9" / "test_logits_v9.npy",
    "v10":            BASE_DIR / "outputs_v10" / "test_logits_v10.npy",
    "convnext_base":  BASE_DIR / "outputs_convnext_base" / "test_logits_convnext_base.npy",
    "v11":            BASE_DIR / "outputs_v11" / "test_logits_v11.npy",
    "v11ms":          BASE_DIR / "outputs_v11_multiseed" / "test_logits_v11_seeds42_43.npy",
    "v11s43":         BASE_DIR / "outputs_v11_seed43" / "test_logits_v11.npy",
    "v14":            BASE_DIR / "outputs_v14_seed42" / "test_logits_v14.npy",
    "v16":            BASE_DIR / "outputs_v16" / "test_logits_v16.npy",
}


def find_available_logits():
    """Find all available logit files."""
    available = {}
    for name, path in LOGIT_SOURCES.items():
        if path.exists():
            available[name] = np.load(path)
            print(f"  Found {name}: {path} (shape={available[name].shape})")
    return available


def make_submission(blended_logits, test_df, pen_list, tag="ensemble"):
    preds = blended_logits.argmax(axis=1)
    pen_preds = [pen_list[p] for p in preds]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({"image_id": test_df["image_id"], "pen_id": pen_preds})
    out_path = OUTPUT_DIR / f"submission_{tag}.csv"
    sub.to_csv(out_path, index=False)
    print(f"\nSubmission saved: {out_path}")
    print(f"Pen distribution:")
    print(sub["pen_id"].value_counts().sort_index().to_string())
    return sub


def agreement_stats(logits_dict):
    """Show pairwise agreement between all models."""
    names = list(logits_dict.keys())
    preds = {n: l.argmax(axis=1) for n, l in logits_dict.items()}
    total = len(next(iter(preds.values())))

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            agree = (preds[a] == preds[b]).sum()
            print(f"\n{a} vs {b}: agree {agree}/{total} ({agree/total*100:.1f}%), "
                  f"disagree {total - agree}")

            disagree_mask = preds[a] != preds[b]
            if disagree_mask.any():
                transitions = Counter()
                for idx in np.where(disagree_mask)[0]:
                    transitions[(int(preds[a][idx]), int(preds[b][idx]))] += 1
                print(f"  Top disagreements ({a}_pred -> {b}_pred):")
                for (pa, pb), count in transitions.most_common(10):
                    print(f"    pen_idx {pa} -> {pb}: {count}")


def blend_logits(logits_dict, weights=None, use_softmax=True):
    """Blend logits with optional probability-space averaging."""
    names = list(logits_dict.keys())
    if weights is None:
        weights = {n: 1.0 / len(names) for n in names}

    if use_softmax:
        # Probability-space averaging (theoretically better)
        blended = np.zeros_like(next(iter(logits_dict.values())), dtype=np.float64)
        for name in names:
            probs = softmax(logits_dict[name], axis=1)
            blended += weights[name] * probs
        return blended
    else:
        # Logit-space averaging
        blended = np.zeros_like(next(iter(logits_dict.values())), dtype=np.float64)
        for name in names:
            blended += weights[name] * logits_dict[name]
        return blended


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--w5", type=float, default=None, help="Weight for v5/v5b logits")
    parser.add_argument("--w7", type=float, default=None, help="Weight for v7/other logits")
    parser.add_argument("--weights", type=str, default=None,
                        help="Custom weights as name:weight pairs, e.g. 'convnext_tiny:0.6,v5b:0.2,v8:0.2'")
    parser.add_argument("--sweep", action="store_true", help="Sweep weights and show stats (2 models only)")
    parser.add_argument("--logit-space", action="store_true",
                        help="Blend in logit space (default: probability space)")
    args = parser.parse_args()

    print("Searching for available logit files...")
    available = find_available_logits()

    if len(available) < 2:
        print(f"\nERROR: Need at least 2 logit files. Found: {list(available.keys())}")
        print("Run inference first:")
        print("  python circleid_pen_v5.py --infer-only")
        print("  python circleid_pen_v7.py --infer-only")
        return

    # Use v5b over v5 if both exist
    if "v5b" in available and "v5" in available:
        print("\nBoth v5 and v5b found — using v5b (fixed version)")
        del available["v5"]

    # Verify shapes match
    shapes = {n: l.shape for n, l in available.items()}
    ref_shape = next(iter(shapes.values()))
    for name, shape in shapes.items():
        assert shape == ref_shape, f"Shape mismatch: {name}={shape} vs {ref_shape}"

    # Load test CSV and pen mapping
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    pen_list = sorted(train_df["pen_id"].unique())
    print(f"\nTest samples: {len(test_df)}, Pens: {pen_list}")

    agreement_stats(available)

    names = sorted(available.keys())
    use_softmax = not args.logit_space

    if args.sweep:
        # For 2 models, sweep w1 from 0.0 to 1.0
        if len(names) != 2:
            print(f"\nSweep only works with exactly 2 models. Found: {names}")
            return
        n1, n2 = names[0], names[1]
        p1 = available[n1].argmax(axis=1)
        p2 = available[n2].argmax(axis=1)

        mode = "probability" if use_softmax else "logit"
        print(f"\n{'='*60}")
        print(f"Weight sweep ({mode}-space blending)")
        print(f"{'='*60}")
        for w1 in np.arange(0.0, 1.05, 0.1):
            w2 = 1.0 - w1
            weights = {n1: w1, n2: w2}
            blended = blend_logits(available, weights, use_softmax=use_softmax)
            preds = blended.argmax(axis=1)
            match1 = (preds == p1).sum()
            match2 = (preds == p2).sum()
            print(f"  {n1}={w1:.1f} {n2}={w2:.1f} | "
                  f"match_{n1}={match1}/{len(preds)} | "
                  f"match_{n2}={match2}/{len(preds)} | "
                  f"unique_blend={len(preds) - max(match1, match2)}")
        print(f"\nRe-run with --w5 X --w7 Y to generate submission.")
        return

    # Build weight dict
    if args.weights:
        # Parse custom weights: "convnext_tiny:0.6,v5b:0.2,v8:0.2"
        weights = {}
        for pair in args.weights.split(","):
            name, w = pair.strip().split(":")
            if name in available:
                weights[name] = float(w)
            else:
                print(f"WARNING: '{name}' not found in available models: {list(available.keys())}")
        total_w = sum(weights.values())
        weights = {n: w / total_w for n, w in weights.items()}
    elif args.w5 is not None or args.w7 is not None:
        v5_name = "v5b" if "v5b" in available else "v5"
        other_name = [n for n in names if n != v5_name][0]
        w5 = args.w5 if args.w5 is not None else 0.5
        w7 = args.w7 if args.w7 is not None else 0.5
        total_w = w5 + w7
        weights = {v5_name: w5 / total_w, other_name: w7 / total_w}
    else:
        # Equal weights
        weights = {n: 1.0 / len(names) for n in names}

    # Filter available logits to only models in weights
    blend_dict = {n: available[n] for n in weights if n in available}
    mode = "probability" if use_softmax else "logit"
    print(f"\nBlending ({mode}-space): {weights}")
    blended = blend_logits(blend_dict, weights, use_softmax=use_softmax)

    # Build tag
    parts = [f"{n}w{int(weights[n]*100)}" for n in sorted(weights.keys())]
    tag = "_".join(parts) + f"_{'prob' if use_softmax else 'logit'}"
    make_submission(blended, test_df, pen_list, tag=tag)


if __name__ == "__main__":
    main()
