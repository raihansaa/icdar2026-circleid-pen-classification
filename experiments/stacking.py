"""
Stacking / Meta-learner for CircleID Pen Classification

Uses OOF (out-of-fold) predictions from v9, v8, v5b to train a meta-learner
that learns per-class optimal blending.

"""

import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import softmax
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm

BASE_DIR = Path(__file__).resolve().parent.parent  # repo root
DATA_DIR = BASE_DIR / "icdar-2026-circleid-pen-classification"
OUTPUT_DIR = BASE_DIR / "outputs_stacking"
EXCLUDE_WRITERS = {"W41", "W50"}

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Model configs for OOF generation
MODELS = {
    "v9": {
        "backbone": "convnextv2_tiny.fcmae_ft_in22k_in1k",
        "checkpoint_dir": BASE_DIR / "checkpoints_v9",
        "image_size": 336,
        "num_classes": 8,
        "drop_path_rate": 0.2,
        "head_dropout": 0.3,
    },
    "v8": {
        "backbone": "caformer_s36.sail_in22k_ft_in1k_384",
        "checkpoint_dir": BASE_DIR / "checkpoints_v8",
        "image_size": 336,
        "num_classes": 8,
        "drop_path_rate": 0.2,
        "head_dropout": 0.3,
    },
}


class PenModel(nn.Module):
    def __init__(self, backbone_name, num_classes=8, drop_path_rate=0.2,
                 head_dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, num_classes=0,
            drop_path_rate=drop_path_rate,
        )
        embed_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


class ValDataset(Dataset):
    def __init__(self, df, data_dir, transform):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.data_dir / row["image_path"]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, idx


def get_val_transforms(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size),
                          interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def generate_oof():
    """Generate OOF predictions from all models."""
    print("=" * 60)
    print("GENERATING OOF PREDICTIONS")
    print("=" * 60)

    # Load training data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    train_df = train_df[~train_df["writer_id"].isin(EXCLUDE_WRITERS)].reset_index(drop=True)
    pen_ids = sorted(train_df["pen_id"].unique())
    pen2idx = {p: i for i, p in enumerate(pen_ids)}
    n_train = len(train_df)
    n_classes = len(pen_ids)

    print(f"  Train samples: {n_train}")
    print(f"  Pens: {pen_ids}")

    # Same fold split as training
    writers = train_df["writer_id"].values
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(train_df, groups=writers))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, mcfg in MODELS.items():
        print(f"\n  --- {model_name} ---")
        oof_logits = np.zeros((n_train, n_classes), dtype=np.float64)
        folds_done = 0

        for fold_idx in range(5):
            ckpt_path = mcfg["checkpoint_dir"] / f"fold{fold_idx}" / "best_model.pt"
            if not ckpt_path.exists():
                print(f"    Fold {fold_idx}: checkpoint not found, skipping")
                continue

            _, val_indices = folds[fold_idx]
            val_df = train_df.iloc[val_indices]

            # Load model
            ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
            model = PenModel(mcfg["backbone"], mcfg["num_classes"],
                           mcfg["drop_path_rate"], mcfg["head_dropout"])

            state = ckpt["model"]
            if "ema_shadow" in ckpt:
                for n in ckpt["ema_shadow"]:
                    if n in state:
                        state[n] = ckpt["ema_shadow"][n]
            model.load_state_dict(state)
            model.to("cuda")
            model.eval()

            # Run inference on validation set
            tfm = get_val_transforms(mcfg["image_size"])
            ds = ValDataset(val_df, DATA_DIR, tfm)
            loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

            fold_logits = []
            with torch.no_grad():
                for images, _ in loader:
                    images = images.to("cuda", non_blocking=True)
                    with torch.amp.autocast("cuda"):
                        logits = model(images)
                    fold_logits.append(logits.cpu().float().numpy())

            fold_logits = np.concatenate(fold_logits, axis=0)

            # Place into correct positions
            for i, train_idx in enumerate(val_indices):
                oof_logits[train_idx] = fold_logits[i]

            val_preds = fold_logits.argmax(axis=1)
            val_labels = [pen2idx[train_df.iloc[vi]["pen_id"]] for vi in val_indices]
            acc = accuracy_score(val_labels, val_preds)
            print(f"    Fold {fold_idx}: val_acc={acc:.4f} (epoch={ckpt.get('epoch', '?')})")

            folds_done += 1
            del model
            torch.cuda.empty_cache()

        if folds_done == 5:
            np.save(OUTPUT_DIR / f"oof_{model_name}.npy", oof_logits)
            print(f"  Saved: {OUTPUT_DIR / f'oof_{model_name}.npy'}")

            # Overall OOF accuracy
            oof_preds = oof_logits.argmax(axis=1)
            true_labels = np.array([pen2idx[p] for p in train_df["pen_id"]])
            acc = accuracy_score(true_labels, oof_preds)
            f1 = f1_score(true_labels, oof_preds, average="macro")
            print(f"  {model_name} OOF: acc={acc:.4f}, F1={f1:.4f}")
        else:
            print(f"  WARNING: Only {folds_done}/5 folds found for {model_name}")

    # Save labels
    true_labels = np.array([pen2idx[p] for p in train_df["pen_id"]])
    np.save(OUTPUT_DIR / "oof_labels.npy", true_labels)
    print(f"\n  Labels saved: {OUTPUT_DIR / 'oof_labels.npy'}")


def train_meta_learner():
    """Train stacking meta-learner on OOF predictions."""
    print("\n" + "=" * 60)
    print("TRAINING META-LEARNER")
    print("=" * 60)

    # Load OOF predictions (v9 and v8 only — v5b has custom architecture)
    labels = np.load(OUTPUT_DIR / "oof_labels.npy")
    n_samples = len(labels)

    oof_data = {}
    for model_name in ["v9", "v8"]:
        path = OUTPUT_DIR / f"oof_{model_name}.npy"
        if path.exists():
            oof_data[model_name] = np.load(path)
            print(f"  Loaded {model_name}: shape={oof_data[model_name].shape}")

    if len(oof_data) < 2:
        print("ERROR: Need at least 2 OOF files")
        return

    # Build feature matrix: concatenate softmax probabilities from all models
    features_list = []
    for name in sorted(oof_data.keys()):
        probs = softmax(oof_data[name], axis=1)
        features_list.append(probs)
    X = np.hstack(features_list)  # shape: (n_samples, n_models * n_classes)
    y = labels

    print(f"\n  Feature matrix: {X.shape} (samples x features)")
    print(f"  Labels: {y.shape}")

    # Also try with raw logits
    X_logits = np.hstack([oof_data[name] for name in sorted(oof_data.keys())])

    # Load fold splits
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    train_df = train_df[~train_df["writer_id"].isin(EXCLUDE_WRITERS)].reset_index(drop=True)
    writers = train_df["writer_id"].values
    gkf = GroupKFold(n_splits=5)
    fold_splits = list(gkf.split(X, y, groups=writers))

    def cv_evaluate(clf_factory, X_cur, tag=""):
        cv_preds = np.zeros(n_samples, dtype=int)
        for fold_idx, (tr_idx, va_idx) in enumerate(fold_splits):
            clf = clf_factory()
            clf.fit(X_cur[tr_idx], y[tr_idx])
            cv_preds[va_idx] = clf.predict(X_cur[va_idx])
        f1 = f1_score(y, cv_preds, average="macro")
        acc = accuracy_score(y, cv_preds)
        return f1, acc, cv_preds

    best_f1 = 0
    best_params = {}
    best_clf_factory = None
    best_X = None

    # --- LogisticRegression ---
    print("\n  LogisticRegression:")
    for mode, X_cur in [("probs", X), ("logits", X_logits)]:
        for C in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
            factory = lambda C=C: LogisticRegression(
                C=C, max_iter=1000, solver="lbfgs",
                multi_class="multinomial", random_state=42)
            f1, acc, _ = cv_evaluate(factory, X_cur)

            if f1 > best_f1:
                best_f1 = f1
                best_params = {"type": "logreg", "mode": mode, "C": C}
                best_clf_factory = factory
                best_X = X_cur

            print(f"    {mode} C={C:>5.2f}: CV acc={acc:.4f}, F1={f1:.4f}"
                  f"{' *' if f1 == best_f1 else ''}")

    # --- XGBoost ---
    try:
        from xgboost import XGBClassifier
        print("\n  XGBoost:")
        for mode, X_cur in [("probs", X), ("logits", X_logits)]:
            for n_est in [50, 100, 200]:
                for max_depth in [3, 5]:
                    for lr in [0.05, 0.1]:
                        factory = lambda n=n_est, d=max_depth, l=lr: XGBClassifier(
                            n_estimators=n, max_depth=d, learning_rate=l,
                            objective="multi:softmax", num_class=8,
                            random_state=42, verbosity=0,
                            eval_metric="mlogloss")
                        f1, acc, _ = cv_evaluate(factory, X_cur)

                        if f1 > best_f1:
                            best_f1 = f1
                            best_params = {"type": "xgb", "mode": mode,
                                          "n_est": n_est, "max_depth": max_depth, "lr": lr}
                            best_clf_factory = factory
                            best_X = X_cur

                        print(f"    {mode} n={n_est:>3} d={max_depth} lr={lr}: "
                              f"CV acc={acc:.4f}, F1={f1:.4f}"
                              f"{' *' if f1 == best_f1 else ''}")
    except ImportError:
        print("\n  XGBoost not installed — skipping. Install with: pip install xgboost")

    print(f"\n  Best: {best_params}, F1={best_f1:.4f}")

    # Compare with simple weighted average (v9+v8 only, no v5b OOF)
    blend_weights = {"v9": 0.65, "v8": 0.35}  # renormalized without v5b
    blended = np.zeros((n_samples, 8), dtype=np.float64)
    for name in sorted(oof_data.keys()):
        if name in blend_weights:
            probs = softmax(oof_data[name], axis=1)
            blended += blend_weights[name] * probs
    blend_preds = blended.argmax(axis=1)
    blend_f1 = f1_score(y, blend_preds, average="macro")
    blend_acc = accuracy_score(y, blend_preds)
    print(f"  Simple weighted avg (v9:65/v8:35): acc={blend_acc:.4f}, F1={blend_f1:.4f}")

    # Per-pen comparison
    print(f"\n  Per-pen accuracy comparison:")
    pen_ids = sorted(pd.read_csv(DATA_DIR / "train.csv")["pen_id"].unique())

    # Retrain best meta-learner on full data
    clf_final = best_clf_factory()
    clf_final.fit(best_X, y)

    # CV predictions for per-pen analysis
    _, _, meta_cv_preds = cv_evaluate(best_clf_factory, best_X)

    for i, pen in enumerate(pen_ids):
        mask = y == i
        meta_acc = (meta_cv_preds[mask] == i).mean()
        blend_acc_pen = (blend_preds[mask] == i).mean()
        diff = meta_acc - blend_acc_pen
        marker = " <<<" if abs(diff) > 0.01 else ""
        print(f"    Pen {pen}: meta={meta_acc:.3f}, blend={blend_acc_pen:.3f}, diff={diff:+.3f}{marker}")

    # Save the final model
    import pickle
    with open(OUTPUT_DIR / "meta_learner.pkl", "wb") as f:
        pickle.dump({"clf": clf_final, "params": best_params,
                      "model_order": sorted(oof_data.keys())}, f)
    print(f"\n  Meta-learner saved: {OUTPUT_DIR / 'meta_learner.pkl'}")


def apply_meta_learner():
    """Apply meta-learner to test logits."""
    print("\n" + "=" * 60)
    print("APPLYING META-LEARNER TO TEST")
    print("=" * 60)

    import pickle
    with open(OUTPUT_DIR / "meta_learner.pkl", "rb") as f:
        meta = pickle.load(f)

    clf = meta["clf"]
    params = meta["params"]
    mode = params["mode"]
    model_order = meta["model_order"]
    print(f"  Params: {params}")
    print(f"  Models: {model_order}")

    # Load test logits
    test_logit_paths = {
        "v5b": BASE_DIR / "outputs_v5b" / "test_logits_v5b.npy",
        "v8": BASE_DIR / "outputs_v8" / "test_logits_v8.npy",
        "v9": BASE_DIR / "outputs_v9" / "test_logits_v9.npy",
    }

    features_list = []
    for name in model_order:
        logits = np.load(test_logit_paths[name])
        if mode == "probs":
            features_list.append(softmax(logits, axis=1))
        else:
            features_list.append(logits)
        print(f"  Loaded {name}: shape={logits.shape}")

    X_test = np.hstack(features_list)
    print(f"  Test features: {X_test.shape}")

    # Predict
    predictions = clf.predict(X_test)
    proba = clf.predict_proba(X_test)

    # Map to pen IDs
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    pen_list = sorted(train_df["pen_id"].unique())
    pen_preds = [pen_list[p] for p in predictions]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({"image_id": test_df["image_id"], "pen_id": pen_preds})
    out_path = OUTPUT_DIR / "submission_stacking.csv"
    sub.to_csv(out_path, index=False)

    print(f"\n  Submission saved: {out_path}")
    print(f"  Pen distribution:")
    print(sub["pen_id"].value_counts().sort_index().to_string())

    # Compare with simple blend
    blend_weights = {"v9": 0.55, "v8": 0.30, "v5b": 0.15}
    blended = np.zeros_like(proba, dtype=np.float64)
    for name in model_order:
        logits = np.load(test_logit_paths[name])
        probs = softmax(logits, axis=1)
        blended += blend_weights.get(name, 1/3) * probs
    blend_preds = blended.argmax(axis=1)
    agree = (predictions == blend_preds).sum()
    print(f"\n  Agreement with simple blend: {agree}/{len(predictions)} ({agree/len(predictions)*100:.1f}%)")
    disagree = (predictions != blend_preds).sum()
    print(f"  Disagreements: {disagree}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-oof", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        generate_oof()
        train_meta_learner()
        apply_meta_learner()
    elif args.generate_oof:
        generate_oof()
    elif args.train:
        train_meta_learner()
    elif args.apply:
        apply_meta_learner()
    else:
        print("Usage:")
        print("  python stacking.py --generate-oof   # generate OOF predictions")
        print("  python stacking.py --train           # train meta-learner")
        print("  python stacking.py --apply           # apply to test")
        print("  python stacking.py --all             # all steps")


if __name__ == "__main__":
    main()
