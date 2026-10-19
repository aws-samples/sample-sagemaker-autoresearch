"""Starter scikit-learn training script for SageMaker.

Runs as-is on a `redshift_unload` slice: when the hypothesis lives entirely in
select_sql + hyperparameters, pass this file straight to run_training. Copy it
to your own path only for in-script feature engineering (edit `_features`).

Data contract: the training channel holds the UNLOAD slice -- flat *.parquet,
no split directories. `_splits` shuffles it deterministically and splits it
60/20/20 (train/val/test) stratified on the target with seed 42, so a slice
always yields the same test rows. A channel that already has train/ and test/
subdirectories (e.g. Processing output) is used as given. The target column
name arrives as --target (run_training passes data.target automatically).

`--model rf` (RandomForest) or `hgb` (HistGradientBoosting); tunes the decision
threshold for F1 on val, prints the held-out test metric as `f1=<value>;`.
"""
import argparse
import glob
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

SEED = 42


def _load(path):
    files = sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no *.parquet files under {path}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _splits(base, target):
    """(train, val, test) frames from the training channel (see module doc)."""
    train_dir, test_dir = os.path.join(base, "train"), os.path.join(base, "test")
    if os.path.isdir(train_dir) and os.path.isdir(test_dir):
        rest, test = _load(train_dir), _load(test_dir)
    else:
        df = _load(base)
        # UNLOAD row order is not guaranteed; order by row content first.
        order = pd.util.hash_pandas_object(df.astype(str), index=False)
        df = df.loc[order.sort_values(kind="mergesort").index].reset_index(drop=True)
        rest, test = train_test_split(df, test_size=0.2, random_state=SEED,
                                      stratify=df[target])
    train, val = train_test_split(rest, test_size=0.25, random_state=SEED,
                                  stratify=rest[target])
    return train, val, test


def _features(df):
    """Per-experiment feature hook, applied to each split separately (row-wise
    transforms only). Join keys (`id`, `*_id`) are not features."""
    return df.drop(columns=[c for c in df.columns if c == "id" or c.endswith("_id")])


# Redshift DECIMAL columns (prices, AVG/SUM aggregates) arrive from Parquet as
# object columns of Python Decimals. Treating them as categorical one-hot encodes
# every distinct value and kills the job with an out-of-memory error that leaves
# no traceback. Keep this step when you edit the script.
MAX_ONE_HOT_LEVELS = 50


def _prepare(train, test):
    """Numeric-like object columns -> float; true categoricals -> one-hot (low
    cardinality) or integer codes (high cardinality); identical train/test columns."""
    train, test = train.copy(), test.copy()
    for col in train.columns:
        # Any non-numeric, non-bool column: object (Decimals, strings), pandas 3's
        # str dtype, or category.
        if pd.api.types.is_numeric_dtype(train[col]) or pd.api.types.is_bool_dtype(train[col]):
            continue
        as_num = pd.to_numeric(train[col], errors="coerce")
        if as_num.notna().mean() >= 0.95:
            train[col] = as_num.astype("float64")
            test[col] = pd.to_numeric(test[col], errors="coerce").astype("float64")
            continue
        levels = pd.Index(train[col].astype(str).unique())
        if len(levels) <= MAX_ONE_HOT_LEVELS:
            for lv in levels:
                train[f"{col}={lv}"] = (train[col].astype(str) == lv).astype("int8")
                test[f"{col}={lv}"] = (test[col].astype(str) == lv).astype("int8")
            train, test = train.drop(columns=col), test.drop(columns=col)
        else:
            codes = {lv: i for i, lv in enumerate(levels)}
            train[col] = train[col].astype(str).map(codes).astype("int32")
            test[col] = test[col].astype(str).map(codes).fillna(-1).astype("int32")
    for col in train.columns:
        if train[col].dtype == bool:
            train[col], test[col] = train[col].astype("int8"), test[col].astype("int8")
    return train, test[train.columns]


def _best_threshold(y, proba):
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y, (proba >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=("rf", "hgb"), default="rf")
    p.add_argument("--n-estimators", type=int, default=300)  # rf
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--min-samples-leaf", type=int, default=5)  # rf
    p.add_argument("--learning-rate", type=float, default=0.05)  # hgb
    p.add_argument("--max-iter", type=int, default=300)  # hgb
    p.add_argument("--class-weight", default="balanced", help="'balanced' or 'none'")
    p.add_argument("--target", default=os.environ.get("SM_HP_TARGET", "target"))
    args, _ = p.parse_known_args()

    base = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
    train, val, test = _splits(base, args.target)
    y_tr, y_val, y_te = (d.pop(args.target).astype(float).astype(int)
                         for d in (train, val, test))
    train, val, test = (_features(d) for d in (train, val, test))
    X_tr, X_val = _prepare(train, val)
    _, X_te = _prepare(train, test)

    cw = None if args.class_weight == "none" else args.class_weight
    if args.model == "hgb":
        clf = HistGradientBoostingClassifier(
            learning_rate=args.learning_rate, max_iter=args.max_iter,
            max_depth=args.max_depth, early_stopping=True, random_state=SEED)
        weights = None
        if cw == "balanced":
            pos = max(float(y_tr.mean()), 1e-6)
            weights = np.where(y_tr == 1, 0.5 / pos, 0.5 / max(1 - pos, 1e-6))
        clf.fit(X_tr, y_tr, sample_weight=weights)
    else:
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators, max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf, class_weight=cw,
            n_jobs=-1, random_state=SEED)
        clf.fit(X_tr, y_tr)

    p_val = clf.predict_proba(X_val)[:, 1]
    thr = _best_threshold(y_val, p_val)
    preds = (clf.predict_proba(X_te)[:, 1] >= thr).astype(int)
    val_f1 = f1_score(y_val, (p_val >= thr).astype(int))
    print(f"val_f1={val_f1};threshold={thr};")
    # The held-out test metric, printed exactly once; run_training parses it.
    print(f"f1={f1_score(y_te, preds)};")

    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    joblib.dump(clf, os.path.join(model_dir, "model.joblib"))


if __name__ == "__main__":
    main()
