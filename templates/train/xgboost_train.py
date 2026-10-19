"""Starter XGBoost training script for SageMaker.

Runs as-is on a `redshift_unload` slice: when the hypothesis lives entirely in
select_sql + hyperparameters, pass this file straight to run_training. Copy it
to your own path only for in-script feature engineering (edit `_features`).

Data contract: the training channel holds the UNLOAD slice -- flat *.parquet,
no split directories. `_splits` shuffles it deterministically and splits it
60/20/20 (train/val/test) stratified on the target with seed 42, so a slice
always yields the same test rows. A channel that already has train/ and test/
subdirectories (e.g. Processing output) is used as given. The target column
name arrives as --target (run_training passes data.target automatically).

Trains with early stopping on val, tunes the decision threshold for F1 on val,
prints the held-out test metric as `f1=<value>;`, and saves the model.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import xgboost as xgb
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
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--eta", type=float, default=0.05)
    p.add_argument("--num-round", type=int, default=500)
    p.add_argument("--early-stopping-rounds", type=int, default=30)
    p.add_argument("--subsample", type=float, default=1.0)
    p.add_argument("--colsample-bytree", type=float, default=1.0)
    p.add_argument("--min-child-weight", type=float, default=1.0)
    p.add_argument("--scale-pos-weight", type=float, default=-1.0,
                   help="<= 0: use the train split's negative/positive ratio")
    p.add_argument("--target", default=os.environ.get("SM_HP_TARGET", "target"))
    args, _ = p.parse_known_args()

    base = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
    train, val, test = _splits(base, args.target)
    y_tr, y_val, y_te = (d.pop(args.target).astype(float).astype(int)
                         for d in (train, val, test))
    train, val, test = (_features(d) for d in (train, val, test))
    X_tr, X_val = _prepare(train, val)
    _, X_te = _prepare(train, test)

    spw = args.scale_pos_weight
    if spw <= 0:
        spw = float((y_tr == 0).sum()) / max(int(y_tr.sum()), 1)
    params = {"max_depth": args.max_depth, "eta": args.eta,
              "subsample": args.subsample, "colsample_bytree": args.colsample_bytree,
              "min_child_weight": args.min_child_weight, "scale_pos_weight": spw,
              "objective": "binary:logistic", "eval_metric": "logloss", "seed": SEED}
    dval = xgb.DMatrix(X_val, label=y_val)
    model = xgb.train(params, xgb.DMatrix(X_tr, label=y_tr),
                      num_boost_round=args.num_round, evals=[(dval, "val")],
                      early_stopping_rounds=args.early_stopping_rounds,
                      verbose_eval=False)
    rounds = (0, model.best_iteration + 1)

    p_val = model.predict(dval, iteration_range=rounds)
    thr = _best_threshold(y_val, p_val)
    preds = (model.predict(xgb.DMatrix(X_te), iteration_range=rounds) >= thr).astype(int)
    val_f1 = f1_score(y_val, (p_val >= thr).astype(int))
    print(f"val_f1={val_f1};threshold={thr};best_iteration={model.best_iteration};")
    # The held-out test metric, printed exactly once; run_training parses it.
    print(f"f1={f1_score(y_te, preds)};")

    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    model.save_model(os.path.join(model_dir, "xgboost-model.json"))


if __name__ == "__main__":
    main()
