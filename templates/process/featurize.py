"""Starter SageMaker Processing script for feature engineering. Agent edits per experiment.

Reads the UNLOADed slice from the Processing input channel, engineers features,
writes deterministic train/val/test parquet to the output channel. Use this
instead of inline pandas when a slice will not comfortably fit or transform in
the agent's own process.

Container contract (set up by sm_research.sm_process):
    /opt/ml/processing/input   <- slice parquet (read-only)
    /opt/ml/processing/output  -> train/, val/, test/ (synced to S3 at job end)
"""
import argparse
import glob
import hashlib
import os

import pandas as pd

INPUT_DIR = "/opt/ml/processing/input"
OUTPUT_DIR = "/opt/ml/processing/output"

ID_COL = "customer_id"


def load_slice(input_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(input_dir, "**", "*.parquet"),
                             recursive=True))
    if not files:
        raise SystemExit(f"no parquet found under {input_dir}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def featurize(df: pd.DataFrame, target: str) -> pd.DataFrame:
    df = df.copy()
    # Redshift DECIMAL arrives as object; make it numeric so models can use it.
    for col in df.columns:
        if col not in (target, ID_COL) and df[col].dtype == object:
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().all():
                df[col] = coerced
    y = df.pop(target).astype(int)
    ids = df.pop(ID_COL) if ID_COL in df.columns else None
    X = pd.get_dummies(df, drop_first=False).astype("float32")
    X["target"] = y.to_numpy()
    if ids is not None:
        X[ID_COL] = ids.to_numpy()
    return X


def _bucket(value, salt: str) -> int:
    """Stable hash bucket 0-99, so splits are identical across rounds and hosts.

    Python's hash() is salted per process and would reshuffle the test set
    between runs, making round-over-round metrics incomparable.
    """
    digest = hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()
    return int(digest, 16) % 100


def split(df: pd.DataFrame, salt: str, val_pct: int, test_pct: int):
    if ID_COL not in df.columns:
        raise SystemExit(f"{ID_COL} required for a deterministic split")
    buckets = df[ID_COL].map(lambda v: _bucket(v, salt))
    test = df[buckets >= 100 - test_pct]
    val = df[(buckets >= 100 - test_pct - val_pct) & (buckets < 100 - test_pct)]
    train = df[buckets < 100 - test_pct - val_pct]
    return train, val, test


def write_split(df: pd.DataFrame, name: str, out_dir: str) -> None:
    dest = os.path.join(out_dir, name)
    os.makedirs(dest, exist_ok=True)
    # Drop the id before training so the model cannot key on it.
    df.drop(columns=[ID_COL], errors="ignore").to_parquet(
        os.path.join(dest, f"{name}.parquet"), index=False)
    print(f"{name}: rows={len(df)} churn_rate={df['target'].mean():.4f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="churned")
    p.add_argument("--seed-salt", default="seed42")
    p.add_argument("--val-pct", type=int, default=15)
    p.add_argument("--test-pct", type=int, default=15)
    p.add_argument("--input-dir", default=INPUT_DIR)
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    args = p.parse_args()

    raw = load_slice(args.input_dir)
    print(f"loaded rows={len(raw)} cols={len(raw.columns)}")
    feats = featurize(raw, args.target)
    print(f"features={len(feats.columns) - 2}")  # minus target and id

    train, val, test = split(feats, args.seed_salt, args.val_pct, args.test_pct)
    for name, part in (("train", train), ("val", val), ("test", test)):
        write_split(part, name, args.output_dir)
    print("featurize complete;")


if __name__ == "__main__":
    main()
