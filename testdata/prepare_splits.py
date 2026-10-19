#!/usr/bin/env python3
"""Split an unloaded slice into train/test parquet under the slice prefix.

Mirrors what a dry-run experiment does inline (small data, no Processing job):
read the UNLOADed parquet, one-hot encode the categoricals, drop the id column,
and write `train/` and `test/` subprefixes where the training templates look
(SM_CHANNEL_TRAINING/{train,test}).

The split is seeded so rounds stay comparable, and the target column is renamed
to `target` to match the template's default.
"""
import argparse
import io

import boto3
import pandas as pd
from sklearn.model_selection import train_test_split

ID_COL = "customer_id"


def _split_uri(uri: str) -> tuple[str, str]:
    without = uri[len("s3://"):]
    bucket, _, key = without.partition("/")
    return bucket, key


def read_slice(s3, prefix: str) -> pd.DataFrame:
    bucket, key = _split_uri(prefix)
    objs = s3.list_objects_v2(Bucket=bucket, Prefix=key).get("Contents", [])
    parquets = [o["Key"] for o in objs if o["Key"].endswith(".parquet")]
    if not parquets:
        raise SystemExit(f"no parquet found under {prefix}")
    frames = []
    for k in parquets:
        body = s3.get_object(Bucket=bucket, Key=k)["Body"].read()
        frames.append(pd.read_parquet(io.BytesIO(body)))
    return pd.concat(frames, ignore_index=True)


def featurize(df: pd.DataFrame, target: str) -> pd.DataFrame:
    df = df.drop(columns=[ID_COL], errors="ignore")
    # UNLOAD emits Redshift DECIMAL as object; make it numeric for XGBoost.
    for col in df.columns:
        if col != target and df[col].dtype == object:
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().all():
                df[col] = coerced
    y = df.pop(target).astype(int)
    X = pd.get_dummies(df, drop_first=False).astype("float32")
    X["target"] = y
    return X


def write_split(s3, df: pd.DataFrame, prefix: str, name: str) -> str:
    bucket, key = _split_uri(prefix)
    dest = f"{key.rstrip('/')}/{name}/{name}.parquet"
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=dest, Body=buf.getvalue())
    uri = f"s3://{bucket}/{dest}"
    print(f"wrote {uri}  rows={len(df):,} cols={len(df.columns)}")
    return uri


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--region", required=True)
    ap.add_argument("--slice-prefix", required=True)
    ap.add_argument("--target", default="churned")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    s3 = boto3.Session(profile_name=args.profile,
                       region_name=args.region).client("s3")
    raw = read_slice(s3, args.slice_prefix)
    print(f"read slice: {raw.shape[0]:,} rows x {raw.shape[1]} cols")

    feats = featurize(raw, args.target)
    train, test = train_test_split(
        feats, test_size=args.test_size, random_state=args.seed,
        stratify=feats["target"])

    write_split(s3, train, args.slice_prefix, "train")
    write_split(s3, test, args.slice_prefix, "test")
    print(f"\nfeatures: {len(feats.columns) - 1}")
    print(f"train churn rate {train['target'].mean():.4f} / "
          f"test {test['target'].mean():.4f}")


if __name__ == "__main__":
    main()
