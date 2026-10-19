#!/usr/bin/env python3
"""Upload the generated CSVs to S3 and COPY them into Redshift.

Runs entirely through the Redshift Data API (no VPC connectivity needed), using
the namespace admin secret for superuser auth and the dual-trust IAM role for COPY.

Usage:
  uv run load_to_redshift.py --profile oregon --region us-west-2 \
      --bucket sm-auto-research-<account>-<region> \
      --workgroup autoresearch-wg --database autoresearch \
      --secret-arn <arn> --role-arn <arn>
"""
import argparse
import pathlib
import re
import sys
import time

import boto3

SCHEMA = "churn"

# DDL mirrors the dtypes written by generate_churn_data.py.
TABLES: dict[str, str] = {
    "customers": """
        create table {schema}.customers (
            customer_id    integer not null,
            age            integer,
            region         varchar(32),
            signup_channel varchar(32),
            tenure_months  integer,
            primary key (customer_id)
        )
    """,
    "subscriptions": """
        create table {schema}.subscriptions (
            customer_id   integer not null,
            plan_tier     varchar(32),
            monthly_price decimal(10,2),
            contract_type varchar(32),
            autopay       boolean,
            primary key (customer_id)
        )
    """,
    "usage_monthly": """
        create table {schema}.usage_monthly (
            customer_id     integer not null,
            month_num       integer,
            avg_minutes     decimal(12,2),
            support_tickets integer,
            late_payments   integer,
            data_gb         decimal(12,2)
        )
    """,
    "labels": """
        create table {schema}.labels (
            customer_id integer not null,
            churned     integer,
            primary key (customer_id)
        )
    """,
}

# Load order is irrelevant to correctness but keeps the verify output readable.
LOAD_ORDER = ["customers", "subscriptions", "usage_monthly", "labels"]


def upload_csvs(s3, csv_dir: pathlib.Path, bucket: str, prefix: str) -> None:
    for name in LOAD_ORDER:
        path = csv_dir / f"{name}.csv"
        if not path.exists():
            sys.exit(f"missing {path} — run generate_churn_data.py first")
        key = f"{prefix}/{name}/{name}.csv"
        s3.upload_file(str(path), bucket, key)
        size_mb = path.stat().st_size / 1e6
        print(f"uploaded s3://{bucket}/{key}  ({size_mb:.1f} MB)")


class DataApi:
    """Thin Redshift Data API wrapper that waits for each statement."""

    def __init__(self, client, workgroup: str, database: str, secret_arn: str):
        self.client = client
        self.kwargs = {
            "WorkgroupName": workgroup,
            "Database": database,
            "SecretArn": secret_arn,
        }

    def run(self, sql: str, label: str = "", timeout: int = 900) -> str:
        started = self.client.execute_statement(Sql=sql, **self.kwargs)
        sid = started["Id"]
        deadline = time.time() + timeout
        while True:
            desc = self.client.describe_statement(Id=sid)
            status = desc["Status"]
            if status == "FINISHED":
                if label:
                    print(f"  ok: {label}")
                return sid
            if status in ("FAILED", "ABORTED"):
                sys.exit(f"statement {status} ({label}): {desc.get('Error')}\nSQL: {sql[:400]}")
            if time.time() > deadline:
                sys.exit(f"timed out after {timeout}s waiting for {label or sid}")
            # A cold serverless workgroup can take several minutes to resume.
            time.sleep(5)

    def scalar(self, sql: str) -> str | None:
        sid = self.run(sql)
        recs = self.client.get_statement_result(Id=sid)["Records"]
        if not recs:
            return None
        return next(iter(recs[0][0].values()))


_COPY_ARG_PATTERNS = {
    "bucket": r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]",
    "prefix": r"[A-Za-z0-9._/-]+",
    "region": r"[a-z]{2}(-[a-z]+)+-\d",
    "role_arn": r"arn:aws[a-z-]*:iam::\d{12}:role/[\w+=,.@/-]+",
}


def _validate_copy_args(args) -> None:
    # These values are interpolated into the COPY statement, so reject anything
    # that isn't shaped like a bucket, prefix, region, or role ARN.
    for name, pattern in _COPY_ARG_PATTERNS.items():
        if not re.fullmatch(pattern, getattr(args, name)):
            sys.exit(f"invalid --{name.replace('_', '-')}: {getattr(args, name)!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--region", required=True)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--workgroup", required=True)
    ap.add_argument("--database", required=True)
    ap.add_argument("--secret-arn", required=True)
    ap.add_argument("--role-arn", required=True)
    ap.add_argument("--prefix", default="raw-csv")
    ap.add_argument("--csv-dir", default=str(pathlib.Path(__file__).parent / "csv"))
    ap.add_argument("--skip-upload", action="store_true")
    args = ap.parse_args()
    _validate_copy_args(args)

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    s3 = session.client("s3")
    api = DataApi(session.client("redshift-data"), args.workgroup,
                  args.database, args.secret_arn)

    csv_dir = pathlib.Path(args.csv_dir)
    if not args.skip_upload:
        upload_csvs(s3, csv_dir, args.bucket, args.prefix)

    print(f"\ncreating schema {SCHEMA} (first query may take minutes if the "
          f"workgroup is cold)")
    api.run(f"create schema if not exists {SCHEMA}", f"schema {SCHEMA}")

    print("\ncreating tables")
    for name in LOAD_ORDER:
        # Idempotent: drop and recreate so re-running gives a clean load.
        api.run(f"drop table if exists {SCHEMA}.{name}", f"drop {name}")
        api.run(TABLES[name].format(schema=SCHEMA), f"create {name}")

    print("\ncopying data from S3")
    for name in LOAD_ORDER:
        # Schema/table are constants; the rest was checked by _validate_copy_args.
        copy_sql = (
            f"copy {SCHEMA}.{name} from 's3://{args.bucket}/{args.prefix}/{name}/' "  # nosec B608
            f"iam_role '{args.role_arn}' "
            f"format as csv ignoreheader 1 region '{args.region}' "
            f"blanksasnull emptyasnull"
        )
        api.run(copy_sql, f"copy {name}")

    print("\nverifying row counts")
    for name in LOAD_ORDER:
        count = api.scalar(f"select count(*) from {SCHEMA}.{name}")  # nosec B608 - constants
        print(f"  {SCHEMA}.{name:<16} {int(count):>8,} rows")

    rate = api.scalar(
        f"select round(avg(churned::float), 4) from {SCHEMA}.labels")  # nosec B608 - constant
    print(f"\nchurn base rate in Redshift: {rate}")
    print("\nload complete.")


if __name__ == "__main__":
    main()
