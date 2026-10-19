#!/usr/bin/env python3
"""Write a research.yaml for the churn test bed from deployed stack outputs.

Reads the CloudFormation outputs of AutoResearchTestBed, discovers the Redshift
admin secret ARN (not available as a CFN attribute in this CDK version), and
writes a config the plugin can validate and run.

Usage:
  uv run write_research_yaml.py --profile oregon --region us-west-2 \
      --out ../research.yaml
"""
import argparse
import pathlib
import sys

import boto3

STACK_NAME = "AutoResearchTestBed"
NAMESPACE_NAME = "autoresearch"
SCHEMA = "churn"

# Threshold chosen from measured held-out F1 (generate_churn_data.py --report):
#   single tables      0.54 - 0.62
#   all three joined   0.73
# 0.70 sits above every single-table ceiling but below the joined ceiling, so the
# loop cannot reach the goal without discovering the join + usage aggregation.
THRESHOLD = 0.70


def stack_outputs(cfn, stack_name: str) -> dict[str, str]:
    try:
        stacks = cfn.describe_stacks(StackName=stack_name)["Stacks"]
    except Exception as e:  # noqa: BLE001
        sys.exit(f"could not read stack {stack_name}: {e}\nDeploy the CDK stack first.")
    return {o["OutputKey"]: o["OutputValue"]
            for o in stacks[0].get("Outputs", [])}


def admin_secret_arn(rss, namespace_name: str) -> str:
    ns = rss.get_namespace(namespaceName=namespace_name)["namespace"]
    arn = ns.get("adminPasswordSecretArn")
    if not arn:
        sys.exit(f"namespace {namespace_name} has no adminPasswordSecretArn; "
                 "was it created with manageAdminPassword=true?")
    return arn


def build_config(out: dict[str, str], secret_arn: str, profile: str,
                 region: str) -> dict:
    return {
        "aws": {"profile": profile, "region": region},
        "data": {
            "redshift": {
                "workgroup_or_cluster": out["WorkgroupName"],
                "database": out["DatabaseName"],
                "serverless": True,
                "secret_arn": secret_arn,
            },
            # The allowlist spans all three feature tables plus labels, so the
            # agent must decide how to join and aggregate them.
            "allowed": [
                {"schema": SCHEMA, "table": "customers",
                 "columns": ["customer_id", "age", "region", "signup_channel",
                             "tenure_months"]},
                {"schema": SCHEMA, "table": "subscriptions",
                 "columns": ["customer_id", "plan_tier", "monthly_price",
                             "contract_type", "autopay"]},
                {"schema": SCHEMA, "table": "usage_monthly",
                 "columns": ["customer_id", "month_num", "avg_minutes",
                             "support_tickets", "late_payments", "data_gb"]},
                {"schema": SCHEMA, "table": "labels",
                 "columns": ["customer_id", "churned"]},
            ],
            "target": "churned",
        },
        "problem": {
            "type": "binary_classification",
            "success_metric": "f1",
            "threshold": THRESHOLD,
            "metric_goal": "maximize",
        },
        "budget": {
            "max_rounds": 4,
            "experiments_per_round": 3,
            "max_total_cost_usd": 25,
            "max_wallclock_hours": 3,
            "plateau": {"patience_rounds": 2, "min_delta": 0.005},
        },
        "sagemaker": {
            "role_arn": out["RoleArn"],
            "s3_bucket": out["S3Bucket"],
            "instance_type": "ml.m5.xlarge",
            "mlflow": {
                "tracking_server_arn": out["MlflowTrackingServerArn"],
                "experiment_name": "auto-research-churn",
            },
        },
        "model_registry": {"package_group": "auto-research-churn-models"},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--region", required=True)
    ap.add_argument("--out", default=str(
        pathlib.Path(__file__).resolve().parent.parent / "research.yaml"))
    args = ap.parse_args()

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    out = stack_outputs(session.client("cloudformation"), STACK_NAME)
    secret = admin_secret_arn(session.client("redshift-serverless"), NAMESPACE_NAME)
    cfg = build_config(out, secret, args.profile, args.region)

    import yaml
    dest = pathlib.Path(args.out)
    with open(dest, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    print(f"wrote {dest}")

    # Validate with the plugin's own validator so a bad config fails here.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
    from sm_research import config as smconfig  # noqa: PLC0415

    errs = smconfig.validate(smconfig.load(str(dest)))
    if errs:
        sys.exit("config INVALID:\n  " + "\n  ".join(errs))
    print("config VALID (checked with sm_research.config.validate)")
    print(f"\nthreshold f1 >= {THRESHOLD} — above every single-table ceiling "
          f"(max 0.62), below the joined ceiling (0.73)")


if __name__ == "__main__":
    main()
