#!/usr/bin/env python3
"""CDK app provisioning the test bed for the SageMaker Auto Research plugin.

Creates the four things research.yaml needs and this account does not yet have:
  1. An S3 bucket for slices, training output, and model artifacts.
  2. ONE IAM role trusted by BOTH sagemaker.amazonaws.com and redshift.amazonaws.com.
     This dual trust is required, not stylistic: redshift_unload.py passes
     `sagemaker.role_arn` straight into `UNLOAD ... IAM_ROLE '<arn>'`, so the
     SageMaker execution role and the Redshift UNLOAD role must be the same role.
  3. A small (8 RPU) Redshift Serverless namespace + workgroup holding the test
     data, with that role attached so COPY and UNLOAD work.
  4. A Secrets Manager admin secret, used for Redshift Data API auth so the
     loader and the plugin both connect as a superuser.

The existing `mlflowts` MLflow tracking server is reused (not recreated) — the
role is granted access to it instead.

Deploy: cd infra && uv run cdk deploy --profile oregon
"""
import os

import aws_cdk as cdk
from aws_cdk import CfnOutput, RemovalPolicy, Stack
from aws_cdk import aws_iam as iam
from aws_cdk import aws_redshiftserverless as redshiftserverless
from aws_cdk import aws_s3 as s3
from constructs import Construct

NAMESPACE_NAME = "autoresearch"
WORKGROUP_NAME = "autoresearch-wg"
DATABASE_NAME = "autoresearch"
# 8 RPU is the minimum base capacity. The test tables are tiny; a larger
# workgroup would bill the same per-query minimum at a much higher rate.
BASE_CAPACITY_RPU = 8


def discover_subnet_ids(session_region: str, profile: str | None) -> list[str]:
    """Return one default-VPC subnet in each of three AZs.

    Discovered with boto3 rather than ec2.Vpc.from_lookup: without bootstrapped
    lookup context, from_lookup returns dummy values with only 2 fake AZs and
    fails Redshift Serverless's 3-AZ requirement at synth time.
    """
    import boto3

    session = boto3.Session(profile_name=profile, region_name=session_region)
    ec2 = session.client("ec2")
    vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"]
    if not vpcs:
        raise SystemExit(
            f"no default VPC in {session_region}; pass -c subnet_ids=<id>,<id>,<id>"
        )
    subnets = ec2.describe_subnets(
        Filters=[{"Name": "vpc-id", "Values": [vpcs[0]["VpcId"]]}]
    )["Subnets"]
    by_az: dict[str, str] = {}
    for s in sorted(subnets, key=lambda s: s["SubnetId"]):
        by_az.setdefault(s["AvailabilityZone"], s["SubnetId"])
    ids = [by_az[az] for az in sorted(by_az)][:3]
    if len(ids) < 3:
        raise SystemExit(
            f"Redshift Serverless needs subnets in 3 AZs; {session_region} default "
            f"VPC offered {len(ids)}. Pass -c subnet_ids=<id>,<id>,<id>"
        )
    return ids


def discover_mlflow(session_region: str, profile: str | None) -> tuple[str | None, str | None]:
    """Find an existing MLflow tracking server and its artifact bucket.

    Reusing one avoids creating a second billable server (~$0.50-1.00/hour).
    Returns (None, None) when the account has none, in which case the stack
    simply omits the MLflow grants.
    """
    import boto3
    from urllib.parse import urlparse

    session = boto3.Session(profile_name=profile, region_name=session_region)
    sm = session.client("sagemaker")
    servers = sm.list_mlflow_tracking_servers().get("TrackingServerSummaries", [])
    if not servers:
        return None, None
    arn = servers[0]["TrackingServerArn"]
    name = servers[0]["TrackingServerName"]
    detail = sm.describe_mlflow_tracking_server(TrackingServerName=name)
    bucket = urlparse(detail.get("ArtifactStoreUri", "")).netloc or None
    return arn, bucket


class AutoResearchTestBedStack(Stack):
    def __init__(self, scope: Construct, cid: str, *, subnet_ids: list[str],
                 mlflow_arn: str | None, mlflow_bucket: str | None, **kwargs) -> None:
        super().__init__(scope, cid, **kwargs)

        # --- 1. Artifact bucket ------------------------------------------
        bucket = s3.Bucket(
            self,
            "ArtifactBucket",
            bucket_name=f"sm-auto-research-{self.account}-{self.region}",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            versioned=False,
            # RETAIN so that tearing down the stack can never destroy
            # experiment output or registered model artifacts.
            removal_policy=RemovalPolicy.RETAIN,
        )

        # --- 2. Dual-trust SageMaker + Redshift role ---------------------
        role = iam.Role(
            self,
            "AutoResearchRole",
            role_name="SageMakerAutoResearchRole",
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal("sagemaker.amazonaws.com"),
                iam.ServicePrincipal("redshift.amazonaws.com"),
            ),
            description=(
                "SageMaker execution role for auto-research training/processing jobs, "
                "and the IAM role Redshift assumes for COPY/UNLOAD."
            ),
        )

        bucket.grant_read_write(role)

        # MLflow grants apply only when an existing tracking server was found or
        # supplied; otherwise the stack deploys without them and the plugin will
        # offer to create a server (a billable action needing confirmation).
        if mlflow_bucket:
            role.add_to_policy(
                iam.PolicyStatement(
                    actions=["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
                    resources=[f"arn:aws:s3:::{mlflow_bucket}/mlflow/*"],
                )
            )
            role.add_to_policy(
                iam.PolicyStatement(
                    actions=["s3:ListBucket", "s3:GetBucketLocation"],
                    resources=[f"arn:aws:s3:::{mlflow_bucket}"],
                    conditions={"StringLike": {"s3:prefix": ["mlflow/*", "mlflow"]}},
                )
            )
        if mlflow_arn:
            role.add_to_policy(
                iam.PolicyStatement(
                    actions=["sagemaker-mlflow:*"],
                    resources=[mlflow_arn],
                )
            )

        # Pull training images; emit logs and metrics.
        role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "ecr:GetAuthorizationToken",
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage",
                ],
                resources=["*"],  # ECR auth token is not resource-scopable
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "logs:DescribeLogStreams",
                ],
                resources=[
                    f"arn:aws:logs:{self.region}:{self.account}:log-group:/aws/sagemaker/*"
                ],
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["cloudwatch:PutMetricData"],
                resources=["*"],
                conditions={
                    "StringEquals": {"cloudwatch:namespace": ["/aws/sagemaker", "AWS/SageMaker"]}
                },
            )
        )

        # --- 3. Redshift Serverless namespace + workgroup ----------------
        namespace = redshiftserverless.CfnNamespace(
            self,
            "Namespace",
            namespace_name=NAMESPACE_NAME,
            db_name=DATABASE_NAME,
            admin_username="admin",
            # Let Redshift generate and rotate the admin password into a secret;
            # the Data API then authenticates with SecretArn as a superuser.
            manage_admin_password=True,
            iam_roles=[role.role_arn],
            default_iam_role_arn=role.role_arn,
        )

        workgroup = redshiftserverless.CfnWorkgroup(
            self,
            "Workgroup",
            workgroup_name=WORKGROUP_NAME,
            namespace_name=NAMESPACE_NAME,
            base_capacity=BASE_CAPACITY_RPU,
            # Private: the Data API reaches Redshift over the AWS API, not the VPC,
            # so nothing here needs to be internet-reachable.
            publicly_accessible=False,
            subnet_ids=subnet_ids,
        )
        workgroup.add_resource_dependency(namespace)

        # --- Outputs consumed when writing research.yaml -----------------
        # The admin password secret ARN is NOT output here: this CDK version's
        # resource spec has no AdminPasswordSecretArn attribute. write_research_yaml.py
        # discovers it with `redshift-serverless get-namespace` after deploy.
        CfnOutput(self, "S3Bucket", value=bucket.bucket_name)
        CfnOutput(self, "RoleArn", value=role.role_arn)
        CfnOutput(self, "WorkgroupName", value=WORKGROUP_NAME)
        CfnOutput(self, "DatabaseName", value=DATABASE_NAME)
        # Empty string when the account has no tracking server; write_research_yaml.py
        # leaves sagemaker.mlflow.tracking_server_arn blank, which makes the plugin
        # ask before creating one.
        CfnOutput(self, "MlflowTrackingServerArn", value=mlflow_arn or "")


app = cdk.App()

_region = os.environ["CDK_DEFAULT_REGION"]
_profile = os.environ.get("AWS_PROFILE")

# Everything account-specific is context-overridable, and otherwise discovered at
# synth time, so the stack works in any account/region without editing this file:
#   -c subnet_ids=subnet-a,subnet-b,subnet-c
#   -c mlflow_arn=arn:aws:sagemaker:...:mlflow-tracking-server/name
#   -c mlflow_bucket=amzn-s3-demo-artifact-bucket
#   -c mlflow_arn=none    (skip MLflow reuse entirely)
_subnets_ctx = app.node.try_get_context("subnet_ids")
_subnet_ids = ([s.strip() for s in _subnets_ctx.split(",")] if _subnets_ctx
               else discover_subnet_ids(_region, _profile))

_mlflow_ctx = app.node.try_get_context("mlflow_arn")
if _mlflow_ctx == "none":
    _mlflow_arn, _mlflow_bucket = None, None
elif _mlflow_ctx:
    _mlflow_arn = _mlflow_ctx
    _mlflow_bucket = app.node.try_get_context("mlflow_bucket")
else:
    _mlflow_arn, _mlflow_bucket = discover_mlflow(_region, _profile)
    _mlflow_bucket = app.node.try_get_context("mlflow_bucket") or _mlflow_bucket

AutoResearchTestBedStack(
    app,
    "AutoResearchTestBed",
    subnet_ids=_subnet_ids,
    mlflow_arn=_mlflow_arn,
    mlflow_bucket=_mlflow_bucket,
    env=cdk.Environment(
        account=os.environ["CDK_DEFAULT_ACCOUNT"],
        region=_region,
    ),
)
app.synth()
