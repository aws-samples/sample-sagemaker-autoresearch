"""Register a winning model in the SageMaker Model Registry."""
import argparse
import json

import boto3
from botocore.exceptions import ClientError

from sm_research import config, contract


def _ensure_group(sm_client, group: str):
    try:
        sm_client.describe_model_package_group(ModelPackageGroupName=group)
    except ClientError:
        sm_client.create_model_package_group(
            ModelPackageGroupName=group,
            ModelPackageGroupDescription="SageMaker Auto Research candidates")


def register(cfg: dict, sm_client, model: dict) -> dict:
    group = cfg["model_registry"]["package_group"]
    _ensure_group(sm_client, group)
    resp = sm_client.create_model_package(
        ModelPackageGroupName=group,
        ModelApprovalStatus="PendingManualApproval",
        InferenceSpecification={
            "Containers": [{"Image": model["image_uri"],
                            "ModelDataUrl": model["model_data_s3"]}],
            "SupportedContentTypes": ["text/csv"],
            "SupportedResponseMIMETypes": ["text/csv"],
        },
        CustomerMetadataProperties={
            model["metric_name"]: str(model["metric_value"]),
        },
    )
    return {"model_package_arn": resp["ModelPackageArn"], "package_group": group}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-json", required=True)
    args = parser.parse_args()
    try:
        cfg = config.load(args.config)
        model = json.loads(args.model_json)
        session = boto3.Session(profile_name=cfg["aws"]["profile"],
                                region_name=cfg["aws"]["region"])
        sm_client = session.client("sagemaker")
        contract.emit(contract.ok(register(cfg, sm_client, model)))
    except Exception as e:  # noqa: BLE001
        contract.emit(contract.err(str(e)))


if __name__ == "__main__":
    main()
