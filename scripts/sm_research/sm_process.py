"""Launch and poll a SageMaker Processing job for feature engineering.

Use this instead of inline pandas when a slice is too large to fit or transform
in the agent's own process. The script runs inside a SageMaker-managed container
with the slice mounted as an input channel and writes its output back to S3.

Container layout (matching the defaults below):
    /opt/ml/processing/input/     <- the UNLOADed slice (parquet)
    /opt/ml/processing/output/    -> whatever the script writes, synced to S3

Cost is reported the same way as sm_launch (live Pricing API, fallback map, and
an explicit rate_source) so Processing time counts against the loop's budget
instead of being invisible.
"""
import argparse
import json
import time

import boto3

from sm_research import config, contract
from sm_research.sm_launch import DEFAULT_PRICE_MAP, estimate_cost, lookup_hourly_rate

INPUT_DIR = "/opt/ml/processing/input"
OUTPUT_DIR = "/opt/ml/processing/output"


def build_processing_args(cfg: dict, job: dict) -> dict:
    """Build the CreateProcessingJob request.

    job keys: job_name, image_uri, input_s3, output_s3, code_s3 (a .py file in
    S3), instance_type; optional container_arguments, instance_count, volume_gb,
    max_runtime_seconds, env.
    """
    inputs = [{
        "InputName": "slice",
        "S3Input": {
            "S3Uri": job["input_s3"],
            "LocalPath": INPUT_DIR,
            "S3DataType": "S3Prefix",
            "S3InputMode": "File",
            "S3DataDistributionType": "FullyReplicated",
        },
    }]
    # The entry point is staged as its own input so the container can run it
    # directly; Processing has no sourcedir/tarball convention like training.
    container_entrypoint = None
    if job.get("code_s3"):
        inputs.append({
            "InputName": "code",
            "S3Input": {
                "S3Uri": job["code_s3"],
                "LocalPath": "/opt/ml/processing/code",
                "S3DataType": "S3Prefix",
                "S3InputMode": "File",
                "S3DataDistributionType": "FullyReplicated",
            },
        })
        script = job["code_s3"].rstrip("/").rsplit("/", 1)[-1]
        container_entrypoint = ["python3", f"/opt/ml/processing/code/{script}"]

    req = {
        "ProcessingJobName": job["job_name"],
        "RoleArn": cfg["sagemaker"]["role_arn"],
        "AppSpecification": {"ImageUri": job["image_uri"]},
        "ProcessingInputs": inputs,
        "ProcessingOutputConfig": {"Outputs": [{
            "OutputName": "features",
            "S3Output": {
                "S3Uri": job["output_s3"],
                "LocalPath": OUTPUT_DIR,
                "S3UploadMode": "EndOfJob",
            },
        }]},
        "ProcessingResources": {"ClusterConfig": {
            "InstanceType": job["instance_type"],
            "InstanceCount": job.get("instance_count", 1),
            "VolumeSizeInGB": job.get("volume_gb", 30),
        }},
        "StoppingCondition": {
            "MaxRuntimeInSeconds": job.get("max_runtime_seconds", 3600)
        },
    }
    if container_entrypoint:
        req["AppSpecification"]["ContainerEntrypoint"] = container_entrypoint
    if job.get("container_arguments"):
        req["AppSpecification"]["ContainerArguments"] = [
            str(a) for a in job["container_arguments"]
        ]
    if job.get("env"):
        req["Environment"] = {k: str(v) for k, v in job["env"].items()}
    return req


def launch_processing(cfg: dict, sm_client, job: dict):
    return sm_client.create_processing_job(**build_processing_args(cfg, job))


def poll(sm_client, job_name: str, interval: int = 30) -> dict:
    while True:
        desc = sm_client.describe_processing_job(ProcessingJobName=job_name)
        status = desc["ProcessingJobStatus"]
        if status in ("Completed", "Failed", "Stopped"):
            seconds = 0
            start = desc.get("ProcessingStartTime")
            end = desc.get("ProcessingEndTime")
            # Compare against None, not truthiness: an epoch-0 timestamp is falsy.
            if start is not None and end is not None:
                seconds = _epoch(end) - _epoch(start)
            cluster = (desc.get("ProcessingResources") or {}).get("ClusterConfig") or {}
            return {"status": status, "failure_reason": desc.get("FailureReason"),
                    "billable_seconds": seconds,
                    "instance_count": cluster.get("InstanceCount", 1)}
        time.sleep(interval)


def _epoch(v) -> float:
    return v.timestamp() if hasattr(v, "timestamp") else float(v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--job-json", required=True)
    args = parser.parse_args()
    try:
        cfg = config.load(args.config)
        job = json.loads(args.job_json)
        session = boto3.Session(profile_name=cfg["aws"]["profile"],
                                region_name=cfg["aws"]["region"])
        sm_client = session.client("sagemaker")
        launch_processing(cfg, sm_client, job)
        result = poll(sm_client, job["job_name"])

        instance_type = job["instance_type"]
        rate = lookup_hourly_rate(session, instance_type, cfg["aws"]["region"],
                                  usage_family="Processing")
        if rate is not None:
            result["rate_source"] = "pricing_api"
        elif instance_type in DEFAULT_PRICE_MAP:
            rate = DEFAULT_PRICE_MAP[instance_type]
            result["rate_source"] = "fallback_map"
        else:
            result["rate_source"] = "unknown"
        result["hourly_rate_usd"] = rate
        result["cost_usd"] = estimate_cost(
            instance_type, result.get("billable_seconds", 0), DEFAULT_PRICE_MAP,
            instance_count=result.get("instance_count", 1), rate=rate)
        result["output_s3"] = job["output_s3"]
        contract.emit(contract.ok(result))
    except Exception as e:  # noqa: BLE001
        contract.emit(contract.err(str(e)))


if __name__ == "__main__":
    main()
