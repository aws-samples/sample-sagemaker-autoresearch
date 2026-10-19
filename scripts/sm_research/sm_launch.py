"""Launch and poll SageMaker training jobs; estimate cost."""
import argparse
import json
import time

import boto3

from sm_research import config, contract

# Coarse on-demand $/hour fallbacks used only when the Pricing API is
# unreachable (no pricing:GetProducts permission, or an untabulated type).
# These are us-east-1/us-west-2 figures and WILL understate other regions.
DEFAULT_PRICE_MAP = {
    "ml.m5.xlarge": 0.23,
    "ml.m5.2xlarge": 0.46,
    "ml.c5.xlarge": 0.204,
}

# The Pricing API itself is only served from a few regions.
_PRICING_API_REGION = "us-east-1"


def lookup_hourly_rate(session, instance_type: str, region: str,
                       usage_family: str = "Train") -> float | None:
    """Look up the real on-demand $/hour for an ML instance, or None.

    Cost was previously estimated from a three-entry hardcoded map, so any
    instance type or region outside it silently priced at $0.00 — making the
    budget cap unenforceable exactly when it mattered.
    """
    try:
        client = session.client("pricing", region_name=_PRICING_API_REGION)
        resp = client.get_products(
            ServiceCode="AmazonSageMaker",
            Filters=[
                {"Type": "TERM_MATCH", "Field": "regionCode", "Value": region},
                {"Type": "TERM_MATCH", "Field": "instanceName", "Value": instance_type},
            ],
            MaxResults=100,
        )
    except Exception:  # noqa: BLE001 — pricing is best-effort; fall back below
        return None

    for raw in resp.get("PriceList", []):
        product = json.loads(raw)
        usagetype = product["product"]["attributes"].get("usagetype", "")
        # e.g. "USW2-Train:ml.m5.xlarge" — pick the training (or Processing) family
        # so we do not price a training job off a notebook or endpoint rate.
        if f"{usage_family}:{instance_type}" not in usagetype:
            continue
        for term in product["terms"].get("OnDemand", {}).values():
            for dim in term["priceDimensions"].values():
                usd = dim["pricePerUnit"].get("USD")
                if usd is not None:
                    return float(usd)
    return None


def launch_training(cfg: dict, sm_client, job: dict):
    algo_spec = {
        "TrainingImage": job["image_uri"],
        "TrainingInputMode": "File",
    }
    # SageMaker rejects MetricDefinitions for its own built-in algorithm images
    # ("You can't override the metric definitions for Amazon SageMaker
    # algorithms"), so only send them when a regex is supplied. Framework and
    # BYO containers need them to surface the objective metric.
    if job.get("metric_regex"):
        algo_spec["MetricDefinitions"] = [
            {"Name": "objective", "Regex": job["metric_regex"]}
        ]
    return sm_client.create_training_job(
        TrainingJobName=job["job_name"],
        RoleArn=cfg["sagemaker"]["role_arn"],
        AlgorithmSpecification=algo_spec,
        HyperParameters=job.get("hyperparameters", {}),
        InputDataConfig=[{
            "ChannelName": "training",
            "DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix", "S3Uri": job["input_s3"],
                "S3DataDistributionType": "FullyReplicated"}},
        }],
        OutputDataConfig={"S3OutputPath": job["output_s3"]},
        ResourceConfig={"InstanceType": job["instance_type"],
                        "InstanceCount": 1, "VolumeSizeInGB": 30},
        StoppingCondition={"MaxRuntimeInSeconds": 3600},
    )


def poll(sm_client, job_name: str, interval: int = 30) -> dict:
    while True:
        desc = sm_client.describe_training_job(TrainingJobName=job_name)
        status = desc["TrainingJobStatus"]
        if status in ("Completed", "Failed", "Stopped"):
            metrics = desc.get("FinalMetricDataList", [])
            final_metric = metrics[0]["Value"] if metrics else None
            # Prefer SageMaker's own billable figure. Otherwise fall back to
            # Training{Start,End}Time, which EXCLUDES instance provisioning and
            # so undercounts what is actually billed.
            seconds = desc.get("BillableTimeInSeconds")
            billable_exact = seconds is not None
            start, end = desc.get("TrainingStartTime"), desc.get("TrainingEndTime")
            # Compare against None, not truthiness: an epoch-0 timestamp is
            # falsy and would silently zero out the fallback.
            if not billable_exact and start is not None and end is not None:
                seconds = _epoch(end) - _epoch(start)
            resources = desc.get("ResourceConfig") or {}
            return {"status": status, "final_metric": final_metric,
                    "failure_reason": desc.get("FailureReason"),
                    "billable_seconds": seconds or 0,
                    "billable_exact": billable_exact,
                    "instance_count": resources.get("InstanceCount", 1)}
        time.sleep(interval)


def _epoch(v) -> float:
    return v.timestamp() if hasattr(v, "timestamp") else float(v)


def estimate_cost(instance_type: str, seconds: float, price_map: dict,
                  instance_count: int = 1, rate: float | None = None) -> float:
    """Estimate job cost. rate (if given) overrides the price map.

    Returns 0.0 only when the rate is genuinely unknown; callers should check
    rate_source to tell "free" apart from "unpriced" — a silent 0.0 would make
    the budget cap unenforceable.
    """
    hourly = rate if rate is not None else price_map.get(instance_type, 0.0)
    return hourly * (seconds / 3600.0) * max(1, instance_count)


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
        launch_training(cfg, sm_client, job)
        result = poll(sm_client, job["job_name"])

        instance_type = job["instance_type"]
        price_map = job.get("price_map", DEFAULT_PRICE_MAP)
        # Live pricing first; hardcoded map only as a fallback.
        rate = lookup_hourly_rate(session, instance_type, cfg["aws"]["region"])
        if rate is not None:
            result["rate_source"] = "pricing_api"
        elif instance_type in price_map:
            rate = price_map[instance_type]
            result["rate_source"] = "fallback_map"
        else:
            result["rate_source"] = "unknown"
        result["hourly_rate_usd"] = rate
        result["cost_usd"] = estimate_cost(
            instance_type, result.get("billable_seconds", 0), price_map,
            instance_count=result.get("instance_count", 1), rate=rate)
        contract.emit(contract.ok(result))
    except Exception as e:  # noqa: BLE001
        contract.emit(contract.err(str(e)))


if __name__ == "__main__":
    main()
