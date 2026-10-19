"""Package, launch, diagnose, and MLflow-log one training attempt in ONE call.

Collapses the experiment tail that otherwise takes 8-15 tool calls --
package_code -> sm_launch (retry on hyperparameter / metric-definition
rejects) -> read the metric or traceback out of CloudWatch -> describe the
model artifact -> hand-written MLflow logging -> assemble the return record --
into a single command whose compact result IS the experiment record.

Retries: rerun with the SAME --experiment. Attempts are named
`<prefix>-a1`, `<prefix>-a2`, ... and cost_usd / billable_seconds sum every
attempt, so the record always prices all billable jobs of the experiment.

Run with MLflow available:
    cd ${CLAUDE_PLUGIN_ROOT} && uv run --with mlflow --with sagemaker-mlflow \
      python -m sm_research.run_training --config "$CFG" ...
"""
import argparse
import json
import os
import re
import sys
import time

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from sm_research import config, contract, package_code, sm_launch

LOG_GROUP = "/aws/sagemaker/TrainingJobs"

# `name=number;` pairs printed by the training script. The lookbehind stops
# `val_f1=0.6;` from also matching as `f1=0.6;`.
_KV = re.compile(
    r"(?<![\w.])([A-Za-z_]\w*)=([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?);")
# Container chatter that buries the real error in a raw log tail.
_NOISE = ("pkg_resources", "UserWarning", "DeprecationWarning", "DEPRECATION:",
          "warnings.warn", "Building wheel", "Created wheel", "Stored in directory",
          "Successfully built", "Successfully installed", "Processing /opt/ml/code")
_EXC_LINE = re.compile(r"^[\w.]+(?:Error|Exception|Exit|Interrupt)\b")
_MAX_PREFIX = 59  # 63-char job-name limit minus "-a99"
_MAX_ATTEMPTS = 99
# Parallel experiments share one account's SageMaker/CloudWatch API quotas;
# the default legacy mode gives up after 4 quick retries ("ThrottlingException
# ... Rate exceeded") before any job launches.
_RETRY = Config(retries={"mode": "adaptive", "max_attempts": 12})


def job_prefix(run_id: str, experiment: str) -> str:
    raw = re.sub(r"[^A-Za-z0-9-]+", "-", f"{run_id}-{experiment}")
    return raw[:_MAX_PREFIX].strip("-")


def _job_exists(sm, name: str) -> bool:
    try:
        sm.describe_training_job(TrainingJobName=name)
        return True
    except ClientError as e:
        if "not found" in str(e).lower() or \
                e.response.get("Error", {}).get("Code") == "ResourceNotFound":
            return False
        raise


def existing_attempts(sm, prefix: str) -> list[str]:
    """Names of earlier attempts `<prefix>-a<N>`, oldest first.

    Probes `-a1`, `-a2`, ... with DescribeTrainingJob instead of paginating
    ListTrainingJobs: List has a far lower account-wide rate limit and scans
    every job in the account, so concurrent experiments throttled it before
    launching anything. Attempts are contiguous, so the first gap ends the scan.
    """
    found = []
    for n in range(1, _MAX_ATTEMPTS + 1):
        name = f"{prefix}-a{n}"
        if not _job_exists(sm, name):
            break
        found.append(name)
    return found


def stringify_hps(hps: dict) -> dict:
    # CreateTrainingJob only accepts string values ("Invalid type for parameter
    # HyperParameters.max-depth ... valid types: <class 'str'>").
    return {str(k): v if isinstance(v, str) else json.dumps(v) for k, v in hps.items()}


def entry_script(scripts: list[str], entry_point: str | None) -> str | None:
    """The packaged script SageMaker runs (`--entry-point` names its file)."""
    for path in scripts:
        if os.path.basename(path) == entry_point:
            return path
    return scripts[0] if scripts else None


def with_target(hps: dict, cfg: dict, script: str | None) -> dict:
    """Pass `data.target` as the `target` hyperparameter when the script takes
    `--target` and the caller did not set it. The templates default it to the
    literal "target", so forgetting it failed jobs with `KeyError: 'target'`."""
    target = (cfg.get("data") or {}).get("target")
    if not target or "target" in hps or not script:
        return hps
    try:
        with open(script, encoding="utf-8") as f:
            src = f.read()
        accepts = '"--target"' in src or "'--target'" in src
    except OSError:
        return hps
    return {**hps, "target": target} if accepts else hps


def log_lines(logs, job_name: str, tries: int = 3, wait: int = 5) -> list[str]:
    """All log lines of the job, read stream by stream (fast, unlike a
    group-wide filter, which can run for minutes on a busy log group)."""
    lines: list[str] = []
    for i in range(tries):
        streams = logs.describe_log_streams(
            logGroupName=LOG_GROUP, logStreamNamePrefix=f"{job_name}/").get("logStreams", [])
        for s in streams:
            token = None
            while True:
                kw = {"logGroupName": LOG_GROUP, "logStreamName": s["logStreamName"],
                      "startFromHead": True}
                if token:
                    kw["nextToken"] = token
                resp = logs.get_log_events(**kw)
                for e in resp.get("events", []):
                    lines.extend(e["message"].splitlines())
                nxt = resp.get("nextForwardToken")
                if not resp.get("events") or nxt == token:
                    break
                token = nxt
        if lines or i == tries - 1:
            return lines
        time.sleep(wait)  # logs can lag the terminal status by a few seconds
    return lines


def parse_metrics(lines: list[str]) -> dict:
    found = {}
    for line in lines:
        for k, v in _KV.findall(line):
            try:
                found[k] = float(v)  # last value wins
            except ValueError:
                pass
    return found


def pick_metric(found: dict, name: str, key: str | None):
    for k in ([key] if key else [name, f"test_{name}", f"{name}_test"]):
        if k in found:
            return k, found[k]
    return None, None


def failure_tail(lines: list[str], max_lines: int = 40, max_chars: int = 3000) -> str:
    """The user script's traceback, not 20 KB of container chatter."""
    keep = [l for l in lines if l.strip() and not any(n in l for n in _NOISE)]
    blocks = []
    for i, line in enumerate(keep):
        if line.lstrip().startswith("Traceback (most recent call last)"):
            block = [line]
            for nxt in keep[i + 1:i + 1 + max_lines]:
                block.append(nxt)
                if _EXC_LINE.match(nxt.strip()):
                    break
            blocks.append(block)
    user = [b for b in blocks if any("/opt/ml/code" in x for x in b)]
    chosen = (user or blocks or [keep[-max_lines:]])[-1]
    return "\n".join(chosen)[-max_chars:]


def settle(sm, name: str) -> dict:
    """Terminal status + billable time (via sm_launch.poll) plus artifact info."""
    res = sm_launch.poll(sm, name)
    desc = sm.describe_training_job(TrainingJobName=name)
    res["job_name"] = name
    res["instance_type"] = (desc.get("ResourceConfig") or {}).get("InstanceType")
    res["image_uri"] = (desc.get("AlgorithmSpecification") or {}).get("TrainingImage")
    res["model_data_s3"] = (desc.get("ModelArtifacts") or {}).get("S3ModelArtifacts")
    return res


def price(session, region: str, instance_type: str, cache: dict):
    if instance_type not in cache:
        rate = sm_launch.lookup_hourly_rate(session, instance_type, region)
        if rate is not None:
            cache[instance_type] = (rate, "pricing_api")
        elif instance_type in sm_launch.DEFAULT_PRICE_MAP:
            cache[instance_type] = (sm_launch.DEFAULT_PRICE_MAP[instance_type], "fallback_map")
        else:
            cache[instance_type] = (None, "unknown")
    return cache[instance_type]


def log_mlflow(cfg: dict, run_name: str, params: dict, metrics: dict,
               tags: dict, artifacts: list[str]) -> str:
    # Imported here, AFTER main() exported AWS_PROFILE/AWS_REGION: SageMaker's
    # MLflow auth plugin builds its own boto3 session, so a profile set only on
    # a boto3.Session falls back to the default profile and gets a 403.
    import mlflow  # noqa: PLC0415

    ml = cfg["sagemaker"]["mlflow"]
    mlflow.set_tracking_uri(ml["tracking_server_arn"])
    mlflow.set_experiment(ml.get("experiment_name", "auto-research"))
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
        if metrics:
            mlflow.log_metrics(metrics)
        mlflow.set_tags(tags)
        for path in artifacts:
            try:
                mlflow.log_artifact(path)
            except Exception as e:  # noqa: BLE001 — artifacts are best-effort
                print(f"warning: could not log artifact {path}: {e}", file=sys.stderr)
        return run.info.run_id


def run(cfg: dict, session, args) -> dict:
    region = cfg["aws"]["region"]
    sm = session.client("sagemaker", config=_RETRY)
    logs = session.client("logs", config=_RETRY)
    prefix = args.job_prefix or job_prefix(args.run_id, args.experiment)
    prior = existing_attempts(sm, prefix)

    if args.reuse_last:
        if not prior:
            raise ValueError(f"--reuse-last: no earlier attempt named {prefix}-a<N>")
        name, prior = prior[-1], prior[:-1]
        hps = {}
    else:
        name = f"{prefix}-a{len(prior) + 1}"
        if not re.fullmatch(r"[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}", name):
            raise ValueError(f"invalid training job name {name!r}; pass a shorter --job-prefix")
        pkg = package_code.package(cfg, session.client("s3"), args.script,
                                   args.entry_point, args.run_id, args.experiment)
        hps = {**with_target(json.loads(args.hyperparameters_json), cfg,
                             entry_script(args.script, args.entry_point)),
               **pkg["hyperparameters"]}
        job = {"job_name": name, "image_uri": args.image_uri, "input_s3": args.input_s3,
               "output_s3": args.output_s3
               or f"s3://{cfg['sagemaker']['s3_bucket']}/sm-research/{args.run_id}/out/",
               "hyperparameters": stringify_hps(hps),
               "instance_type": args.instance_type or cfg["sagemaker"]["instance_type"]}
        if args.metric_regex:
            job["metric_regex"] = args.metric_regex
        try:
            sm_launch.launch_training(cfg, sm, job)
        except ClientError as e:
            # Built-in-algorithm-backed images (e.g. sagemaker-xgboost) reject
            # MetricDefinitions even in script mode; the log parse covers it.
            if job.pop("metric_regex", None) and "metric definitions" in str(e):
                sm_launch.launch_training(cfg, sm, job)
            else:
                raise

    final = settle(sm, name)
    rates: dict = {}
    attempts = []
    for res in [settle(sm, n) for n in prior] + [final]:
        rate, src = price(session, region, res["instance_type"], rates)
        cost = sm_launch.estimate_cost(res["instance_type"], res["billable_seconds"], {},
                                       instance_count=res["instance_count"], rate=rate)
        attempts.append({"job_name": res["job_name"], "status": res["status"],
                         "instance_type": res["instance_type"],
                         "billable_seconds": res["billable_seconds"],
                         "cost_usd": cost, "rate_source": src})

    lines = log_lines(logs, name)
    found = parse_metrics(lines)
    metric_name = cfg["problem"]["success_metric"]
    key, metric = pick_metric(found, metric_name, args.metric_key)
    if metric is None and final["status"] == "Completed":
        metric = final.get("final_metric")
    failure = final.get("failure_reason")
    if final["status"] != "Completed":
        metric = None
    elif metric is None:
        failure = (f"job completed but printed no '{metric_name}=<value>;' line "
                   f"(log keys: {sorted(found)})")

    hourly, src = rates[final["instance_type"]]
    if any(a["rate_source"] == "unknown" for a in attempts):
        src = "unknown"
    record = {
        "metric": metric,
        "status": final["status"],
        "hypothesis": args.hypothesis,
        "mlflow_run_id": None,
        "model_data_s3": final["model_data_s3"],
        "image_uri": final["image_uri"],
        "cost_usd": sum(a["cost_usd"] for a in attempts) + args.extra_cost_usd,
        "billable_seconds": sum(a["billable_seconds"] for a in attempts)
        + args.extra_billable_seconds,
        "failure_reason": failure[:600] if failure else None,
        "rate_source": src,
        "hourly_rate_usd": hourly,
        "instance_type": final["instance_type"],
    }
    extra = {"job_name": name, "metric_key": key, "job_metric": final.get("final_metric"),
             "log_metrics": found, "attempts": attempts}
    if final["status"] != "Completed":
        extra["log_tail"] = failure_tail(lines)

    params = {f"hp.{k}": v for k, v in hps.items() if not k.startswith("sagemaker_")}
    params.update({"job_name": name, "image_uri": record["image_uri"],
                   "instance_type": record["instance_type"], "input_s3": args.input_s3,
                   "hypothesis": args.hypothesis, "attempts": len(attempts)})
    metrics = {k: v for k, v in found.items() if k != key}
    if metric is not None:
        metrics[metric_name] = metric
    metrics.update({"cost_usd": record["cost_usd"],
                    "billable_seconds": float(record["billable_seconds"])})
    try:
        record["mlflow_run_id"] = log_mlflow(
            cfg, name, params, metrics,
            {"run_id": args.run_id, "experiment": args.experiment,
             "status": record["status"], "sagemaker_training_job": name},
            [] if args.reuse_last else list(args.script))
    except Exception as e:  # noqa: BLE001 — keep the job result; caller can --reuse-last
        extra["mlflow_error"] = str(e)[-500:]
    return {"record": record, **extra}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--experiment", required=True,
                        help="e.g. r1-e2; reuse it on retries so attempts are summed")
    parser.add_argument("--script", action="append", default=[],
                        help="training script; repeat to package helpers alongside it")
    parser.add_argument("--entry-point")
    parser.add_argument("--image-uri")
    parser.add_argument("--input-s3", help="slice or features prefix (training channel)")
    parser.add_argument("--hyperparameters-json", default="{}")
    parser.add_argument("--hypothesis", default="")
    parser.add_argument("--instance-type", help="default: sagemaker.instance_type")
    parser.add_argument("--output-s3")
    parser.add_argument("--metric-regex", help="optional SageMaker MetricDefinitions regex")
    parser.add_argument("--metric-key", help="log key holding the test metric if not "
                        "<success_metric>")
    parser.add_argument("--job-prefix")
    parser.add_argument("--extra-cost-usd", type=float, default=0.0,
                        help="e.g. a Processing job's cost_usd")
    parser.add_argument("--extra-billable-seconds", type=float, default=0.0)
    parser.add_argument("--reuse-last", action="store_true",
                        help="do not launch; re-read and re-log the latest attempt")
    args = parser.parse_args()
    try:
        cfg = config.load(args.config)
        profile, region = cfg["aws"]["profile"], cfg["aws"]["region"]
        # Export before anything imports mlflow / sagemaker_mlflow (see log_mlflow).
        os.environ["AWS_PROFILE"] = profile
        os.environ["AWS_REGION"] = os.environ["AWS_DEFAULT_REGION"] = region
        if not args.reuse_last:
            missing = [f for f in ("script", "entry_point", "image_uri", "input_s3")
                       if not getattr(args, f)]
            if missing:
                raise ValueError(f"missing required option(s): {missing}")
        session = boto3.Session(profile_name=profile, region_name=region)
        contract.emit(contract.ok(run(cfg, session, args)))
    except Exception as e:  # noqa: BLE001
        contract.emit(contract.err(str(e)))


if __name__ == "__main__":
    main()
