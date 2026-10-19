"""Find an existing SageMaker MLflow tracking server, or create one after confirmation."""
import argparse

import boto3

from sm_research import config, contract


def find_server(cfg: dict, client) -> str | None:
    configured = cfg["sagemaker"]["mlflow"].get("tracking_server_arn")
    if configured:
        return configured
    resp = client.list_mlflow_tracking_servers()
    summaries = resp.get("TrackingServerSummaries", [])
    if summaries:
        return summaries[0]["TrackingServerArn"]
    return None


def ensure(cfg: dict, client, confirm_create: bool) -> dict:
    arn = find_server(cfg, client)
    if arn:
        return {"tracking_server_arn": arn, "created": False, "needs_confirmation": False}
    if not confirm_create:
        return {"tracking_server_arn": None, "created": False, "needs_confirmation": True}
    mlflow_cfg = cfg["sagemaker"]["mlflow"]
    name = mlflow_cfg.get("experiment_name", "auto-research") + "-server"
    created = client.create_mlflow_tracking_server(
        TrackingServerName=name,
        ArtifactStoreUri=f"s3://{cfg['sagemaker']['s3_bucket']}/mlflow",
        RoleArn=cfg["sagemaker"]["role_arn"],
    )
    return {"tracking_server_arn": created["TrackingServerArn"], "created": True,
            "needs_confirmation": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--confirm-create", action="store_true")
    args = parser.parse_args()
    try:
        cfg = config.load(args.config)
        session = boto3.Session(profile_name=cfg["aws"]["profile"],
                                region_name=cfg["aws"]["region"])
        client = session.client("sagemaker")
        contract.emit(contract.ok(ensure(cfg, client, args.confirm_create)))
    except Exception as e:  # noqa: BLE001
        contract.emit(contract.err(str(e)))


if __name__ == "__main__":
    main()
