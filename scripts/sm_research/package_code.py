"""Package a training script into the sourcedir.tar.gz that SageMaker expects.

Framework containers (scikit-learn, xgboost, pytorch, ...) do not run a loose
script from S3: they download a gzipped tarball, extract it into /opt/ml/code,
and execute the entry point named by the `sagemaker_program` hyperparameter.
Getting any part of that wrong fails the job with the real cause buried in
CloudWatch, so this module owns the whole step and returns the exact
hyperparameters `sm_launch` needs.

Returns `hyperparameters` ready to merge into a job spec:
    {"sagemaker_program": "<entry_point>",
     "sagemaker_submit_directory": "s3://.../sourcedir.tar.gz"}
"""
import argparse
import json
import os
import pathlib
import tarfile
import tempfile

import boto3

from sm_research import config, contract


def build_tarball(script_paths: list[str], dest: str) -> list[str]:
    """Tar the given scripts at the archive root (not nested in a directory).

    SageMaker extracts the archive directly into /opt/ml/code and runs
    `python <sagemaker_program>` from there, so entries must be at the top level.
    """
    names = []
    with tarfile.open(dest, "w:gz") as tar:
        for path in script_paths:
            p = pathlib.Path(path)
            if not p.is_file():
                raise FileNotFoundError(f"training script not found: {path}")
            # arcname = bare filename so it lands at /opt/ml/code/<name>
            tar.add(str(p), arcname=p.name)
            names.append(p.name)
    return names


def package(cfg: dict, s3, scripts: list[str], entry_point: str,
            run_id: str = "default", experiment: str = "default") -> dict:
    names = [pathlib.Path(s).name for s in scripts]
    if entry_point not in names:
        raise ValueError(
            f"entry_point {entry_point!r} is not among the packaged scripts {names}; "
            "SageMaker would fail with \"can't open file '/opt/ml/code/"
            f"{entry_point}'\"")

    bucket = cfg["sagemaker"]["s3_bucket"]
    key = f"sm-research/{run_id}/code/{experiment}/sourcedir.tar.gz"

    with tempfile.TemporaryDirectory() as tmp:
        local = os.path.join(tmp, "sourcedir.tar.gz")
        packaged = build_tarball(scripts, local)
        s3.upload_file(local, bucket, key)
        size = os.path.getsize(local)

    uri = f"s3://{bucket}/{key}"
    return {
        "sourcedir_s3": uri,
        "entry_point": entry_point,
        "packaged_scripts": packaged,
        "size_bytes": size,
        # Merge these straight into the sm_launch --job-json hyperparameters.
        "hyperparameters": {
            "sagemaker_program": entry_point,
            "sagemaker_submit_directory": uri,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--script", action="append", required=True,
                        help="path to a training script; repeat to include several")
    parser.add_argument("--entry-point", required=True,
                        help="filename SageMaker runs, e.g. sklearn_train.py")
    parser.add_argument("--run-id", default="default")
    parser.add_argument("--experiment", default="default",
                        help="keeps concurrent experiments from overwriting each other")
    args = parser.parse_args()
    try:
        cfg = config.load(args.config)
        session = boto3.Session(profile_name=cfg["aws"]["profile"],
                                region_name=cfg["aws"]["region"])
        s3 = session.client("s3")
        contract.emit(contract.ok(package(
            cfg, s3, args.script, args.entry_point, args.run_id, args.experiment)))
    except Exception as e:  # noqa: BLE001
        contract.emit(contract.err(str(e)))


if __name__ == "__main__":
    main()
