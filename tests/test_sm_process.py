from unittest.mock import MagicMock

from sm_research import sm_process


def _cfg():
    return {"aws": {"profile": "p", "region": "us-west-2"},
            "sagemaker": {"role_arn": "arn:aws:iam::1:role/r"}}


def _job(**over):
    job = {
        "job_name": "fe-1",
        "image_uri": "img",
        "input_s3": "s3://b/slice/",
        "output_s3": "s3://b/features/",
        "code_s3": "s3://b/code/featurize.py",
        "instance_type": "ml.m5.xlarge",
    }
    job.update(over)
    return job


def test_build_processing_args_mounts_slice_and_output():
    req = sm_process.build_processing_args(_cfg(), _job())
    assert req["ProcessingJobName"] == "fe-1"
    slice_in = next(i for i in req["ProcessingInputs"] if i["InputName"] == "slice")
    assert slice_in["S3Input"]["S3Uri"] == "s3://b/slice/"
    assert slice_in["S3Input"]["LocalPath"] == sm_process.INPUT_DIR
    out = req["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]
    assert out["S3Uri"] == "s3://b/features/"
    assert out["LocalPath"] == sm_process.OUTPUT_DIR


def test_build_processing_args_sets_entrypoint_from_code_s3():
    """Processing has no sourcedir convention; the script is staged as an input."""
    req = sm_process.build_processing_args(_cfg(), _job())
    assert req["AppSpecification"]["ContainerEntrypoint"] == [
        "python3", "/opt/ml/processing/code/featurize.py"]
    assert any(i["InputName"] == "code" for i in req["ProcessingInputs"])


def test_build_processing_args_omits_entrypoint_without_code():
    req = sm_process.build_processing_args(_cfg(), _job(code_s3=None))
    assert "ContainerEntrypoint" not in req["AppSpecification"]
    assert all(i["InputName"] != "code" for i in req["ProcessingInputs"])


def test_build_processing_args_stringifies_container_arguments():
    # CreateProcessingJob rejects non-string arguments.
    req = sm_process.build_processing_args(
        _cfg(), _job(container_arguments=["--rows", 500]))
    assert req["AppSpecification"]["ContainerArguments"] == ["--rows", "500"]


def test_build_processing_args_honors_resource_overrides():
    req = sm_process.build_processing_args(
        _cfg(), _job(instance_count=3, volume_gb=100, max_runtime_seconds=60))
    cluster = req["ProcessingResources"]["ClusterConfig"]
    assert cluster["InstanceCount"] == 3
    assert cluster["VolumeSizeInGB"] == 100
    assert req["StoppingCondition"]["MaxRuntimeInSeconds"] == 60


def test_poll_returns_status_and_billable_seconds():
    client = MagicMock()
    client.describe_processing_job.return_value = {
        "ProcessingJobStatus": "Completed",
        "ProcessingStartTime": 0, "ProcessingEndTime": 240,
        "ProcessingResources": {"ClusterConfig": {"InstanceCount": 2}},
    }
    out = sm_process.poll(client, "fe-1")
    assert out["status"] == "Completed"
    assert out["billable_seconds"] == 240
    assert out["instance_count"] == 2


def test_poll_returns_failure_reason():
    client = MagicMock()
    client.describe_processing_job.return_value = {
        "ProcessingJobStatus": "Failed", "FailureReason": "script error",
    }
    out = sm_process.poll(client, "fe-1")
    assert out["status"] == "Failed"
    assert out["failure_reason"] == "script error"


def test_launch_processing_passes_request_through():
    client = MagicMock()
    sm_process.launch_processing(_cfg(), client, _job())
    kwargs = client.create_processing_job.call_args.kwargs
    assert kwargs["RoleArn"] == "arn:aws:iam::1:role/r"
