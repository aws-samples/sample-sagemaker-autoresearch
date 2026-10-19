from unittest.mock import MagicMock

from sm_research import sm_launch


def _cfg():
    return {"aws": {"region": "us-east-1"},
            "sagemaker": {"role_arn": "arn:aws:iam::1:role/r"}}


def _job():
    return {"job_name": "exp-1", "image_uri": "img", "input_s3": "s3://b/in/",
            "output_s3": "s3://b/out/", "hyperparameters": {"max_depth": "5"},
            "metric_regex": "f1=(.*?);", "instance_type": "ml.m5.xlarge"}


def test_launch_training_builds_request():
    client = MagicMock()
    sm_launch.launch_training(_cfg(), client, _job())
    args = client.create_training_job.call_args.kwargs
    assert args["TrainingJobName"] == "exp-1"
    assert args["AlgorithmSpecification"]["TrainingImage"] == "img"
    assert args["HyperParameters"] == {"max_depth": "5"}
    assert args["ResourceConfig"]["InstanceType"] == "ml.m5.xlarge"


def test_launch_training_includes_metric_definitions_when_regex_given():
    client = MagicMock()
    sm_launch.launch_training(_cfg(), client, _job())
    spec = client.create_training_job.call_args.kwargs["AlgorithmSpecification"]
    assert spec["MetricDefinitions"] == [{"Name": "objective", "Regex": "f1=(.*?);"}]


def test_launch_training_omits_metric_definitions_without_regex():
    """Built-in algorithm images reject MetricDefinitions outright.

    Regression: SageMaker returned ValidationException "You can't override the
    metric definitions for Amazon SageMaker algorithms".
    """
    client = MagicMock()
    job = _job()
    del job["metric_regex"]
    sm_launch.launch_training(_cfg(), client, job)
    spec = client.create_training_job.call_args.kwargs["AlgorithmSpecification"]
    assert "MetricDefinitions" not in spec


def test_poll_returns_metric_on_completion():
    client = MagicMock()
    client.describe_training_job.return_value = {
        "TrainingJobStatus": "Completed",
        "FinalMetricDataList": [{"MetricName": "f1", "Value": 0.92}],
        "TrainingStartTime": 0, "TrainingEndTime": 0,
    }
    out = sm_launch.poll(client, "exp-1")
    assert out["status"] == "Completed"
    assert out["final_metric"] == 0.92


def test_poll_returns_failure_reason():
    client = MagicMock()
    client.describe_training_job.return_value = {
        "TrainingJobStatus": "Failed", "FailureReason": "OOM",
        "FinalMetricDataList": [],
    }
    out = sm_launch.poll(client, "exp-1")
    assert out["status"] == "Failed"
    assert out["failure_reason"] == "OOM"
    assert out["final_metric"] is None


def test_estimate_cost_multiplies_rate_by_hours():
    price = {"ml.m5.xlarge": 0.23}
    # 3600 seconds = 1 hour
    assert round(sm_launch.estimate_cost("ml.m5.xlarge", 3600, price), 2) == 0.23


def test_estimate_cost_uses_explicit_rate_over_map():
    # Live Pricing API rate must win over the coarse fallback map.
    assert round(sm_launch.estimate_cost("ml.p9.xlarge", 3600, {}, rate=1.5), 2) == 1.50


def test_estimate_cost_multiplies_by_instance_count():
    price = {"ml.m5.xlarge": 0.23}
    got = sm_launch.estimate_cost("ml.m5.xlarge", 3600, price, instance_count=4)
    assert round(got, 2) == 0.92


def test_estimate_cost_unknown_type_is_zero_not_an_error():
    # Returns 0.0, which is why callers must surface rate_source; a silent zero
    # would make the budget cap unenforceable.
    assert sm_launch.estimate_cost("ml.unknown", 3600, {}) == 0.0


def test_poll_prefers_sagemaker_billable_time():
    """BillableTimeInSeconds includes provisioning; Training*Time does not."""
    client = MagicMock()
    client.describe_training_job.return_value = {
        "TrainingJobStatus": "Completed",
        "FinalMetricDataList": [{"MetricName": "f1", "Value": 0.9}],
        "BillableTimeInSeconds": 300,
        "TrainingStartTime": 0, "TrainingEndTime": 100,  # would undercount at 100
        "ResourceConfig": {"InstanceCount": 2},
    }
    out = sm_launch.poll(client, "exp-1")
    assert out["billable_seconds"] == 300
    assert out["billable_exact"] is True
    assert out["instance_count"] == 2


def test_poll_falls_back_to_training_times_and_flags_inexact():
    client = MagicMock()
    client.describe_training_job.return_value = {
        "TrainingJobStatus": "Completed",
        "FinalMetricDataList": [],
        "TrainingStartTime": 0, "TrainingEndTime": 120,
    }
    out = sm_launch.poll(client, "exp-1")
    assert out["billable_seconds"] == 120
    assert out["billable_exact"] is False


def test_lookup_hourly_rate_picks_training_family():
    """Must not price a training job off a notebook or endpoint usagetype."""
    import json as _json

    session = MagicMock()
    session.client.return_value.get_products.return_value = {
        "PriceList": [
            _json.dumps({
                "product": {"attributes": {"usagetype": "USW2-Notebk:ml.m5.xlarge"}},
                "terms": {"OnDemand": {"a": {"priceDimensions": {
                    "b": {"pricePerUnit": {"USD": "9.99"}}}}}},
            }),
            _json.dumps({
                "product": {"attributes": {"usagetype": "USW2-Train:ml.m5.xlarge"}},
                "terms": {"OnDemand": {"a": {"priceDimensions": {
                    "b": {"pricePerUnit": {"USD": "0.23"}}}}}},
            }),
        ]
    }
    rate = sm_launch.lookup_hourly_rate(session, "ml.m5.xlarge", "us-west-2")
    assert rate == 0.23


def test_lookup_hourly_rate_returns_none_on_api_failure():
    session = MagicMock()
    session.client.side_effect = RuntimeError("no pricing permission")
    assert sm_launch.lookup_hourly_rate(session, "ml.m5.xlarge", "us-west-2") is None
