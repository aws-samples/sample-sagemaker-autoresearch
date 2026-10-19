from unittest.mock import MagicMock

from sm_research import mlflow_ensure


def _cfg(arn=""):
    return {
        "aws": {"region": "us-east-1"},
        "sagemaker": {"role_arn": "arn:aws:iam::1:role/r", "s3_bucket": "bk",
                      "mlflow": {"tracking_server_arn": arn, "experiment_name": "e"}},
    }


def test_uses_configured_arn_without_listing():
    client = MagicMock()
    out = mlflow_ensure.ensure(_cfg("arn:aws:sagemaker:...:server/s"), client, confirm_create=False)
    assert out["tracking_server_arn"] == "arn:aws:sagemaker:...:server/s"
    assert out["created"] is False
    client.list_mlflow_tracking_servers.assert_not_called()


def test_discovers_existing_server():
    client = MagicMock()
    client.list_mlflow_tracking_servers.return_value = {
        "TrackingServerSummaries": [{"TrackingServerArn": "arn:existing"}]
    }
    out = mlflow_ensure.ensure(_cfg(), client, confirm_create=False)
    assert out["tracking_server_arn"] == "arn:existing"
    assert out["created"] is False


def test_requires_confirmation_when_none_exist():
    client = MagicMock()
    client.list_mlflow_tracking_servers.return_value = {"TrackingServerSummaries": []}
    out = mlflow_ensure.ensure(_cfg(), client, confirm_create=False)
    assert out["needs_confirmation"] is True
    assert out["created"] is False
    client.create_mlflow_tracking_server.assert_not_called()


def test_creates_when_confirmed():
    client = MagicMock()
    client.list_mlflow_tracking_servers.return_value = {"TrackingServerSummaries": []}
    client.create_mlflow_tracking_server.return_value = {"TrackingServerArn": "arn:new"}
    out = mlflow_ensure.ensure(_cfg(), client, confirm_create=True)
    assert out["created"] is True
    assert out["tracking_server_arn"] == "arn:new"
    client.create_mlflow_tracking_server.assert_called_once()
