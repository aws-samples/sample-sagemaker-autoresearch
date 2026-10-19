from unittest.mock import MagicMock

from botocore.exceptions import ClientError

from sm_research import register_model


def _cfg():
    return {"aws": {"region": "us-east-1"},
            "model_registry": {"package_group": "grp"}}


def _model():
    return {"image_uri": "img", "model_data_s3": "s3://b/model.tar.gz",
            "metric_name": "f1", "metric_value": 0.93}


def test_creates_group_if_missing_then_registers():
    client = MagicMock()
    client.describe_model_package_group.side_effect = ClientError(
        {"Error": {"Code": "ValidationException"}}, "DescribeModelPackageGroup")
    client.create_model_package.return_value = {"ModelPackageArn": "arn:pkg"}
    out = register_model.register(_cfg(), client, _model())
    client.create_model_package_group.assert_called_once()
    assert out["model_package_arn"] == "arn:pkg"


def test_skips_group_create_if_present():
    client = MagicMock()
    client.describe_model_package_group.return_value = {"ModelPackageGroupName": "grp"}
    client.create_model_package.return_value = {"ModelPackageArn": "arn:pkg2"}
    out = register_model.register(_cfg(), client, _model())
    client.create_model_package_group.assert_not_called()
    assert out["model_package_arn"] == "arn:pkg2"
