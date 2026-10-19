import pathlib
import tarfile
import tempfile
from unittest.mock import MagicMock

import pytest

from sm_research import package_code


def _cfg():
    return {
        "aws": {"profile": "p", "region": "us-west-2"},
        "sagemaker": {"s3_bucket": "bk", "role_arn": "arn:aws:iam::1:role/r"},
    }


@pytest.fixture
def scripts(tmp_path):
    a = tmp_path / "sklearn_train.py"
    a.write_text("print('a')\n")
    b = tmp_path / "xgboost_train.py"
    b.write_text("print('b')\n")
    return [str(a), str(b)]


def test_build_tarball_places_scripts_at_archive_root(scripts):
    """SageMaker extracts into /opt/ml/code, so no nested directories."""
    with tempfile.TemporaryDirectory() as tmp:
        dest = pathlib.Path(tmp) / "sourcedir.tar.gz"
        names = package_code.build_tarball(scripts, str(dest))
        assert sorted(names) == ["sklearn_train.py", "xgboost_train.py"]
        with tarfile.open(dest) as tar:
            members = tar.getnames()
        # Bare filenames only — no "./" prefix or parent dirs.
        assert sorted(members) == ["sklearn_train.py", "xgboost_train.py"]


def test_build_tarball_rejects_missing_script(tmp_path):
    with tempfile.TemporaryDirectory() as tmp:
        dest = pathlib.Path(tmp) / "s.tar.gz"
        with pytest.raises(FileNotFoundError):
            package_code.build_tarball([str(tmp_path / "nope.py")], str(dest))


def test_package_returns_launch_hyperparameters(scripts):
    s3 = MagicMock()
    out = package_code.package(_cfg(), s3, scripts, "sklearn_train.py",
                               run_id="r1", experiment="e2")
    assert out["hyperparameters"] == {
        "sagemaker_program": "sklearn_train.py",
        "sagemaker_submit_directory":
            "s3://bk/sm-research/r1/code/e2/sourcedir.tar.gz",
    }
    assert out["size_bytes"] > 0
    s3.upload_file.assert_called_once()


def test_package_scopes_key_per_experiment(scripts):
    """Concurrent experiments must not overwrite each other's code."""
    s3 = MagicMock()
    a = package_code.package(_cfg(), s3, scripts, "sklearn_train.py",
                             run_id="r1", experiment="exp1")
    b = package_code.package(_cfg(), s3, scripts, "sklearn_train.py",
                             run_id="r1", experiment="exp2")
    assert a["sourcedir_s3"] != b["sourcedir_s3"]


def test_package_rejects_entry_point_not_in_archive(scripts):
    """Regression: mismatched entry point failed only inside the container.

    The real job died with "can't open file '/opt/ml/code/sklearn_train.py'"
    after provisioning an instance; catching it locally is far cheaper.
    """
    s3 = MagicMock()
    with pytest.raises(ValueError, match="not among the packaged scripts"):
        package_code.package(_cfg(), s3, [scripts[1]], "sklearn_train.py")
    s3.upload_file.assert_not_called()
