from unittest.mock import MagicMock

from botocore.exceptions import ClientError

from sm_research import run_training


def _not_found():
    return ClientError({"Error": {"Code": "ValidationException",
                                  "Message": "Requested resource not found."}},
                       "DescribeTrainingJob")


def test_existing_attempts_probes_with_describe_not_list():
    """Regression: paginating ListTrainingJobs throttled ("Rate exceeded")
    under concurrent experiments, so run_training errored before launching."""
    sm = MagicMock()
    sm.describe_training_job.side_effect = [{}, {}, _not_found()]
    assert run_training.existing_attempts(sm, "run-r1-e1") == ["run-r1-e1-a1", "run-r1-e1-a2"]
    sm.get_paginator.assert_not_called()
    names = [c.kwargs["TrainingJobName"] for c in sm.describe_training_job.call_args_list]
    assert names == ["run-r1-e1-a1", "run-r1-e1-a2", "run-r1-e1-a3"]


def test_existing_attempts_empty_on_first_attempt():
    sm = MagicMock()
    sm.describe_training_job.side_effect = _not_found()
    assert run_training.existing_attempts(sm, "p") == []


def test_existing_attempts_reraises_other_errors():
    sm = MagicMock()
    sm.describe_training_job.side_effect = ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "denied"}}, "DescribeTrainingJob")
    try:
        run_training.existing_attempts(sm, "p")
    except ClientError:
        pass
    else:
        raise AssertionError("expected ClientError")


def test_with_target_passes_data_target_to_scripts_that_take_it(tmp_path):
    """Regression: scripts defaulting --target to "target" failed with
    KeyError: 'target' whenever the caller forgot the hyperparameter."""
    cfg = {"data": {"target": "churned"}}
    takes = tmp_path / "train.py"
    takes.write_text('p.add_argument("--target", default="target")\n')
    plain = tmp_path / "plain.py"
    plain.write_text("print('no target arg')\n")
    assert run_training.with_target({"eta": 0.1}, cfg, str(takes)) == {"eta": 0.1, "target": "churned"}
    assert run_training.with_target({"target": "y"}, cfg, str(takes)) == {"target": "y"}
    assert run_training.with_target({}, cfg, str(plain)) == {}
    assert run_training.with_target({}, {"data": {}}, str(takes)) == {}


def test_entry_script_matches_entry_point():
    assert run_training.entry_script(["/a/helper.py", "/b/train.py"], "train.py") == "/b/train.py"
    assert run_training.entry_script(["/a/x.py"], None) == "/a/x.py"
