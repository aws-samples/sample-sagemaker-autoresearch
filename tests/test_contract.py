import json

import pytest

from sm_research import contract


def test_ok_shape():
    assert contract.ok({"a": 1}) == {"status": "ok", "data": {"a": 1}, "error": None}


def test_err_shape():
    assert contract.err("boom") == {"status": "error", "data": {}, "error": "boom"}


def test_emit_ok_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        contract.emit(contract.ok({"x": 2}))
    assert exc.value.code == 0
    assert json.loads(capsys.readouterr().out) == {"status": "ok", "data": {"x": 2}, "error": None}


def test_emit_err_exits_one(capsys):
    with pytest.raises(SystemExit) as exc:
        contract.emit(contract.err("nope"))
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out)["error"] == "nope"
