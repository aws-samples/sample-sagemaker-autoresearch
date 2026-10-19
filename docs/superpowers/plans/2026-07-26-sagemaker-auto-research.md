# SageMaker Auto Research Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Claude Code plugin that runs autonomous ML experimentation loops on Amazon SageMaker — from Redshift data through feature engineering, training, MLflow tracking, and model registration — until a user-defined success metric is met or a guardrail stops it.

**Architecture:** Approach A (workflow-centric). A `/sm-research` slash command interviews the user and scaffolds a reproducible `research.yaml`. A `sagemaker-auto-research` skill is the orchestration contract that validates config, runs Redshift discovery, ensures MLflow, launches the Workflow, checks stop conditions between rounds, and registers the winner. A Workflow script (`auto-research-loop.js`) is the deterministic engine that fans out experiments across subagents and ranks them with a judge stage. Thin, unit-tested Python scripts wrap all AWS calls; pure JS functions handle stop-condition/judge math.

**Tech Stack:** Python 3.12 + `uv` (boto3, PyYAML, pytest, moto), Node.js 22 (Workflow JS, no external deps), Claude Code plugin conventions.

## Global Constraints

- Python environments and dependencies MUST be managed with `uv`; never `pip install` directly.
- All AWS calls MUST use the user's AWS profile and its region (both come from `research.yaml`); never the default profile.
- Prefer read-only / least-privilege AWS operations; describe/list over modify/delete.
- No AWS resource creation without explicit user confirmation stating estimated cost first (applies to the MLflow tracking server).
- No AWS resource deletion under any circumstances in this plugin.
- The agent may only touch Redshift data listed in `data.allowed`; never beyond it.
- Every Python script's stdout is a single JSON object of the form `{"status": "ok"|"error", "data": <object>, "error": <string|null>}` and exits 0 on `ok`, 1 on `error`.
- Intra-plugin path references use `${CLAUDE_PLUGIN_ROOT}`; never hardcoded/absolute/`~` paths.
- All file/directory names use kebab-case.

---

## File Structure

```
sagemaker-auto-research/
├── .claude-plugin/plugin.json          Plugin manifest
├── pyproject.toml                      uv-managed Python project (scripts + tests)
├── commands/sm-research.md             /sm-research command
├── skills/sagemaker-auto-research/SKILL.md   Orchestration contract
├── workflows/auto-research-loop.js     Workflow engine
├── workflows/lib/decisions.js          Pure stop-condition + judge functions
├── workflows/lib/decisions.test.js     Node test for decisions.js
├── scripts/sm_research/__init__.py
├── scripts/sm_research/contract.py     JSON result envelope helper
├── scripts/sm_research/config.py       research.yaml load + validate
├── scripts/sm_research/slicing.py      data-slice content hashing
├── scripts/sm_research/redshift_discover.py
├── scripts/sm_research/redshift_unload.py
├── scripts/sm_research/mlflow_ensure.py
├── scripts/sm_research/sm_launch.py
├── scripts/sm_research/register_model.py
├── templates/research.yaml.tmpl
├── templates/train/xgboost_train.py
├── templates/train/sklearn_train.py
└── tests/                              pytest suite mirroring scripts/
```

Responsibilities: `contract.py`/`config.py`/`slicing.py` are pure, dependency-light units. Each AWS wrapper is one file with one job. `decisions.js` holds all loop math as pure functions so it is testable without AWS. The workflow wires wrappers + decisions together; the skill decides when to run them.

---

### Task 1: Plugin scaffold + Python project

**Files:**
- Create: `.claude-plugin/plugin.json`
- Create: `pyproject.toml`
- Create: `scripts/sm_research/__init__.py`
- Create: `scripts/sm_research/contract.py`
- Test: `tests/test_contract.py`

**Interfaces:**
- Produces: `contract.ok(data: dict) -> dict`, `contract.err(message: str) -> dict` returning `{"status","data","error"}`; `contract.emit(result: dict) -> NoReturn` which prints JSON to stdout and calls `sys.exit(0 if ok else 1)`.

- [ ] **Step 1: Create the plugin manifest**

`.claude-plugin/plugin.json`:
```json
{
  "name": "sagemaker-auto-research",
  "version": "0.1.0",
  "description": "Autonomous ML experimentation loops on Amazon SageMaker — Redshift data to a registered model that meets a success metric.",
  "author": { "name": "rddefauw" },
  "license": "MIT",
  "keywords": ["sagemaker", "mlflow", "redshift", "automl", "auto-research"]
}
```

- [ ] **Step 2: Create the uv project file**

`pyproject.toml`:
```toml
[project]
name = "sm-research-scripts"
version = "0.1.0"
description = "SageMaker Auto Research helper scripts"
requires-python = ">=3.12"
dependencies = [
    "boto3>=1.34",
    "PyYAML>=6.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "moto>=5.0",
]

[tool.pytest.ini_options]
pythonpath = ["scripts"]
testpaths = ["tests"]
```

- [ ] **Step 3: Sync the environment**

Run: `uv sync`
Expected: creates `.venv`, resolves boto3/PyYAML/pytest/moto, exits 0.

- [ ] **Step 4: Write the failing test**

`tests/test_contract.py`:
```python
import json
import subprocess
import sys
import textwrap

from sm_research import contract


def test_ok_shape():
    assert contract.ok({"a": 1}) == {"status": "ok", "data": {"a": 1}, "error": None}


def test_err_shape():
    assert contract.err("boom") == {"status": "error", "data": {}, "error": "boom"}


def test_emit_ok_exits_zero(tmp_path):
    prog = tmp_path / "p.py"
    prog.write_text(textwrap.dedent("""
        from sm_research import contract
        contract.emit(contract.ok({"x": 2}))
    """))
    proc = subprocess.run([sys.executable, str(prog)], capture_output=True, text=True,
                          env={"PYTHONPATH": "scripts"})
    assert proc.returncode == 0
    assert json.loads(proc.stdout) == {"status": "ok", "data": {"x": 2}, "error": None}


def test_emit_err_exits_one(tmp_path):
    prog = tmp_path / "p.py"
    prog.write_text(textwrap.dedent("""
        from sm_research import contract
        contract.emit(contract.err("nope"))
    """))
    proc = subprocess.run([sys.executable, str(prog)], capture_output=True, text=True,
                          env={"PYTHONPATH": "scripts"})
    assert proc.returncode == 1
    assert json.loads(proc.stdout)["error"] == "nope"
```

- [ ] **Step 5: Run test to verify it fails**

Run: `uv run pytest tests/test_contract.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sm_research'` (package/module not yet created).

- [ ] **Step 6: Create the package and contract module**

`scripts/sm_research/__init__.py`:
```python
```
(empty file)

`scripts/sm_research/contract.py`:
```python
"""Standard JSON result envelope for all sm_research scripts."""
import json
import sys


def ok(data: dict) -> dict:
    return {"status": "ok", "data": data, "error": None}


def err(message: str) -> dict:
    return {"status": "error", "data": {}, "error": message}


def emit(result: dict):
    """Print the result as JSON and exit 0 on ok, 1 on error."""
    print(json.dumps(result))
    sys.exit(0 if result.get("status") == "ok" else 1)
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_contract.py -v`
Expected: PASS (4 passed).

- [ ] **Step 8: Commit**

```bash
git add .claude-plugin/plugin.json pyproject.toml uv.lock scripts/sm_research/__init__.py scripts/sm_research/contract.py tests/test_contract.py
git commit -m "feat: plugin scaffold + result-envelope contract"
```

---

### Task 2: Config load + validation

**Files:**
- Create: `scripts/sm_research/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `contract` from Task 1.
- Produces: `config.validate(cfg: dict) -> list[str]` returning a list of human-readable error strings (empty = valid); `config.load(path: str) -> dict` (parses YAML, returns the dict); `config.REQUIRED_SECTIONS = ["aws","data","problem","budget","sagemaker","model_registry"]`.

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:
```python
from sm_research import config


def _valid_cfg():
    return {
        "aws": {"profile": "p", "region": "us-east-1"},
        "data": {
            "redshift": {"workgroup_or_cluster": "wg", "database": "db"},
            "allowed": [{"schema": "public", "table": "t", "columns": ["a"]}],
            "target": "y",
        },
        "problem": {"type": "binary_classification", "success_metric": "f1",
                    "threshold": 0.9, "metric_goal": "maximize"},
        "budget": {"max_rounds": 5, "experiments_per_round": 4,
                   "max_total_cost_usd": 100, "max_wallclock_hours": 6,
                   "plateau": {"patience_rounds": 2, "min_delta": 0.005}},
        "sagemaker": {"role_arn": "arn:aws:iam::1:role/r", "s3_bucket": "b",
                      "instance_type": "ml.m5.xlarge",
                      "mlflow": {"tracking_server_arn": "", "experiment_name": "e"}},
        "model_registry": {"package_group": "g"},
    }


def test_valid_config_has_no_errors():
    assert config.validate(_valid_cfg()) == []


def test_missing_section_reported():
    cfg = _valid_cfg()
    del cfg["problem"]
    errs = config.validate(cfg)
    assert any("problem" in e for e in errs)


def test_empty_allowlist_rejected():
    cfg = _valid_cfg()
    cfg["data"]["allowed"] = []
    errs = config.validate(cfg)
    assert any("allowed" in e for e in errs)


def test_threshold_must_be_numeric():
    cfg = _valid_cfg()
    cfg["problem"]["threshold"] = "high"
    errs = config.validate(cfg)
    assert any("threshold" in e for e in errs)


def test_metric_goal_enum_enforced():
    cfg = _valid_cfg()
    cfg["problem"]["metric_goal"] = "sideways"
    errs = config.validate(cfg)
    assert any("metric_goal" in e for e in errs)


def test_load_reads_yaml(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text("aws:\n  profile: p\n  region: us-east-1\n")
    assert config.load(str(p))["aws"]["profile"] == "p"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sm_research.config'`.

- [ ] **Step 3: Implement config module**

`scripts/sm_research/config.py`:
```python
"""Load and validate research.yaml."""
import numbers

import yaml

REQUIRED_SECTIONS = ["aws", "data", "problem", "budget", "sagemaker", "model_registry"]
_METRIC_GOALS = {"maximize", "minimize"}


def load(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _is_number(v) -> bool:
    return isinstance(v, numbers.Number) and not isinstance(v, bool)


def validate(cfg: dict) -> list[str]:
    errs: list[str] = []
    for section in REQUIRED_SECTIONS:
        if section not in cfg:
            errs.append(f"missing required section: {section}")
    if errs:
        return errs  # deeper checks assume sections exist

    aws = cfg["aws"]
    for k in ("profile", "region"):
        if not aws.get(k):
            errs.append(f"aws.{k} is required")

    data = cfg["data"]
    if not data.get("allowed"):
        errs.append("data.allowed must list at least one table")
    if not data.get("target"):
        errs.append("data.target (label column) is required")

    prob = cfg["problem"]
    if not _is_number(prob.get("threshold")):
        errs.append("problem.threshold must be numeric")
    if prob.get("metric_goal") not in _METRIC_GOALS:
        errs.append("problem.metric_goal must be 'maximize' or 'minimize'")
    if not prob.get("success_metric"):
        errs.append("problem.success_metric is required")

    budget = cfg["budget"]
    for k in ("max_rounds", "experiments_per_round"):
        if not isinstance(budget.get(k), int) or budget.get(k) < 1:
            errs.append(f"budget.{k} must be a positive integer")
    plateau = budget.get("plateau", {})
    if not isinstance(plateau.get("patience_rounds"), int):
        errs.append("budget.plateau.patience_rounds must be an integer")

    sm = cfg["sagemaker"]
    for k in ("role_arn", "s3_bucket", "instance_type"):
        if not sm.get(k):
            errs.append(f"sagemaker.{k} is required")

    if not cfg["model_registry"].get("package_group"):
        errs.append("model_registry.package_group is required")

    return errs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/sm_research/config.py tests/test_config.py
git commit -m "feat: research.yaml load + validation"
```

---

### Task 3: Data-slice content hashing

**Files:**
- Create: `scripts/sm_research/slicing.py`
- Test: `tests/test_slicing.py`

**Interfaces:**
- Produces: `slicing.slice_hash(spec: dict) -> str` — a stable 16-char hex digest of a data-slice spec (tables, columns, filters, split seed), order-independent; `slicing.s3_prefix(bucket: str, run_id: str, spec: dict) -> str` returning `s3://<bucket>/sm-research/<run_id>/slices/<hash>/`.

- [ ] **Step 1: Write the failing test**

`tests/test_slicing.py`:
```python
from sm_research import slicing


def test_hash_is_stable_and_order_independent():
    a = {"tables": ["t1", "t2"], "columns": ["b", "a"], "filter": "x>1", "seed": 42}
    b = {"seed": 42, "filter": "x>1", "columns": ["a", "b"], "tables": ["t2", "t1"]}
    assert slicing.slice_hash(a) == slicing.slice_hash(b)
    assert len(slicing.slice_hash(a)) == 16


def test_hash_changes_with_content():
    a = {"tables": ["t1"], "columns": ["a"], "filter": None, "seed": 1}
    b = {"tables": ["t1"], "columns": ["a"], "filter": None, "seed": 2}
    assert slicing.slice_hash(a) != slicing.slice_hash(b)


def test_s3_prefix_format():
    spec = {"tables": ["t1"], "columns": ["a"], "filter": None, "seed": 1}
    h = slicing.slice_hash(spec)
    assert slicing.s3_prefix("bk", "run9", spec) == f"s3://bk/sm-research/run9/slices/{h}/"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_slicing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sm_research.slicing'`.

- [ ] **Step 3: Implement slicing module**

`scripts/sm_research/slicing.py`:
```python
"""Content-addressable hashing for data slices so UNLOADs can be cached."""
import hashlib
import json


def slice_hash(spec: dict) -> str:
    # sort_keys + sorted lists make the digest order-independent
    normalized = _normalize(spec)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _normalize(value):
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    if isinstance(value, list):
        return sorted((_normalize(v) for v in value), key=lambda x: json.dumps(x, sort_keys=True))
    return value


def s3_prefix(bucket: str, run_id: str, spec: dict) -> str:
    return f"s3://{bucket}/sm-research/{run_id}/slices/{slice_hash(spec)}/"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_slicing.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/sm_research/slicing.py tests/test_slicing.py
git commit -m "feat: content-addressable data-slice hashing"
```

---

### Task 4: Redshift discovery (Data API)

**Files:**
- Create: `scripts/sm_research/redshift_discover.py`
- Test: `tests/test_redshift_discover.py`

**Interfaces:**
- Consumes: `contract`, `config`.
- Produces: CLI `python -m sm_research.redshift_discover --config <path> [--sample-rows N]`. Internally: `redshift_discover.build_discovery_sql(allowed: list, target: str, sample_rows: int) -> list[dict]` returning per-table `{"table","count_sql","sample_sql"}`; `redshift_discover.run(cfg: dict, client, sample_rows: int) -> dict` returning `{"tables":[{"table","row_count","columns","sample":[...]}], "target":<str>}`. `client` is a boto3 `redshift-data` client (injected for testability).

- [ ] **Step 1: Write the failing test**

`tests/test_redshift_discover.py`:
```python
from unittest.mock import MagicMock

from sm_research import redshift_discover


def test_build_discovery_sql_scopes_to_allowlist():
    allowed = [{"schema": "public", "table": "events", "columns": ["a", "b"]}]
    specs = redshift_discover.build_discovery_sql(allowed, "y", sample_rows=5)
    assert len(specs) == 1
    assert 'public"."events"' in specs[0]["count_sql"]
    assert "limit 5" in specs[0]["sample_sql"].lower()


def test_run_aggregates_counts_and_samples():
    cfg = {
        "aws": {"region": "us-east-1"},
        "data": {
            "redshift": {"workgroup_or_cluster": "wg", "database": "db"},
            "allowed": [{"schema": "public", "table": "events", "columns": ["a"]}],
            "target": "y",
        },
    }
    client = MagicMock()
    client.execute_statement.return_value = {"Id": "sid"}
    client.describe_statement.return_value = {"Status": "FINISHED"}
    # First get_statement_result = count query; second = sample query
    client.get_statement_result.side_effect = [
        {"Records": [[{"longValue": 100}]],
         "ColumnMetadata": [{"name": "count"}]},
        {"Records": [[{"stringValue": "v1"}]],
         "ColumnMetadata": [{"name": "a"}]},
    ]
    out = redshift_discover.run(cfg, client, sample_rows=5)
    assert out["target"] == "y"
    assert out["tables"][0]["row_count"] == 100
    assert out["tables"][0]["sample"] == [{"a": "v1"}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_redshift_discover.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sm_research.redshift_discover'`.

- [ ] **Step 3: Implement discovery module**

`scripts/sm_research/redshift_discover.py`:
```python
"""Redshift Data API discovery: row counts + samples, scoped to the allowlist."""
import argparse
import time

import boto3

from sm_research import config, contract


def _qual(schema: str, table: str) -> str:
    return f'"{schema}"."{table}"'


def build_discovery_sql(allowed: list, target: str, sample_rows: int) -> list[dict]:
    specs = []
    for t in allowed:
        fq = _qual(t["schema"], t["table"])
        specs.append({
            "table": f'{t["schema"]}.{t["table"]}',
            "count_sql": f"select count(*) as count from {fq}",
            "sample_sql": f"select * from {fq} limit {sample_rows}",
        })
    return specs


def _execute(client, cfg_rs: dict, region: str, sql: str) -> dict:
    kwargs = {"Database": cfg_rs["database"], "Sql": sql}
    if "secret_arn" in cfg_rs and cfg_rs.get("secret_arn"):
        kwargs["SecretArn"] = cfg_rs["secret_arn"]
    wg = cfg_rs["workgroup_or_cluster"]
    # Serverless workgroups use WorkgroupName; provisioned clusters ClusterIdentifier.
    if cfg_rs.get("serverless", True):
        kwargs["WorkgroupName"] = wg
    else:
        kwargs["ClusterIdentifier"] = wg
    started = client.execute_statement(**kwargs)
    sid = started["Id"]
    while True:
        desc = client.describe_statement(Id=sid)
        status = desc["Status"]
        if status == "FINISHED":
            break
        if status in ("FAILED", "ABORTED"):
            raise RuntimeError(desc.get("Error", f"statement {status}"))
        time.sleep(1)
    return client.get_statement_result(Id=sid)


def _rows_to_dicts(result: dict) -> list[dict]:
    cols = [c["name"] for c in result["ColumnMetadata"]]
    out = []
    for rec in result["Records"]:
        row = {}
        for name, cell in zip(cols, rec):
            row[name] = next(iter(cell.values())) if cell else None
        out.append(row)
    return out


def run(cfg: dict, client, sample_rows: int) -> dict:
    rs = cfg["data"]["redshift"]
    region = cfg["aws"]["region"]
    allowed = cfg["data"]["allowed"]
    target = cfg["data"]["target"]
    tables = []
    for spec in build_discovery_sql(allowed, target, sample_rows):
        count_res = _execute(client, rs, region, spec["count_sql"])
        row_count = _rows_to_dicts(count_res)[0]["count"]
        sample_res = _execute(client, rs, region, spec["sample_sql"])
        sample = _rows_to_dicts(sample_res)
        columns = [c["name"] for c in sample_res["ColumnMetadata"]]
        tables.append({"table": spec["table"], "row_count": row_count,
                       "columns": columns, "sample": sample})
    return {"tables": tables, "target": target}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-rows", type=int, default=10)
    args = parser.parse_args()
    try:
        cfg = config.load(args.config)
        errs = config.validate(cfg)
        if errs:
            contract.emit(contract.err("; ".join(errs)))
        session = boto3.Session(profile_name=cfg["aws"]["profile"],
                                region_name=cfg["aws"]["region"])
        client = session.client("redshift-data")
        contract.emit(contract.ok(run(cfg, client, args.sample_rows)))
    except Exception as e:  # noqa: BLE001 — surface AWS error verbatim
        contract.emit(contract.err(str(e)))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_redshift_discover.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/sm_research/redshift_discover.py tests/test_redshift_discover.py
git commit -m "feat: Redshift Data API discovery scoped to allowlist"
```

---

### Task 5: Redshift UNLOAD with cache

**Files:**
- Create: `scripts/sm_research/redshift_unload.py`
- Test: `tests/test_redshift_unload.py`

**Interfaces:**
- Consumes: `contract`, `config`, `slicing`, and `redshift_discover._execute` (reuse the polling executor).
- Produces: CLI `python -m sm_research.redshift_unload --config <path> --spec-json <json>`. Internally: `redshift_unload.build_unload_sql(spec: dict, s3_prefix: str, iam_role: str) -> str`; `redshift_unload.unload(cfg, client, s3, spec) -> dict` returning `{"s3_prefix","cached":bool,"slice_hash"}`. `spec` = `{"tables":[...],"columns":[...],"filter":<str|None>,"seed":<int>,"select_sql":<str>}`. `s3` is a boto3 S3 client (injected).

- [ ] **Step 1: Write the failing test**

`tests/test_redshift_unload.py`:
```python
from unittest.mock import MagicMock

from sm_research import redshift_unload, slicing


def _cfg():
    return {
        "aws": {"region": "us-east-1"},
        "data": {"redshift": {"workgroup_or_cluster": "wg", "database": "db"}},
        "sagemaker": {"s3_bucket": "bk", "role_arn": "arn:aws:iam::1:role/r"},
    }


def _spec():
    return {"tables": ["public.events"], "columns": ["a", "b"], "filter": None,
            "seed": 1, "select_sql": "select a, b from public.events"}


def test_build_unload_sql_wraps_select_and_targets_prefix():
    sql = redshift_unload.build_unload_sql(_spec(), "s3://bk/x/", "arn:aws:iam::1:role/r")
    assert "unload" in sql.lower()
    assert "s3://bk/x/" in sql
    assert "parquet" in sql.lower()


def test_unload_skips_when_cached():
    cfg, spec = _cfg(), _spec()
    s3 = MagicMock()
    s3.list_objects_v2.return_value = {"KeyCount": 3}  # slice already present
    client = MagicMock()
    out = redshift_unload.unload(cfg, client, s3, spec, run_id="run1")
    assert out["cached"] is True
    client.execute_statement.assert_not_called()
    assert out["s3_prefix"] == slicing.s3_prefix("bk", "run1", spec)


def test_unload_runs_when_not_cached():
    cfg, spec = _cfg(), _spec()
    s3 = MagicMock()
    s3.list_objects_v2.return_value = {"KeyCount": 0}
    client = MagicMock()
    client.execute_statement.return_value = {"Id": "sid"}
    client.describe_statement.return_value = {"Status": "FINISHED"}
    client.get_statement_result.return_value = {"Records": [], "ColumnMetadata": []}
    out = redshift_unload.unload(cfg, client, s3, spec, run_id="run1")
    assert out["cached"] is False
    client.execute_statement.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_redshift_unload.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sm_research.redshift_unload'`.

- [ ] **Step 3: Implement unload module**

`scripts/sm_research/redshift_unload.py`:
```python
"""UNLOAD an allowed data slice to S3 as Parquet, cached by content hash."""
import argparse
import json

import boto3

from sm_research import config, contract, slicing
from sm_research.redshift_discover import _execute


def build_unload_sql(spec: dict, s3_prefix: str, iam_role: str) -> str:
    inner = spec["select_sql"].replace("'", "''")
    return (
        f"unload ('{inner}') "
        f"to '{s3_prefix}' "
        f"iam_role '{iam_role}' "
        f"format as parquet allowoverwrite parallel off"
    )


def _prefix_has_objects(s3, s3_uri: str) -> bool:
    without = s3_uri[len("s3://"):]
    bucket, _, key = without.partition("/")
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1)
    return resp.get("KeyCount", 0) > 0


def unload(cfg: dict, client, s3, spec: dict, run_id: str = "default") -> dict:
    bucket = cfg["sagemaker"]["s3_bucket"]
    s3_prefix = slicing.s3_prefix(bucket, run_id, spec)
    h = slicing.slice_hash(spec)
    if _prefix_has_objects(s3, s3_prefix):
        return {"s3_prefix": s3_prefix, "cached": True, "slice_hash": h, "run_id": run_id}
    sql = build_unload_sql(spec, s3_prefix, cfg["sagemaker"]["role_arn"])
    _execute(client, cfg["data"]["redshift"], cfg["aws"]["region"], sql)
    return {"s3_prefix": s3_prefix, "cached": False, "slice_hash": h, "run_id": run_id}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--spec-json", required=True)
    parser.add_argument("--run-id", default="default")
    args = parser.parse_args()
    try:
        cfg = config.load(args.config)
        spec = json.loads(args.spec_json)
        session = boto3.Session(profile_name=cfg["aws"]["profile"],
                                region_name=cfg["aws"]["region"])
        client = session.client("redshift-data")
        s3 = session.client("s3")
        contract.emit(contract.ok(unload(cfg, client, s3, spec, args.run_id)))
    except Exception as e:  # noqa: BLE001
        contract.emit(contract.err(str(e)))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_redshift_unload.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/sm_research/redshift_unload.py tests/test_redshift_unload.py
git commit -m "feat: Redshift UNLOAD to S3 parquet with content-hash cache"
```

---

### Task 6: MLflow ensure (discover / confirm-create)

**Files:**
- Create: `scripts/sm_research/mlflow_ensure.py`
- Test: `tests/test_mlflow_ensure.py`

**Interfaces:**
- Consumes: `contract`, `config`.
- Produces: CLI `python -m sm_research.mlflow_ensure --config <path> [--confirm-create]`. Internally: `mlflow_ensure.find_server(cfg, client) -> str|None` (returns ARN of configured or first available server); `mlflow_ensure.ensure(cfg, client, confirm_create: bool) -> dict` returning `{"tracking_server_arn","created":bool,"needs_confirmation":bool}`. When no server exists and `confirm_create` is False, returns `needs_confirmation=True` and does NOT create. `client` is a boto3 `sagemaker` client (injected).

- [ ] **Step 1: Write the failing test**

`tests/test_mlflow_ensure.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mlflow_ensure.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sm_research.mlflow_ensure'`.

- [ ] **Step 3: Implement mlflow_ensure module**

`scripts/sm_research/mlflow_ensure.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mlflow_ensure.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/sm_research/mlflow_ensure.py tests/test_mlflow_ensure.py
git commit -m "feat: MLflow tracking server discover/confirm-create"
```

---

### Task 7: SageMaker training-job launch + poll

**Files:**
- Create: `scripts/sm_research/sm_launch.py`
- Test: `tests/test_sm_launch.py`

**Interfaces:**
- Consumes: `contract`, `config`.
- Produces: CLI `python -m sm_research.sm_launch --config <path> --job-json <json>`. Internally: `sm_launch.launch_training(cfg, sm_client, job) -> dict`; `sm_launch.poll(sm_client, job_name) -> dict` returning `{"status","final_metric":float|None,"failure_reason":str|None}`; `sm_launch.estimate_cost(instance_type, seconds, price_map) -> float`. `job` = `{"job_name","image_uri","input_s3","output_s3","hyperparameters":{},"metric_regex","instance_type"}`.

- [ ] **Step 1: Write the failing test**

`tests/test_sm_launch.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sm_launch.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sm_research.sm_launch'`.

- [ ] **Step 3: Implement sm_launch module**

`scripts/sm_research/sm_launch.py`:
```python
"""Launch and poll SageMaker training jobs; estimate cost."""
import argparse
import json
import time

import boto3

from sm_research import config, contract

# Coarse on-demand $/hour fallbacks; overridable via job["price_map"].
DEFAULT_PRICE_MAP = {
    "ml.m5.xlarge": 0.23,
    "ml.m5.2xlarge": 0.46,
    "ml.c5.xlarge": 0.204,
}


def launch_training(cfg: dict, sm_client, job: dict):
    metric_defs = [{"Name": "objective", "Regex": job["metric_regex"]}]
    return sm_client.create_training_job(
        TrainingJobName=job["job_name"],
        RoleArn=cfg["sagemaker"]["role_arn"],
        AlgorithmSpecification={
            "TrainingImage": job["image_uri"],
            "TrainingInputMode": "File",
            "MetricDefinitions": metric_defs,
        },
        HyperParameters=job.get("hyperparameters", {}),
        InputDataConfig=[{
            "ChannelName": "training",
            "DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix", "S3Uri": job["input_s3"],
                "S3DataDistributionType": "FullyReplicated"}},
        }],
        OutputDataConfig={"S3OutputPath": job["output_s3"]},
        ResourceConfig={"InstanceType": job["instance_type"],
                        "InstanceCount": 1, "VolumeSizeInGB": 30},
        StoppingCondition={"MaxRuntimeInSeconds": 3600},
    )


def poll(sm_client, job_name: str, interval: int = 30) -> dict:
    while True:
        desc = sm_client.describe_training_job(TrainingJobName=job_name)
        status = desc["TrainingJobStatus"]
        if status in ("Completed", "Failed", "Stopped"):
            metrics = desc.get("FinalMetricDataList", [])
            final_metric = metrics[0]["Value"] if metrics else None
            seconds = 0
            if desc.get("TrainingStartTime") and desc.get("TrainingEndTime"):
                seconds = (_epoch(desc["TrainingEndTime"]) - _epoch(desc["TrainingStartTime"]))
            return {"status": status, "final_metric": final_metric,
                    "failure_reason": desc.get("FailureReason"),
                    "billable_seconds": seconds}
        time.sleep(interval)


def _epoch(v) -> float:
    return v.timestamp() if hasattr(v, "timestamp") else float(v)


def estimate_cost(instance_type: str, seconds: float, price_map: dict) -> float:
    rate = price_map.get(instance_type, 0.0)
    return rate * (seconds / 3600.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--job-json", required=True)
    args = parser.parse_args()
    try:
        cfg = config.load(args.config)
        job = json.loads(args.job_json)
        session = boto3.Session(profile_name=cfg["aws"]["profile"],
                                region_name=cfg["aws"]["region"])
        sm_client = session.client("sagemaker")
        launch_training(cfg, sm_client, job)
        result = poll(sm_client, job["job_name"])
        price_map = job.get("price_map", DEFAULT_PRICE_MAP)
        result["cost_est_usd"] = estimate_cost(
            job["instance_type"], result.get("billable_seconds", 0), price_map)
        contract.emit(contract.ok(result))
    except Exception as e:  # noqa: BLE001
        contract.emit(contract.err(str(e)))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_sm_launch.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/sm_research/sm_launch.py tests/test_sm_launch.py
git commit -m "feat: SageMaker training-job launch, poll, and cost estimate"
```

---

### Task 8: Model registration

**Files:**
- Create: `scripts/sm_research/register_model.py`
- Test: `tests/test_register_model.py`

**Interfaces:**
- Consumes: `contract`, `config`.
- Produces: CLI `python -m sm_research.register_model --config <path> --model-json <json>`. Internally: `register_model.register(cfg, sm_client, model) -> dict` returning `{"model_package_arn","package_group"}`; ensures the package group exists first (create if absent — this is a metadata container, not billable compute, so no confirmation gate). `model` = `{"image_uri","model_data_s3","metric_name","metric_value"}`.

- [ ] **Step 1: Write the failing test**

`tests/test_register_model.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_register_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sm_research.register_model'`.

- [ ] **Step 3: Implement register_model module**

`scripts/sm_research/register_model.py`:
```python
"""Register a winning model in the SageMaker Model Registry."""
import argparse
import json

import boto3
from botocore.exceptions import ClientError

from sm_research import config, contract


def _ensure_group(sm_client, group: str):
    try:
        sm_client.describe_model_package_group(ModelPackageGroupName=group)
    except ClientError:
        sm_client.create_model_package_group(
            ModelPackageGroupName=group,
            ModelPackageGroupDescription="SageMaker Auto Research candidates")


def register(cfg: dict, sm_client, model: dict) -> dict:
    group = cfg["model_registry"]["package_group"]
    _ensure_group(sm_client, group)
    resp = sm_client.create_model_package(
        ModelPackageGroupName=group,
        ModelApprovalStatus="PendingManualApproval",
        InferenceSpecification={
            "Containers": [{"Image": model["image_uri"],
                            "ModelDataUrl": model["model_data_s3"]}],
            "SupportedContentTypes": ["text/csv"],
            "SupportedResponseMIMETypes": ["text/csv"],
        },
        CustomerMetadataProperties={
            model["metric_name"]: str(model["metric_value"]),
        },
    )
    return {"model_package_arn": resp["ModelPackageArn"], "package_group": group}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-json", required=True)
    args = parser.parse_args()
    try:
        cfg = config.load(args.config)
        model = json.loads(args.model_json)
        session = boto3.Session(profile_name=cfg["aws"]["profile"],
                                region_name=cfg["aws"]["region"])
        sm_client = session.client("sagemaker")
        contract.emit(contract.ok(register(cfg, sm_client, model)))
    except Exception as e:  # noqa: BLE001
        contract.emit(contract.err(str(e)))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_register_model.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/sm_research/register_model.py tests/test_register_model.py
git commit -m "feat: SageMaker Model Registry registration"
```

---

### Task 9: Pure loop-decision functions (JS)

**Files:**
- Create: `workflows/lib/decisions.js`
- Test: `workflows/lib/decisions.test.js`

**Interfaces:**
- Produces (CommonJS exports):
  - `rankExperiments(results, metricGoal)` → results sorted best-first by `.metric` (`"maximize"` desc, `"minimize"` asc), dropping entries with null/undefined `.metric`.
  - `bestMetric(results, metricGoal)` → the best `.metric` value, or `null` if none.
  - `goalMet(best, threshold, metricGoal)` → bool.
  - `budgetExceeded(state, budget)` → `{exceeded: bool, reason: string|null}` checking `state.round >= budget.max_rounds`, `state.costUsd >= budget.max_total_cost_usd`, `state.elapsedHours >= budget.max_wallclock_hours`.
  - `plateaued(history, plateau, metricGoal)` → bool; `history` is an array of per-round best metrics; true when the last `patience_rounds` rounds improved by less than `min_delta` total.

- [ ] **Step 1: Write the failing test**

`workflows/lib/decisions.test.js`:
```javascript
const assert = require("node:assert");
const test = require("node:test");
const d = require("./decisions.js");

test("rankExperiments maximize drops nulls, sorts desc", () => {
  const r = d.rankExperiments(
    [{ metric: 0.5 }, { metric: null }, { metric: 0.9 }, { metric: 0.7 }],
    "maximize"
  );
  assert.deepEqual(r.map((x) => x.metric), [0.9, 0.7, 0.5]);
});

test("rankExperiments minimize sorts asc", () => {
  const r = d.rankExperiments([{ metric: 0.5 }, { metric: 0.1 }], "minimize");
  assert.deepEqual(r.map((x) => x.metric), [0.1, 0.5]);
});

test("bestMetric returns null when no valid metrics", () => {
  assert.equal(d.bestMetric([{ metric: null }], "maximize"), null);
});

test("goalMet maximize", () => {
  assert.equal(d.goalMet(0.91, 0.9, "maximize"), true);
  assert.equal(d.goalMet(0.89, 0.9, "maximize"), false);
});

test("goalMet minimize", () => {
  assert.equal(d.goalMet(0.08, 0.1, "minimize"), true);
});

test("budgetExceeded on rounds", () => {
  const out = d.budgetExceeded(
    { round: 5, costUsd: 0, elapsedHours: 0 },
    { max_rounds: 5, max_total_cost_usd: 100, max_wallclock_hours: 6 }
  );
  assert.equal(out.exceeded, true);
  assert.match(out.reason, /round/);
});

test("budgetExceeded on cost", () => {
  const out = d.budgetExceeded(
    { round: 1, costUsd: 150, elapsedHours: 0 },
    { max_rounds: 5, max_total_cost_usd: 100, max_wallclock_hours: 6 }
  );
  assert.equal(out.exceeded, true);
  assert.match(out.reason, /cost/);
});

test("budgetExceeded false when under all caps", () => {
  const out = d.budgetExceeded(
    { round: 1, costUsd: 1, elapsedHours: 1 },
    { max_rounds: 5, max_total_cost_usd: 100, max_wallclock_hours: 6 }
  );
  assert.equal(out.exceeded, false);
  assert.equal(out.reason, null);
});

test("plateaued true when gains below min_delta over patience window", () => {
  // last 2 rounds: 0.80 -> 0.802, total gain 0.002 < 0.005
  assert.equal(
    d.plateaued([0.70, 0.80, 0.802], { patience_rounds: 2, min_delta: 0.005 }, "maximize"),
    true
  );
});

test("plateaued false when still improving", () => {
  assert.equal(
    d.plateaued([0.70, 0.80, 0.90], { patience_rounds: 2, min_delta: 0.005 }, "maximize"),
    false
  );
});

test("plateaued false before enough history", () => {
  assert.equal(
    d.plateaued([0.80], { patience_rounds: 2, min_delta: 0.005 }, "maximize"),
    false
  );
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node --test workflows/lib/`
Expected: FAIL — `Cannot find module './decisions.js'`.

- [ ] **Step 3: Implement decisions.js**

`workflows/lib/decisions.js`:
```javascript
"use strict";

function _valid(results) {
  return results.filter((r) => r && typeof r.metric === "number");
}

function rankExperiments(results, metricGoal) {
  const valid = _valid(results);
  const dir = metricGoal === "minimize" ? 1 : -1;
  return valid.slice().sort((a, b) => dir * (a.metric - b.metric));
}

function bestMetric(results, metricGoal) {
  const ranked = rankExperiments(results, metricGoal);
  return ranked.length ? ranked[0].metric : null;
}

function goalMet(best, threshold, metricGoal) {
  if (best === null || best === undefined) return false;
  return metricGoal === "minimize" ? best <= threshold : best >= threshold;
}

function budgetExceeded(state, budget) {
  if (state.round >= budget.max_rounds) {
    return { exceeded: true, reason: `max_rounds reached (${budget.max_rounds})` };
  }
  if (state.costUsd >= budget.max_total_cost_usd) {
    return { exceeded: true, reason: `max cost reached ($${state.costUsd})` };
  }
  if (state.elapsedHours >= budget.max_wallclock_hours) {
    return { exceeded: true, reason: `max wallclock reached (${state.elapsedHours}h)` };
  }
  return { exceeded: false, reason: null };
}

function plateaued(history, plateau, metricGoal) {
  const n = plateau.patience_rounds;
  if (history.length < n + 1) return false;
  const window = history.slice(-(n + 1));
  const first = window[0];
  const last = window[window.length - 1];
  const gain = metricGoal === "minimize" ? first - last : last - first;
  return gain < plateau.min_delta;
}

module.exports = { rankExperiments, bestMetric, goalMet, budgetExceeded, plateaued };
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `node --test workflows/lib/`
Expected: PASS (all tests pass).

- [ ] **Step 5: Commit**

```bash
git add workflows/lib/decisions.js workflows/lib/decisions.test.js
git commit -m "feat: pure loop-decision functions (rank/goal/budget/plateau)"
```

---

### Task 10: Workflow engine

**Files:**
- Create: `workflows/auto-research-loop.js`

**Interfaces:**
- Consumes: `workflows/lib/decisions.js` — but note Workflow scripts run in a restricted context (no `require` of arbitrary files is guaranteed). Therefore this task INLINES the decision functions verbatim from Task 9 into the workflow's own scope (they are pure and small). The `decisions.js` module remains the tested source of truth; the inlined copy MUST be byte-identical in logic.
- Consumes at runtime: `args` = `{ configPath, runId, discovery }` passed by the skill.
- Produces: a Workflow returning `{ best: {metric, experiment}, rounds: [...], stopReason, winnerRegistered: bool }`.

- [ ] **Step 1: Write the workflow script**

`workflows/auto-research-loop.js`:
```javascript
export const meta = {
  name: "sagemaker-auto-research-loop",
  description: "Iterative SageMaker experimentation loop: fan out experiments, judge, check stop conditions until goal met or budget/plateau reached.",
  phases: [
    { title: "Experiment" },
    { title: "Judge" },
  ],
};

// --- Inlined pure decisions (source of truth: workflows/lib/decisions.js, unit-tested) ---
function _valid(results) { return results.filter((r) => r && typeof r.metric === "number"); }
function rankExperiments(results, metricGoal) {
  const valid = _valid(results);
  const dir = metricGoal === "minimize" ? 1 : -1;
  return valid.slice().sort((a, b) => dir * (a.metric - b.metric));
}
function bestMetric(results, metricGoal) {
  const ranked = rankExperiments(results, metricGoal);
  return ranked.length ? ranked[0].metric : null;
}
function goalMet(best, threshold, metricGoal) {
  if (best === null || best === undefined) return false;
  return metricGoal === "minimize" ? best <= threshold : best >= threshold;
}
function budgetExceeded(state, budget) {
  if (state.round >= budget.max_rounds) return { exceeded: true, reason: `max_rounds reached (${budget.max_rounds})` };
  if (state.costUsd >= budget.max_total_cost_usd) return { exceeded: true, reason: `max cost reached ($${state.costUsd})` };
  if (state.elapsedHours >= budget.max_wallclock_hours) return { exceeded: true, reason: `max wallclock reached (${state.elapsedHours}h)` };
  return { exceeded: false, reason: null };
}
function plateaued(history, plateau, metricGoal) {
  const n = plateau.patience_rounds;
  if (history.length < n + 1) return false;
  const window = history.slice(-(n + 1));
  const gain = metricGoal === "minimize" ? window[0] - window[window.length - 1] : window[window.length - 1] - window[0];
  return gain < plateau.min_delta;
}
// --- end inlined ---

const EXPERIMENT_SCHEMA = {
  type: "object",
  required: ["metric", "status"],
  properties: {
    metric: { type: ["number", "null"] },
    status: { type: "string" },
    hypothesis: { type: "string" },
    mlflow_run_id: { type: "string" },
    model_data_s3: { type: "string" },
    image_uri: { type: "string" },
    cost_usd: { type: "number" },
    failure_reason: { type: ["string", "null"] },
  },
};

const cfg = args.config;               // parsed research.yaml (skill passes it in)
const budget = cfg.budget;
const metricGoal = cfg.problem.metric_goal;
const threshold = cfg.problem.threshold;
const metric = cfg.problem.success_metric;

const history = [];        // best metric per round
const rounds = [];
let cumulativeCost = 0;
let best = { metric: null, experiment: null };
let stopReason = null;

for (let round = 0; round < budget.max_rounds; round++) {
  const pre = budgetExceeded(
    { round, costUsd: cumulativeCost, elapsedHours: 0 }, budget);
  if (pre.exceeded) { stopReason = pre.reason; break; }

  phase("Experiment");
  log(`Round ${round + 1}/${budget.max_rounds}: launching ${budget.experiments_per_round} experiments`);

  const priorSummary = rounds.length
    ? `Prior rounds best=${best.metric}. What worked: ${JSON.stringify(rounds.at(-1).topHypotheses || [])}.`
    : "First round — establish a baseline and diversify hypotheses.";

  const experiments = await parallel(
    Array.from({ length: budget.experiments_per_round }, (_, i) => () =>
      agent(
        `You are experiment ${i + 1} in round ${round + 1} of a SageMaker auto-research loop.\n` +
        `Goal: ${metricGoal} ${metric} to >= ${threshold} (problem: ${cfg.problem.type}).\n` +
        `${priorSummary}\n` +
        `Config path: ${args.configPath}. Run id: ${args.runId}. Round: ${round + 1}.\n` +
        `Allowed data (DO NOT exceed): ${JSON.stringify(cfg.data.allowed)}; target=${cfg.data.target}.\n\n` +
        `Follow the sagemaker-auto-research skill's "Run one experiment" section EXACTLY:\n` +
        `1. Choose a distinct hypothesis (feature subset/mix, algorithm, hyperparameters).\n` +
        `2. Build a data-slice select_sql over allowed data and call redshift_unload.py (cached by hash).\n` +
        `3. Author a training script from templates/train and launch it via sm_launch.py.\n` +
        `4. Log params/metrics to MLflow experiment '${cfg.sagemaker.mlflow.experiment_name}'.\n` +
        `5. Return the test-split ${metric} as 'metric' (null if the job failed), plus status, ` +
        `hypothesis, mlflow_run_id, model_data_s3, image_uri, cost_usd, failure_reason.`,
        { label: `exp r${round + 1}#${i + 1}`, phase: "Experiment", schema: EXPERIMENT_SCHEMA }
      )
    )
  );

  phase("Judge");
  const survivors = experiments.filter(Boolean);
  if (survivors.length === 0) {
    stopReason = `all experiments failed in round ${round + 1}`;
    rounds.push({ round: round + 1, results: [], roundBest: null });
    break;
  }

  const ranked = rankExperiments(survivors, metricGoal);
  const roundCost = survivors.reduce((s, e) => s + (e.cost_usd || 0), 0);
  cumulativeCost += roundCost;
  const roundBest = bestMetric(survivors, metricGoal);
  history.push(roundBest);

  if (roundBest !== null &&
      (best.metric === null || goalMet(roundBest, best.metric, metricGoal) || _better(roundBest, best.metric, metricGoal))) {
    best = { metric: roundBest, experiment: ranked[0] };
  }

  rounds.push({
    round: round + 1,
    roundBest,
    roundCost,
    cumulativeCost,
    topHypotheses: ranked.slice(0, 2).map((e) => e.hypothesis),
    results: survivors,
  });
  log(`Round ${round + 1} best ${metric}=${roundBest}; cumulative cost ~$${cumulativeCost.toFixed(2)}`);

  if (goalMet(roundBest, threshold, metricGoal)) { stopReason = "goal met"; break; }
  const post = budgetExceeded({ round: round + 1, costUsd: cumulativeCost, elapsedHours: 0 }, budget);
  if (post.exceeded) { stopReason = post.reason; break; }
  if (plateaued(history, budget.plateau, metricGoal)) { stopReason = "plateau detected"; break; }
}

if (!stopReason) stopReason = "max_rounds reached";

return { best, rounds, stopReason, metric, threshold, metricGoal };

function _better(candidate, current, metricGoal) {
  return metricGoal === "minimize" ? candidate < current : candidate > current;
}
```

- [ ] **Step 2: Syntax-check the workflow script**

Run: `node --check workflows/auto-research-loop.js`
Expected: exits 0 with no output (valid syntax). Note: this only validates syntax; the workflow executes via the Workflow tool during real runs.

- [ ] **Step 3: Verify inlined decisions match the tested module**

Run: `node --test workflows/lib/`
Expected: PASS. Manually confirm the five inlined functions in `auto-research-loop.js` are logically identical to `workflows/lib/decisions.js`. If they diverge, fix the workflow copy.

- [ ] **Step 4: Commit**

```bash
git add workflows/auto-research-loop.js
git commit -m "feat: auto-research Workflow engine (fan-out experiments + judge + stop checks)"
```

---

### Task 11: Templates (research.yaml + training scripts)

**Files:**
- Create: `templates/research.yaml.tmpl`
- Create: `templates/train/xgboost_train.py`
- Create: `templates/train/sklearn_train.py`
- Test: `tests/test_templates.py`

**Interfaces:**
- Produces: `templates/research.yaml.tmpl` that MUST parse as valid YAML and satisfy `config.validate` once placeholders are filled with the sample values used in the test. Training templates are starting points the agent copies and edits per experiment; they MUST be syntactically valid Python.

- [ ] **Step 1: Write the failing test**

`tests/test_templates.py`:
```python
import ast
import pathlib

import yaml

from sm_research import config

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_research_template_parses_and_validates():
    raw = (ROOT / "templates/research.yaml.tmpl").read_text()
    cfg = yaml.safe_load(raw)
    # Template ships with working sample values, so it must validate clean.
    assert config.validate(cfg) == []


def test_training_templates_are_valid_python():
    for name in ("xgboost_train.py", "sklearn_train.py"):
        src = (ROOT / "templates/train" / name).read_text()
        ast.parse(src)  # raises SyntaxError if invalid
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_templates.py -v`
Expected: FAIL — `FileNotFoundError` for the template files.

- [ ] **Step 3: Create the research.yaml template**

`templates/research.yaml.tmpl`:
```yaml
# SageMaker Auto Research config. Edit values, then run the loop.
aws:
  profile: my-profile          # AWS profile to use (never the default profile)
  region: us-east-1

data:
  redshift:
    workgroup_or_cluster: analytics-wg
    database: analytics
    serverless: true           # true = Redshift Serverless workgroup; false = provisioned cluster
    secret_arn: ""             # optional Secrets Manager ARN for Data API auth
  allowed:                     # ALLOWLIST — the only data the agent may use
    - schema: public
      table: customer_events
      columns: [age, tenure, region, monthly_spend]
  target: churned              # label column
problem:
  type: binary_classification  # binary_classification | multiclass | regression
  success_metric: f1
  threshold: 0.9
  metric_goal: maximize        # maximize | minimize
budget:
  max_rounds: 5
  experiments_per_round: 4
  max_total_cost_usd: 100
  max_wallclock_hours: 6
  plateau:
    patience_rounds: 2
    min_delta: 0.005
sagemaker:
  role_arn: arn:aws:iam::123456789012:role/SageMakerExecutionRole
  s3_bucket: amzn-s3-demo-sagemaker-bucket
  instance_type: ml.m5.xlarge
  mlflow:
    tracking_server_arn: ""    # blank => discover; if none exists you'll be asked to create one
    experiment_name: auto-research
model_registry:
  package_group: auto-research-models
```

- [ ] **Step 4: Create the XGBoost training template**

`templates/train/xgboost_train.py`:
```python
"""Starter XGBoost training script for SageMaker. Agent edits per experiment.

Reads train/validation/test parquet from the SageMaker `training` channel,
trains XGBoost, prints the objective metric in a regex-parseable form, and
saves the model to the SageMaker model dir.
"""
import argparse
import glob
import os

import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score


def _load(split_dir):
    files = glob.glob(os.path.join(split_dir, "*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--eta", type=float, default=0.3)
    p.add_argument("--num-round", type=int, default=100)
    p.add_argument("--target", default=os.environ.get("SM_HP_TARGET", "target"))
    args = p.parse_args()

    base = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
    train = _load(os.path.join(base, "train"))
    test = _load(os.path.join(base, "test"))

    y_train = train.pop(args.target)
    y_test = test.pop(args.target)
    dtrain = xgb.DMatrix(train, label=y_train)
    dtest = xgb.DMatrix(test)

    params = {"max_depth": args.max_depth, "eta": args.eta,
              "objective": "binary:logistic"}
    model = xgb.train(params, dtrain, num_boost_round=args.num_round)

    preds = (model.predict(dtest) >= 0.5).astype(int)
    f1 = f1_score(y_test, preds)
    # SageMaker MetricDefinitions regex parses this line: e.g. "f1=(.*?);"
    print(f"f1={f1};")

    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    model.save_model(os.path.join(model_dir, "xgboost-model.json"))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Create the sklearn training template**

`templates/train/sklearn_train.py`:
```python
"""Starter scikit-learn training script for SageMaker. Agent edits per experiment."""
import argparse
import glob
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def _load(split_dir):
    files = glob.glob(os.path.join(split_dir, "*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--target", default=os.environ.get("SM_HP_TARGET", "target"))
    args = p.parse_args()

    base = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
    train = _load(os.path.join(base, "train"))
    test = _load(os.path.join(base, "test"))

    y_train = train.pop(args.target)
    y_test = test.pop(args.target)

    clf = RandomForestClassifier(n_estimators=args.n_estimators,
                                 max_depth=args.max_depth, n_jobs=-1)
    clf.fit(train, y_train)

    preds = clf.predict(test)
    f1 = f1_score(y_test, preds)
    print(f"f1={f1};")

    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    joblib.dump(clf, os.path.join(model_dir, "model.joblib"))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_templates.py -v`
Expected: PASS (2 passed). (The test only parses YAML and Python ASTs; it does not import xgboost/sklearn.)

- [ ] **Step 7: Commit**

```bash
git add templates/research.yaml.tmpl templates/train/xgboost_train.py templates/train/sklearn_train.py tests/test_templates.py
git commit -m "feat: research.yaml + training-script templates"
```

---

### Task 12: The orchestration skill

**Files:**
- Create: `skills/sagemaker-auto-research/SKILL.md`

**Interfaces:**
- Consumes: all scripts (Tasks 4-8), templates (Task 11), and the workflow (Task 10).
- Produces: the human/agent-readable contract that ties them together. No code; prose + exact commands.

- [ ] **Step 1: Write the skill file**

`skills/sagemaker-auto-research/SKILL.md`:
````markdown
---
name: SageMaker Auto Research
description: Use when the user wants to run an autonomous ML experimentation loop on Amazon SageMaker — from Redshift data through feature engineering, training, MLflow tracking, and model registration — until a model meets a success metric (e.g. F1 >= 0.9). Triggers on "auto research", "run experiments until", "find a model that hits", or an existing research.yaml.
---

# SageMaker Auto Research

Orchestration contract for the autonomous ML loop. You are the "brain": validate
config, discover data, ensure MLflow, launch the Workflow, and register the winner.
The Workflow (`workflows/auto-research-loop.js`) is the engine; the Python scripts
under `${CLAUDE_PLUGIN_ROOT}/scripts` are your AWS tools.

## Hard rules

- Use the AWS profile and region from `research.yaml` for EVERY AWS call. Never the default profile.
- Only touch Redshift data listed in `data.allowed`. Never query beyond it.
- Prefer read-only operations. NEVER delete AWS resources.
- Before creating ANY billable resource (only the MLflow tracking server here),
  state the estimated cost and get explicit user confirmation.
- Respect `budget` caps as hard ceilings.

## Environment

Run all scripts with `uv` from the plugin root so imports resolve:
```bash
cd ${CLAUDE_PLUGIN_ROOT} && uv run python -m sm_research.<module> ...
```

## Workflow

### 0. Config
If no `research.yaml` exists in the working directory, tell the user to run
`/sm-research` first (it scaffolds one). Load and sanity-check it:
```bash
cd ${CLAUDE_PLUGIN_ROOT} && uv run python -c "from sm_research import config; import sys; errs=config.validate(config.load('$CFG')); print('\n'.join(errs) or 'OK')"
```
`$CFG` is the absolute path to the user's research.yaml. Stop on any errors.

### 1. Discover data (read-only, Redshift Data API)
```bash
cd ${CLAUDE_PLUGIN_ROOT} && uv run python -m sm_research.redshift_discover --config "$CFG" --sample-rows 10
```
Review row counts, columns, and target class balance. If the target is severely
imbalanced, note it — experiments should account for it.

### 2. Ensure MLflow tracking server
```bash
cd ${CLAUDE_PLUGIN_ROOT} && uv run python -m sm_research.mlflow_ensure --config "$CFG"
```
If the result has `needs_confirmation: true`, tell the user a new SageMaker MLflow
tracking server will be created (roughly $0.60–$1.00/hour while running, plus S3
artifact storage) and ask for explicit approval. Only on approval, re-run with
`--confirm-create`. Write the returned ARN back into `research.yaml`
(`sagemaker.mlflow.tracking_server_arn`) so it is reused.

### 3. Launch the loop (Workflow)
Generate a `runId` (e.g. from the current timestamp — ask the harness for one; do
not fabricate). Invoke the Workflow tool with `scriptPath`
`${CLAUDE_PLUGIN_ROOT}/workflows/auto-research-loop.js` and
`args: { configPath: "$CFG", runId: "<id>", config: <parsed research.yaml as JSON> }`.

The Workflow runs rounds of experiments and returns
`{ best, rounds, stopReason, ... }`. While it runs, each experiment subagent
follows "Run one experiment" below.

### 4. Register the winner
If `best.metric` meets the threshold, register it:
```bash
cd ${CLAUDE_PLUGIN_ROOT} && uv run python -m sm_research.register_model --config "$CFG" \
  --model-json '{"image_uri":"<img>","model_data_s3":"<s3>","metric_name":"<metric>","metric_value":<value>}'
```
If the loop stopped without meeting the goal (budget/plateau/all-failed), report
the best result, the `stopReason`, and concrete suggestions (more rounds, wider
hypotheses, different features), and ask whether to continue.

## Run one experiment (executed by each Workflow subagent)

1. **Pick a distinct hypothesis** — a feature subset/mix from `data.allowed`, an
   algorithm (xgboost or sklearn template), and hyperparameters. Diverge from
   hypotheses listed as already-tried in the prompt.
2. **Materialize the data slice.** Build a `select_sql` over allowed columns/tables
   (joins/filters allowed within the allowlist). UNLOAD it (cached by content hash):
   ```bash
   cd ${CLAUDE_PLUGIN_ROOT} && uv run python -m sm_research.redshift_unload --config "$CFG" --run-id "<id>" \
     --spec-json '{"tables":[...],"columns":[...],"filter":null,"seed":42,"select_sql":"<SQL>"}'
   ```
3. **Feature engineering + splits.** Produce train/val/test parquet under the slice
   prefix (a SageMaker Processing job for heavy work, or inline for small data).
   Keep the test split deterministic (fixed seed) so rounds are comparable.
4. **Author the training script** from `${CLAUDE_PLUGIN_ROOT}/templates/train/` and
   launch it, passing a `metric_regex` that matches the script's printed metric
   (e.g. `f1=(.*?);`):
   ```bash
   cd ${CLAUDE_PLUGIN_ROOT} && uv run python -m sm_research.sm_launch --config "$CFG" \
     --job-json '{"job_name":"<unique>","image_uri":"<xgb-or-sklearn-image>","input_s3":"<slice>/","output_s3":"s3://<bucket>/sm-research/<id>/out/","hyperparameters":{...},"metric_regex":"f1=(.*?);","instance_type":"ml.m5.xlarge"}'
   ```
5. **Log to MLflow** (params, metrics, artifacts) under the configured experiment name.
6. **Return** the held-out test metric as `metric` (null if the job failed), plus
   `status`, `hypothesis`, `mlflow_run_id`, `model_data_s3`, `image_uri`,
   `cost_usd`, and `failure_reason`.

## Failure handling

- Any script returning `{"status":"error"}` — read `error`, fix inputs, retry once;
  if it persists, surface it to the user verbatim.
- A failed training job — `sm_launch` returns the status and `failure_reason`; return
  `metric: null` so the judge drops it. Adjust the hypothesis next round.
- If every experiment in a round fails, the Workflow stops. Investigate the common
  cause (IAM, image, data) before rerunning.

## Dry run

For `--dry-run` (see the `/sm-research` command): do steps 0–1 and a single tiny
experiment on a `LIMIT`-sampled slice with `experiments_per_round=1, max_rounds=1`,
then report — so the user validates the pipeline cheaply before a full run.
````

- [ ] **Step 2: Verify the skill frontmatter parses**

Run: `cd ${CLAUDE_PLUGIN_ROOT:-.} && uv run python -c "import pathlib,yaml; t=pathlib.Path('skills/sagemaker-auto-research/SKILL.md').read_text(); fm=t.split('---')[1]; d=yaml.safe_load(fm); assert d['name'] and d['description']; print('OK')"`
Expected: prints `OK`.

- [ ] **Step 3: Commit**

```bash
git add skills/sagemaker-auto-research/SKILL.md
git commit -m "feat: sagemaker-auto-research orchestration skill"
```

---

### Task 13: The /sm-research command

**Files:**
- Create: `commands/sm-research.md`

**Interfaces:**
- Consumes: `templates/research.yaml.tmpl`, `config.validate`, and the skill (Task 12).
- Produces: the `/sm-research` slash command that interviews the user, writes `research.yaml`, and hands off to the skill. Supports a `--dry-run` argument.

- [ ] **Step 1: Write the command file**

`commands/sm-research.md`:
```markdown
---
name: sm-research
description: Set up and launch a SageMaker auto-research run — interview for goal/data/criteria, scaffold research.yaml, then run the loop until a model meets the goal or a guardrail stops it.
argument-hint: "[--dry-run]"
---

# /sm-research

You are setting up a SageMaker Auto Research run. Arguments: `$ARGUMENTS`
(if it contains `--dry-run`, do a cheap validation run — see the skill's "Dry run").

## Step 1 — Prerequisites
Confirm the user has: an AWS profile with SageMaker + Redshift Data API + S3 access,
a SageMaker execution role ARN, an S3 bucket, and Redshift data they're allowed to use.
Per global rules, ALWAYS ask which AWS profile to use; never assume the default.

## Step 2 — Interview (one topic at a time)
Read the scaffold at `${CLAUDE_PLUGIN_ROOT}/templates/research.yaml.tmpl` and fill it
in with the user. Collect, asking only for what you don't already know:
- AWS profile + region
- Redshift: workgroup/cluster, database, serverless?, optional secret ARN
- Allowed data: schema.table(s) and columns (the allowlist), and the target/label column
- Problem type, success metric, threshold, and whether to maximize or minimize it
- Budget: max rounds, experiments per round, max cost (USD), max wallclock hours, plateau patience + min delta
- SageMaker: execution role ARN, S3 bucket, instance type
- MLflow: existing tracking server ARN if any (else leave blank), experiment name
- Model registry: package group name

## Step 3 — Write research.yaml
Write the completed config to `./research.yaml` in the user's working directory
(NOT into the plugin). Then validate it:
```bash
cd ${CLAUDE_PLUGIN_ROOT} && uv run python -c "from sm_research import config; print('\n'.join(config.validate(config.load('$PWD/research.yaml'))) or 'VALID')"
```
Fix any reported errors with the user before continuing.

## Step 4 — Hand off to the loop
Invoke the **SageMaker Auto Research** skill to run the loop (or, with `--dry-run`,
the dry-run path). Set `$CFG` to the absolute path of the written `research.yaml`.

Show the user the final result: best metric, whether the goal was met, the
registered model package ARN (if any), the stop reason, and per-round summary.
```

- [ ] **Step 2: Verify the command frontmatter parses**

Run: `cd ${CLAUDE_PLUGIN_ROOT:-.} && uv run python -c "import pathlib,yaml; t=pathlib.Path('commands/sm-research.md').read_text(); d=yaml.safe_load(t.split('---')[1]); assert d['name']=='sm-research'; print('OK')"`
Expected: prints `OK`.

- [ ] **Step 3: Commit**

```bash
git add commands/sm-research.md
git commit -m "feat: /sm-research command (interview + scaffold + launch)"
```

---

### Task 14: Full suite green + README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Run the full Python suite**

Run: `uv run pytest -v`
Expected: all tests from Tasks 1-11 pass (0 failures).

- [ ] **Step 2: Run the full JS suite**

Run: `node --test workflows/lib/`
Expected: all decision tests pass.

- [ ] **Step 3: Syntax-check the workflow**

Run: `node --check workflows/auto-research-loop.js`
Expected: exits 0.

- [ ] **Step 4: Write the README**

`README.md`:
```markdown
# SageMaker Auto Research

A Claude Code plugin that runs autonomous ML experimentation loops on Amazon
SageMaker. Give it a goal (e.g. F1 >= 0.9), an allowlist of Redshift data, and
budget guardrails; it iterates — discover data, engineer features, train models,
track experiments in MLflow — until a candidate model meets the goal and is
registered in the SageMaker Model Registry, or a budget/plateau guardrail stops it.

## Prerequisites
- An AWS profile with access to SageMaker, the Redshift Data API, and S3
- A SageMaker execution role, an S3 bucket, and Redshift data you may use
- `uv` (Python 3.12+) and Node.js 22+

## Usage
1. Run `/sm-research` (add `--dry-run` for a cheap end-to-end validation first).
2. Answer the interview; a reproducible `research.yaml` is written to your working directory.
3. The loop runs: rounds of parallel experiments, a judge stage ranks them, and
   stop conditions (goal met / budget / plateau) decide when to stop.
4. The winning model is registered in the SageMaker Model Registry.

## Guardrails
- Only data in `research.yaml`'s `data.allowed` is ever queried.
- Billable resource creation (the MLflow server) requires explicit confirmation.
- No AWS resources are ever deleted. Budget caps are hard ceilings.

## Development
```bash
uv sync
uv run pytest -v          # Python scripts
node --test workflows/lib/  # loop-decision logic
```
```

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: add README"
```

---

## Self-Review

**Spec coverage:**
- Redshift discovery (Data API) → Task 4. UNLOAD to S3 → Task 5. ✓
- Feature engineering + custom training on SageMaker → Task 7 + templates (Task 11) + skill step 3 (Task 12). ✓
- MLflow (create-on-the-fly, confirmation-gated) → Task 6 + skill step 2. ✓
- Model Registry registration on goal → Task 8 + skill step 4. ✓
- Workflow orchestration with fan-out + judge → Task 10. ✓
- Stop conditions (goal/budget/plateau) → Task 9 (tested) + Task 10 (applied). ✓
- Config: command scaffolds YAML → Task 13 + template (Task 11); validation → Task 2. ✓
- Data-slice caching, consistent test split → Task 5 + skill step 3. ✓
- Error handling (structured envelope, failed-experiment survival, all-fail stop) → Task 1 contract, Task 10 filtering, skill failure section. ✓
- Cost/production safety (profile from config, no delete, confirm create) → Global Constraints + skill hard rules. ✓
- Testing (mocked boto3, pure decision fns, dry-run) → per-task tests + Task 9 + skill dry-run. ✓

**Placeholder scan:** No TBD/TODO. Every code step shows complete code; every command has expected output.

**Type consistency:** Script envelope `{status,data,error}` consistent across Tasks 1-8. `slice_hash`/`s3_prefix` names consistent (Task 3 → Task 5). Decision fn names (`rankExperiments`, `bestMetric`, `goalMet`, `budgetExceeded`, `plateaued`) consistent Task 9 → inlined Task 10. Experiment return fields (`metric`, `status`, `hypothesis`, `mlflow_run_id`, `model_data_s3`, `image_uri`, `cost_usd`, `failure_reason`) consistent between the workflow schema (Task 10) and the skill's "Run one experiment" (Task 12).

**Known simplification:** decision functions are inlined into the workflow (Task 10) rather than `require`d, because Workflow scripts run in a restricted context. `workflows/lib/decisions.js` remains the tested source of truth; Task 10 Step 3 checks the copy matches.
