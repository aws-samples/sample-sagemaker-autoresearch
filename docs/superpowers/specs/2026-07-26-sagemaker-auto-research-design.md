# SageMaker Auto Research — Design

**Date:** 2026-07-26
**Status:** Approved (design), pending implementation plan

## Overview

A Claude Code plugin, **SageMaker Auto Research**, that runs autonomous ML
experimentation loops on Amazon SageMaker. Given a goal (e.g. "binary
classifier with F1 >= 0.9"), an allowlist of Redshift data, and budget
guardrails, it iterates — discover data, engineer features, train models,
track experiments — until a candidate model meets the success criterion and is
registered in the SageMaker Model Registry, or a budget/plateau stop condition
is hit.

Auto research is a variant of loop engineering: start with a goal and let the
agent keep working until it hits the goal (or a guardrail).

## Design Decisions (locked)

| Area | Decision |
|---|---|
| Loop engine | **Workflow orchestration** — a deterministic Workflow script fans out experiments across subagents with a judge stage that ranks candidates. |
| Training | **Custom training scripts** authored by the agent (xgboost/sklearn/pytorch starters), launched as SageMaker Training Jobs. |
| Data ingest | **Both** — Redshift Data API for discovery/sampling/validation; UNLOAD to S3 (Parquet) for training datasets. |
| MLflow | **Create on the fly via SDK** when missing — gated behind explicit user confirmation with cost stated first. |
| Config | **Both** — `/sm-research` interviews the user and scaffolds a reproducible `research.yaml` that drives the workflow. |
| Stop conditions | **Goal met**, **budget caps**, and **plateau detection** (fully autonomous within guardrails). |

## Architecture — Approach A (Workflow-centric plugin)

- **Command** (`/sm-research`) = setup: interviews the user, writes `research.yaml`.
- **Skill** (`sagemaker-auto-research`) = orchestration contract / the "brain":
  validates config, runs Redshift discovery, ensures MLflow, launches the
  workflow, checks stop conditions between rounds, registers the winner.
- **Workflow** (`auto-research-loop.js`) = the engine: per-round fan-out of
  experiments + judge stage.
- **Scripts** = thin, testable AWS wrappers (Data API, UNLOAD, MLflow, SageMaker
  jobs), all using the user's AWS profile + region, read-only where possible.

### Plugin structure

```
sagemaker-auto-research/
├── .claude-plugin/plugin.json         manifest
├── commands/
│   └── sm-research.md                 /sm-research — interview → writes research.yaml
├── skills/
│   └── sagemaker-auto-research/
│       └── SKILL.md                   orchestration contract (the "brain")
├── workflows/
│   └── auto-research-loop.js           Workflow engine (fan-out + judge + stop checks)
├── scripts/
│   ├── redshift_discover.py            Data API: list schemas, sample, validate selection
│   ├── redshift_unload.py              UNLOAD chosen dataset → S3 parquet
│   ├── mlflow_ensure.py                find-or-(confirm)-create MLflow tracking server
│   └── sm_launch.py                    launch Processing/Training jobs, poll, return metrics
└── templates/
    ├── research.yaml.tmpl              config scaffold
    └── train/                          starter training-script templates (xgboost/sklearn/pytorch)
```

Each unit has one job: command collects intent, skill decides/orchestrates,
workflow runs the deterministic loop, scripts wrap AWS.

## Configuration — `research.yaml`

The reproducible artifact scaffolded by `/sm-research` and consumed by the
workflow.

```yaml
aws:
  profile: my-profile          # inherited; prompted, never defaulted
  region: us-east-1

data:
  redshift:
    workgroup_or_cluster: analytics-wg
    database: analytics
    secret_arn: arn:aws:secretsmanager:...   # or Data API auth
  allowed:                      # user-specified allowlist — the ONLY data usable
    - schema: public
      table: customer_events
      columns: [age, tenure, region, ...]    # or "*"
  target: churned              # label column
  # agent is free to select/mix from `allowed` — never beyond it

problem:
  type: binary_classification  # | multiclass | regression
  success_metric: f1           # metric name
  threshold: 0.9               # >= for goal-met
  metric_goal: maximize

budget:
  max_rounds: 5
  experiments_per_round: 4
  max_total_cost_usd: 100
  max_wallclock_hours: 6
  plateau:
    patience_rounds: 2         # stop if no improvement...
    min_delta: 0.005           # ...of at least this much

sagemaker:
  role_arn: arn:aws:iam::...:role/SageMakerExecutionRole
  s3_bucket: amzn-s3-demo-sagemaker-bucket
  instance_type: ml.m5.xlarge
  mlflow:
    tracking_server_arn: ""    # blank → discover; if none, confirm-and-create
    experiment_name: churn-auto-research

model_registry:
  package_group: churn-models
```

**Guardrails baked in:** `data.allowed` strictly bounds what the agent may
touch; `budget` bounds spend/time. Anything needed but absent becomes a prompt
during `/sm-research`.

## Data flow (one round)

1. **[skill] Validate config + Redshift discovery** via Data API — schemas, row
   counts, sample rows, null/skew stats, target class balance. Confirms the
   allowlist is queryable.
2. **[skill] Ensure MLflow server** — discover; if none exists, CONFIRM with the
   user (cost stated) then create via SDK.
3. **[workflow] Round `r`: fan out `experiments_per_round` subagents.** Each
   experiment subagent:
   1. picks a hypothesis (feature subset/mix, algorithm, hyperparameters),
   2. renders UNLOAD SQL over allowed data → S3 parquet (idempotent key by
      data-slice content hash, so slices are reused across experiments),
   3. runs a SageMaker Processing job for feature engineering → train/val/test
      splits,
   4. writes a training script from a template → SageMaker Training job,
   5. logs params/metrics/artifacts to MLflow; returns structured
      `{metric on held-out test, cost est, job ARNs, mlflow run}`.
4. **[workflow judge stage]** rank experiments by `success_metric` on the
   consistent held-out **test** split.
5. **[workflow] Update best-so-far; evaluate STOP conditions:**
   - goal met (best >= threshold) → register winner, stop;
   - budget exceeded (rounds/cost/time) → stop, report best;
   - plateau (no `min_delta` gain in `patience_rounds`) → stop, report best;
   - else → next round, seeding subagents with prior results (what worked/didn't).
6. **[skill] Register winning model** in the SageMaker Model Registry (package
   group), with MLflow run link + metrics; emit final report.

**Key details:** data slices are content-hashed so repeated UNLOADs are cached
in S3 (saves cost); the held-out test split is scored consistently across all
experiments so the judge compares apples to apples; each round feeds prior
results forward so the search learns rather than restarting blind.

## Error handling

- **Scripts** return structured JSON (`{status, data, error}`), never bare
  crashes. AWS calls use the user's profile; read-only ops (Data API discovery,
  describe/list) preferred; failures surface the AWS error verbatim.
- **A failed experiment subagent** doesn't kill the round — the workflow drops
  it to `null`, logs why, and the judge ranks survivors. If *all* experiments in
  a round fail, the loop stops and reports rather than burning budget blindly.
- **SageMaker job failures** (job errors, throttling) are caught by
  `sm_launch.py`, which returns the CloudWatch log tail so the agent can
  diagnose and adjust next round.
- **Partial progress is durable:** best-so-far, per-round results, and job ARNs
  are written to a run-state file (`.sm-research/run-<id>.json`) so a killed loop
  can be inspected/resumed.

## Cost & production safety

Follows the user's production-safety rules:

- MLflow server creation, and any resource creation, requires explicit user
  confirmation with cost stated first.
- Every round logs cumulative estimated cost; `budget` caps are hard ceilings.
- No deletion of any AWS resource without explicit confirmation. Never assume
  production is safe to touch; assume production when uncertain.
- Prefer ReadOnly / least-privilege operations.

## Testing

- **Scripts:** unit-tested with mocked boto3 (`moto` where possible) — SQL
  rendering, config validation, data-slice hashing, stop-condition math. Run via
  `uv`.
- **Config validation:** schema-validate `research.yaml` (required fields,
  metric/threshold sanity, allowlist non-empty) — fail fast before any AWS call.
- **Workflow logic:** stop-condition and judge-ranking functions are pure and
  unit-tested independently of AWS.
- **Dry-run mode:** `/sm-research --dry-run` validates config + Redshift
  discovery + a single tiny experiment on a sampled slice, so users can
  smoke-test end-to-end cheaply before a full run.

## Out of scope (YAGNI)

- Standing server-side orchestration (Step Functions/Lambda) — the loop lives in
  the Workflow, not deployed infra.
- AutoML/Autopilot — custom scripts only.
- Creating Redshift clusters or SageMaker execution roles — assumed to exist.
- Non-Redshift data sources.
