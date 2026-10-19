---
name: sm-research
description: Set up and launch a SageMaker auto-research run — interview for goal/data/criteria, scaffold research.yaml, then run the loop until a model meets the goal or a guardrail stops it.
argument-hint: "[--dry-run]"
---

# /sm-research

You are setting up a SageMaker Auto Research run. Arguments: `$ARGUMENTS`
(if it contains `--dry-run`, do a cheap validation run — see the skill's "Dry run").

## Step 1 — Prerequisites

Confirm the user has:
- An AWS profile with SageMaker + Redshift Data API + S3 access
- A SageMaker execution role ARN
- An S3 bucket for artifacts
- Redshift data they're allowed to use

Per global rules, ALWAYS ask which AWS profile to use; never assume the default profile.

## Step 2 — Interview (one topic at a time)

Read the scaffold at `${CLAUDE_PLUGIN_ROOT}/templates/research.yaml.tmpl` and fill it in with the user. Collect the following, asking only for what you don't already know:

- **AWS**: profile (ask first, never assume default) + region
- **Redshift**: workgroup/cluster name, database, serverless? (true/false), optional secret ARN
- **Allowed data**: schema.table(s) and columns (the allowlist), and the target/label column
- **Problem**: type (binary_classification | multiclass | regression), success metric, threshold, metric_goal (maximize | minimize)
- **Budget**: max_rounds, experiments_per_round, max_total_cost_usd, max_wallclock_hours, plateau (patience_rounds + min_delta)
- **SageMaker**: execution role ARN, S3 bucket, instance type
- **MLflow**: existing tracking server ARN if any (else leave blank), experiment name
- **Model registry**: package group name

## Step 3 — Write research.yaml and Validate

Write the completed config to `./research.yaml` in the user's working directory (NOT into the plugin). Then validate it:

```bash
CWD="$PWD"; cd ${CLAUDE_PLUGIN_ROOT} && uv run python -c "from sm_research import config; print('\n'.join(config.validate(config.load('$CWD/research.yaml'))) or 'VALID')"
```

Fix any reported errors with the user before continuing. Only proceed when validation passes.

## Step 4 — Hand off to the loop

Invoke the **sagemaker-auto-research** skill to run the loop (or, with `--dry-run`, the dry-run path). Set `$CFG` to the absolute path of the written `research.yaml`.

Show the user the final result: best metric, whether the goal was met, the registered model package ARN (if any), the stop reason, and per-round summary.
