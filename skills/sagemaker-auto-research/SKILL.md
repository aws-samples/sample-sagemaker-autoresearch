---
name: sagemaker-auto-research
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
cd ${CLAUDE_PLUGIN_ROOT} && uv run python -c "from sm_research import config; errs=config.validate(config.load('$CFG')); print('\n'.join(errs) or 'OK')"
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
   (joins/filters allowed within the allowlist). Do the feature engineering here:
   per-entity aggregates (`GROUP BY` on the key), joins, ratios, CASE buckets.
   Include the target column. UNLOAD it (cached by content hash):
   ```bash
   cd ${CLAUDE_PLUGIN_ROOT} && uv run python -m sm_research.redshift_unload --config "$CFG" --run-id "<id>" \
     --spec-json '{"tables":[...],"columns":[...],"filter":null,"seed":42,"select_sql":"<SQL>"}'
   ```
   The result's `data.s3_prefix` holds the whole slice as flat `*.parquet` files
   (no train/val/test directories). Pass that prefix as `--input-s3` in step 5.
   You do not need to read any `sm_research` source or list S3 to learn this.
3. **Splits are done by the templates — do not split the data yourself.** Both
   templates load every `*.parquet` in the training channel and split it 60/20/20
   train/val/test, stratified on the target with seed 42, so the test rows are
   fixed for a slice. They drop `id`/`*_id` columns, tune the F1 threshold on
   val, and print `f1=<test value>;`. The target name is passed for you
   (`data.target` → `--target`).

   Only if the slice will not comfortably fit or transform in memory, use a
   **Processing job** so the work runs on SageMaker and its cost counts against
   the budget; write `train/` and `test/` subdirectories to `output_s3` (the
   templates then use them as given) and pass `output_s3` as `--input-s3`:
   ```bash
   # upload your featurize script first, then:
   cd ${CLAUDE_PLUGIN_ROOT} && uv run python -m sm_research.sm_process --config "$CFG" \
     --job-json '{"job_name":"fe-<unique>","image_uri":"<sklearn-image>","input_s3":"<slice>/","output_s3":"s3://<bucket>/sm-research/<id>/features/<exp>/","code_s3":"s3://<bucket>/sm-research/<id>/code/<exp>/featurize.py","instance_type":"ml.m5.xlarge"}'
   ```
   The container sees the slice at `/opt/ml/processing/input` and must write to
   `/opt/ml/processing/output`, which is synced to `output_s3`. Add `cost_usd`
   from the result to the experiment's total so Processing time is not free.
4. **Pick the training script.** The templates run unmodified on a slice:
   - `${CLAUDE_PLUGIN_ROOT}/templates/train/xgboost_train.py` (xgboost image) —
     hyperparameters `max-depth` (4), `eta` (0.05), `num-round` (500, early
     stopping on val), `early-stopping-rounds` (30), `subsample` (1.0),
     `colsample-bytree` (1.0), `min-child-weight` (1.0), `scale-pos-weight`
     (default: train neg/pos ratio).
   - `${CLAUDE_PLUGIN_ROOT}/templates/train/sklearn_train.py` (sklearn image) —
     `model` (`rf` | `hgb`), `n-estimators` (300), `max-depth` (none),
     `min-samples-leaf` (5), `learning-rate` (0.05, hgb), `max-iter` (300, hgb),
     `class-weight` (`balanced` | `none`).

   If the hypothesis lives in `select_sql` + hyperparameters (the usual case),
   pass the template path itself as `--script` — no Read, no copy. For in-script
   feature engineering, copy the template to an absolute path under the directory
   that holds `research.yaml` or `$TMPDIR` — never inside `${CLAUDE_PLUGIN_ROOT}`,
   and never modify the templates — and edit only `_features()` (row-wise, applied to each split) and
   `main()`'s model. Keep `_splits()` and `_prepare()`: `_prepare` converts
   Redshift DECIMAL columns (which arrive as `object`) to floats and encodes
   categoricals safely; one-hot encoding those columns yourself runs the job out
   of memory with no traceback.

5. **Package, launch, read the result, and log to MLflow in ONE call.** Do NOT
   call `package_code`/`sm_launch`, `aws logs`, `describe-training-job`, or write
   MLflow code yourself — `run_training` does all of it, and its result is your
   return record:
   ```bash
   cd ${CLAUDE_PLUGIN_ROOT} && uv run --with mlflow --with sagemaker-mlflow python -m sm_research.run_training \
     --config "$CFG" --run-id "<id>" --experiment "r<round>-e<n>" \
     --script <template-or-your-script.py> --entry-point <its-filename.py> \
     --image-uri <framework-image> --input-s3 "<data.s3_prefix-or-features-prefix>" \
     --hyperparameters-json '{"max-depth":4,"eta":0.05}' --hypothesis "<one line>"
   ```
   Give this Bash call `timeout: 600000`; it blocks until the job ends (~3–5 min).
   - Use a **framework** image (in us-west-2:
     `246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3`
     or `246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:1.7-1`).
     Hyperparameter values may be numbers (they are stringified for you); pass
     only keys your script's argparse accepts. Instance type defaults to
     `sagemaker.instance_type` (`--instance-type` to override).
   - The script must print the held-out **test** metric exactly once as
     `<success_metric>=<value>;` (e.g. `f1=0.73;`), as the templates do. Print
     anything else under another name (`val_f1=...;`, `threshold=...;`) — every
     `name=number;` pair is logged to MLflow as a metric.
   - If you ran a Processing job, add `--extra-cost-usd <c> --extra-billable-seconds <s>`.
6. **If `record.status` is `Failed`**, read `log_tail` (the script's own
   traceback, already extracted from CloudWatch) and `failure_reason`, fix the
   script, and rerun the SAME command with the SAME `--experiment` (at most 3
   attempts in total). Attempts are named `<prefix>-a1`, `-a2`, …, and the
   record's `cost_usd`/`billable_seconds` already sum every attempt.
   If the job completed but `mlflow_error` is set, rerun with `--reuse-last`
   added: it re-reads and re-logs the latest attempt without launching a job.
7. **Return `data.record` as your result, field for field** — it already holds
   `metric`, `status`, `hypothesis`, `mlflow_run_id`, `model_data_s3`,
   `image_uri`, `cost_usd`, `billable_seconds`, `failure_reason`, `rate_source`,
   `hourly_rate_usd`, and `instance_type`, all consistent with the job and the
   MLflow run. Do not recompute or re-read any of them.

## Failure handling

- Any script returning `{"status":"error"}` — read `error`, fix inputs, retry once;
  if it persists, surface it to the user verbatim.
- A failed training job — `sm_launch` returns the status and `failure_reason`; return
  `metric: null` so the judge drops it. Adjust the hypothesis next round.
  `failure_reason` is truncated by SageMaker and often hides the real cause. For
  the actual Python traceback, read the job's log stream:
  ```bash
  aws logs tail /aws/sagemaker/TrainingJobs --log-stream-name-prefix <job_name> \
    --profile <profile> --region <region> --since 30m
  ```
- If every experiment in a round fails, the Workflow stops. Investigate the common
  cause (IAM, image, data) before rerunning.

## Dry run

For `--dry-run` (see the `/sm-research` command): do steps 0–1 and a single tiny
experiment on a `LIMIT`-sampled slice with `experiments_per_round=1, max_rounds=1`,
then report — so the user validates the pipeline cheaply before a full run.

`redshift_unload` nests your `select_sql` in a subquery, so a trailing `LIMIT` is
safe to write even though Redshift rejects a top-level `LIMIT` inside `UNLOAD`.
