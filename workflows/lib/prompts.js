"use strict";

// Source of truth for the experiment-subagent prompt. auto-research-loop.js
// carries a verbatim inline copy between the "inlined prompts" markers (Workflow
// scripts cannot require() siblings); prompts.test.js fails if the two drift.
// The eval in evals/run-one-experiment renders cases through this module.

// --- inlined prompts ---
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
    billable_seconds: { type: "number" },
    failure_reason: { type: ["string", "null"] },
    // Cost provenance: lets the loop distinguish a cheap round from an unpriced
    // one instead of trusting a total that may silently omit charges.
    rate_source: { type: ["string", "null"] },
    hourly_rate_usd: { type: ["number", "null"] },
    instance_type: { type: ["string", "null"] },
  },
};

function priorSummaryText(best, rounds) {
  return rounds.length
    ? `Prior rounds best=${best.metric}. What worked: ${JSON.stringify(rounds.at(-1).topHypotheses || [])}.`
    : "First round — establish a baseline and diversify hypotheses.";
}

function experimentPrompt({ cfg, configPath, runId, round, index, priorSummary }) {
  const metricGoal = cfg.problem.metric_goal;
  const threshold = cfg.problem.threshold;
  const metric = cfg.problem.success_metric;
  return (
    `You are experiment ${index + 1} in round ${round + 1} of a SageMaker auto-research loop.\n` +
    `Goal: ${metricGoal} ${metric} to >= ${threshold} (problem: ${cfg.problem.type}).\n` +
    `${priorSummary}\n` +
    `Config path: ${configPath}. Run id: ${runId}. Round: ${round + 1}.\n` +
    `Allowed data (DO NOT exceed): ${JSON.stringify(cfg.data.allowed)}; target=${cfg.data.target}.\n\n` +
    `Follow the sagemaker-auto-research skill's "Run one experiment" section EXACTLY:\n` +
    `1. Choose a distinct hypothesis (feature subset/mix, algorithm, hyperparameters).\n` +
    `2. Build a data-slice select_sql over allowed data and call redshift_unload.py (cached by hash).\n` +
    `3. Author a training script from templates/train and launch it via sm_launch.py.\n` +
    `4. Log params/metrics to MLflow experiment '${cfg.sagemaker.mlflow.experiment_name}'.\n` +
    `5. Return the test-split ${metric} as 'metric' (null if the job failed), plus status, ` +
    `hypothesis, mlflow_run_id, model_data_s3, image_uri, cost_usd, billable_seconds, failure_reason.`
  );
}
// --- end inlined prompts ---

module.exports = { EXPERIMENT_SCHEMA, priorSummaryText, experimentPrompt };
