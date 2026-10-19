export const meta = {
  name: "sagemaker-auto-research-loop",
  description: "Iterative SageMaker experimentation loop: fan out experiments, judge, check stop conditions until goal met or budget/plateau reached.",
  phases: [
    { title: "Experiment" },
    { title: "Judge" },
  ],
};

// --- Inlined pure decisions (source of truth: workflows/lib/decisions.js, unit-tested) ---
// These are copied VERBATIM in logic from workflows/lib/decisions.js. Workflow scripts run
// in a restricted context where require() of a sibling file is not guaranteed, so the pure
// decision functions are inlined here. decisions.js remains the tested source of truth.
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
  if (history.length <= n) return false;
  const window = history.slice(-n);
  const first = window[0];
  const last = window[window.length - 1];
  const gain = metricGoal === "minimize" ? first - last : last - first;
  return gain < plateau.min_delta;
}

function untrackedExperiments(results) {
  return results.filter(
    (r) => r && typeof r.metric === "number" && !r.mlflow_run_id
  );
}

function unpricedExperiments(results) {
  return results.filter(
    (r) => r && (r.rate_source === "unknown" || r.hourly_rate_usd == null)
  );
}
// --- end inlined ---

// --- Inlined prompts (source of truth: workflows/lib/prompts.js; prompts.test.js checks they match) ---
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

function _better(candidate, current, metricGoal) {
  return metricGoal === "minimize" ? candidate < current : candidate > current;
}

// args can arrive either as a real object or JSON-encoded, depending on how the
// caller hands it over. Accept both rather than failing on a property access
// with an opaque "undefined is not an object".
let input = args;
if (typeof input === "string") {
  try {
    input = JSON.parse(input);
  } catch (e) {
    throw new Error(`args was a string but is not valid JSON: ${e.message}`);
  }
}
if (!input || !input.config || !input.config.budget) {
  throw new Error(
    "args.config must be the parsed research.yaml (needs budget, problem, " +
    `data, sagemaker). Got: ${JSON.stringify(input)?.slice(0, 200)}`
  );
}

const cfg = input.config;              // parsed research.yaml (skill passes it in)
const budget = cfg.budget;
const metricGoal = cfg.problem.metric_goal;
const threshold = cfg.problem.threshold;
const metric = cfg.problem.success_metric;

const history = [];        // best metric per round
const rounds = [];
const untrackedRuns = [];  // experiments that trained but never reached MLflow
const unpricedRuns = [];   // experiments with no known hourly rate

// Each round spends real time outside the training job itself (Redshift UNLOAD,
// feature engineering, code packaging) that no billable_seconds figure captures.
// Measured across three runs, a ~100s job sat inside a ~7min round, so charge
// the wallclock budget a multiple of the longest job rather than treating that
// setup time as free. Cost is unaffected — only training instances bill.
const ROUND_OVERHEAD_FACTOR = 4;
let cumulativeCost = 0;
let cumulativeHours = 0;
let best = { metric: null, experiment: null };
let stopReason = null;

for (let round = 0; round < budget.max_rounds; round++) {
  const pre = budgetExceeded(
    { round, costUsd: cumulativeCost, elapsedHours: cumulativeHours }, budget);
  if (pre.exceeded) { stopReason = pre.reason; break; }

  phase("Experiment");
  log(`Round ${round + 1}/${budget.max_rounds}: launching ${budget.experiments_per_round} experiments`);

  const priorSummary = priorSummaryText(best, rounds);

  const experiments = await parallel(
    Array.from({ length: budget.experiments_per_round }, (_, i) => () =>
      agent(
        experimentPrompt({ cfg, configPath: input.configPath, runId: input.runId,
                           round, index: i, priorSummary }),
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
  // Wall-clock is approximated: parallel jobs overlap, so the round's contribution
  // is the longest job's billable time (Workflow scripts can't call Date.now()).
  // Still excludes Redshift UNLOAD and packaging time, so it remains a floor —
  // but it now counts an overhead factor so long setup phases are not free.
  const roundSeconds = survivors.reduce((m, e) => Math.max(m, e.billable_seconds || 0), 0);
  cumulativeHours += (roundSeconds * ROUND_OVERHEAD_FACTOR) / 3600;

  // A round whose cost is partly unpriced makes the budget cap unenforceable.
  const unpriced = unpricedExperiments(survivors);
  if (unpriced.length) {
    unpricedRuns.push(...unpriced.map((e) => ({
      round: round + 1,
      instance_type: e.instance_type || null,
      rate_source: e.rate_source || null,
    })));
    log(`WARNING round ${round + 1}: ${unpriced.length}/${survivors.length} ` +
        `experiment(s) had no known hourly rate — cumulative cost is a LOWER BOUND.`);
  }
  const roundBest = bestMetric(survivors, metricGoal);
  history.push(roundBest);

  // Training succeeded but MLflow logging did not — the metric is real, so the
  // experiment ranks normally and the gap would otherwise go unnoticed.
  const untracked = untrackedExperiments(survivors);
  if (untracked.length) {
    untrackedRuns.push(...untracked.map((e) => ({
      round: round + 1,
      metric: e.metric,
      failure_reason: e.failure_reason || null,
    })));
    log(`WARNING round ${round + 1}: ${untracked.length}/${survivors.length} ` +
        `experiment(s) were NOT logged to MLflow — results exist but are untracked.`);
  }

  if (roundBest !== null && (best.metric === null || _better(roundBest, best.metric, metricGoal))) {
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
  const post = budgetExceeded({ round: round + 1, costUsd: cumulativeCost, elapsedHours: cumulativeHours }, budget);
  if (post.exceeded) { stopReason = post.reason; break; }
  if (plateaued(history, budget.plateau, metricGoal)) { stopReason = "plateau detected"; break; }
}

if (!stopReason) stopReason = "max_rounds reached";

const winnerRegisterable = best.experiment
  ? Boolean(best.experiment.model_data_s3 && stopReason === "goal met")
  : false;

// The winner itself being untracked is worth calling out separately: registering
// a model whose experiment was never recorded loses its provenance.
const winnerTracked = best.experiment
  ? Boolean(best.experiment.mlflow_run_id) : false;

return {
  best, rounds, stopReason, metric, threshold, metricGoal, winnerRegisterable,
  winnerTracked, untrackedRuns,
  // costIsLowerBound: the reported total omits at least one unpriced experiment,
  // and always omits Redshift and MLflow charges.
  unpricedRuns, costIsLowerBound: unpricedRuns.length > 0,
  totalCostUsd: cumulativeCost, estimatedHours: cumulativeHours,
};
