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
  if (history.length <= n) return false;
  const window = history.slice(-n);
  const first = window[0];
  const last = window[window.length - 1];
  const gain = metricGoal === "minimize" ? first - last : last - first;
  return gain < plateau.min_delta;
}

// An experiment that trained fine but could not log to MLflow still returns a
// metric, so it ranks and can even win — silently defeating the point of
// tracking every experiment. Surface those instead of letting them pass.
function untrackedExperiments(results) {
  return results.filter(
    (r) => r && typeof r.metric === "number" && !r.mlflow_run_id
  );
}

// An experiment whose instance type had no known price contributes $0.00 to the
// running total, silently making max_total_cost_usd unenforceable. Surface those
// rather than trusting a total that may be missing arbitrary amounts.
function unpricedExperiments(results) {
  return results.filter(
    (r) => r && (r.rate_source === "unknown" || r.hourly_rate_usd == null)
  );
}

module.exports = {
  rankExperiments, bestMetric, goalMet, budgetExceeded, plateaued,
  untrackedExperiments, unpricedExperiments,
};
