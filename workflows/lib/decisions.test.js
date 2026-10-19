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

test("untrackedExperiments flags a trained-but-unlogged experiment", () => {
  // Shape taken from a real run: training succeeded, MLflow returned 403, so
  // mlflow_run_id came back as "" while metric was a genuine number.
  const results = [
    { metric: 0.7345, mlflow_run_id: "f4a7b5fa417047c284165691dbf9c60f" },
    { metric: 0.7241, mlflow_run_id: "56db14e90dd7409f8749ca6fd7f6056b" },
    { metric: 0.7267, mlflow_run_id: "", failure_reason: "MLflow logging failed" },
  ];
  const out = d.untrackedExperiments(results);
  assert.equal(out.length, 1);
  assert.equal(out[0].metric, 0.7267);
});

test("untrackedExperiments ignores failed experiments with no metric", () => {
  // metric: null means the job itself failed; that is already handled by ranking
  // and is not a tracking gap.
  assert.equal(d.untrackedExperiments([{ metric: null, mlflow_run_id: "" }]).length, 0);
});

test("untrackedExperiments returns empty when everything is logged", () => {
  assert.equal(d.untrackedExperiments([{ metric: 0.8, mlflow_run_id: "abc" }]).length, 0);
});

test("untrackedExperiments treats a missing mlflow_run_id key as untracked", () => {
  assert.equal(d.untrackedExperiments([{ metric: 0.8 }]).length, 1);
});

test("unpricedExperiments flags unknown rate_source", () => {
  const out = d.unpricedExperiments([
    { metric: 0.7, rate_source: "pricing_api", hourly_rate_usd: 0.23 },
    { metric: 0.7, rate_source: "fallback_map", hourly_rate_usd: 0.23 },
    { metric: 0.7, rate_source: "unknown", hourly_rate_usd: null },
  ]);
  assert.equal(out.length, 1);
  assert.equal(out[0].rate_source, "unknown");
});

test("unpricedExperiments flags a null rate even when source looks valid", () => {
  // Guards the silent-$0.00 path: a missing rate must never read as free.
  assert.equal(
    d.unpricedExperiments([{ metric: 0.7, rate_source: "pricing_api" }]).length, 1);
});

test("unpricedExperiments returns empty when all are priced", () => {
  assert.equal(
    d.unpricedExperiments([{ metric: 0.7, rate_source: "pricing_api", hourly_rate_usd: 0.23 }]).length,
    0);
});
