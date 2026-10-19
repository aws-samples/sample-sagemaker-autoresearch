const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const p = require("./prompts.js");

const BEGIN = "// --- inlined prompts ---";
const END = "// --- end inlined prompts ---";

function inlinedBlock(file) {
  const src = fs.readFileSync(file, "utf8");
  const start = src.indexOf(BEGIN);
  const end = src.indexOf(END);
  assert.ok(start >= 0 && end > start, `markers missing in ${file}`);
  return src.slice(start, end);
}

test("workflow's inlined prompt block matches prompts.js verbatim", () => {
  assert.equal(
    inlinedBlock(path.join(__dirname, "..", "auto-research-loop.js")),
    inlinedBlock(path.join(__dirname, "prompts.js"))
  );
});

test("priorSummaryText first round", () => {
  assert.match(p.priorSummaryText({ metric: null }, []), /^First round/);
});

test("priorSummaryText lists last round's top hypotheses", () => {
  const s = p.priorSummaryText({ metric: 0.61 }, [{ topHypotheses: ["a", "b"] }]);
  assert.equal(s, 'Prior rounds best=0.61. What worked: ["a","b"].');
});

test("experimentPrompt carries allowlist, round and experiment name", () => {
  const cfg = {
    problem: { metric_goal: "maximize", threshold: 0.7, success_metric: "f1",
               type: "binary_classification" },
    data: { allowed: [{ schema: "s", table: "t", columns: ["c"] }], target: "y" },
    sagemaker: { mlflow: { experiment_name: "exp" } },
  };
  const out = p.experimentPrompt({ cfg, configPath: "/c.yaml", runId: "r1",
                                   round: 2, index: 0, priorSummary: "PS" });
  assert.match(out, /experiment 1 in round 3/);
  assert.match(out, /\[\{"schema":"s","table":"t","columns":\["c"\]\}\]; target=y/);
  assert.match(out, /MLflow experiment 'exp'/);
  assert.match(out, /\nPS\n/);
});
