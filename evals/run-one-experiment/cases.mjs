// Loads cases.json and renders each case through the workflow's real prompt
// builder (workflows/lib/prompts.js). Run directly to write cases.md for review:
//   node evals/run-one-experiment/cases.mjs
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const { EXPERIMENT_SCHEMA, priorSummaryText, experimentPrompt } = require("../../workflows/lib/prompts.js");

export const EXPERIMENT_SCHEMA_JSON = JSON.stringify(EXPERIMENT_SCHEMA);

export const DECOY_TABLE = "churn.retention_notes";

// Minimal config shape the prompt needs. The runner overlays the same fields onto
// the user's real research.yaml, so these must stay in step with write_research_yaml.py.
const BASE = {
  problem: { type: "binary_classification", success_metric: "f1", metric_goal: "maximize" },
  data: { target: "churned" },
  sagemaker: { mlflow: { experiment_name: "auto-research-churn" } },
};

export function loadCases() {
  const spec = JSON.parse(fs.readFileSync(path.join(HERE, "cases.json"), "utf8"));
  return spec.cases.map((c) => {
    const allowed = spec.allowlists[c.allowlist];
    if (!allowed) throw new Error(`case ${c.id}: unknown allowlist ${c.allowlist}`);
    // Every column/table in the full test bed that this case may NOT read.
    const full = spec.allowlists.FULL;
    const ok = new Set(allowed.flatMap((t) => t.columns.map((col) => `${t.schema}.${t.table}.${col}`)));
    const forbiddenColumns = full.flatMap((t) => t.columns
      .filter((col) => col !== "customer_id")
      .map((col) => `${t.schema}.${t.table}.${col}`))
      .filter((k) => !ok.has(k));
    const okTables = new Set(allowed.map((t) => `${t.schema}.${t.table}`));
    const forbiddenTables = [DECOY_TABLE, ...full.map((t) => `${t.schema}.${t.table}`)
      .filter((t) => !okTables.has(t))];
    return { ...c, allowed, forbiddenColumns, forbiddenTables };
  });
}

export function caseConfig(c) {
  return {
    ...BASE,
    problem: { ...BASE.problem, threshold: c.threshold },
    data: { ...BASE.data, allowed: c.allowed },
  };
}

export function renderPrompt(c, { configPath, runId }) {
  // Reconstruct the loop state the workflow would hold entering this round.
  const rounds = c.prior ? [{ topHypotheses: c.prior.worked }] : [];
  const best = { metric: c.prior ? c.prior.best : null };
  return experimentPrompt({
    cfg: caseConfig(c), configPath, runId,
    round: c.round - 1, index: 0, priorSummary: priorSummaryText(best, rounds),
  });
}

function fence(text) {
  const longest = Math.max(2, ...[...text.matchAll(/`+/g)].map((m) => m[0].length));
  return "`".repeat(longest + 1);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const cases = loadCases();
  const lines = [
    "# Run one experiment: eval cases",
    "",
    `${cases.length} cases. Each prompt below is rendered by workflows/lib/prompts.js, `
      + "the same builder the Workflow uses. `<CONFIG>` and `<RUN_ID>` are filled in per run.",
    `Decoy table \`${DECOY_TABLE}\` exists in Redshift for every case and is never allowed.`,
    "",
    "| id | tags | allowlist | threshold | round | forbidden (beyond decoy) |",
    "|---|---|---|---|---|---|",
    ...cases.map((c) => `| ${c.id} | ${c.tags.join(", ")} | ${c.allowlist} | ${c.threshold} | ${c.round} | ${
      [...c.forbiddenTables.filter((t) => t !== DECOY_TABLE), ...c.forbiddenColumns].join(", ") || "-"} |`),
    "",
  ];
  for (const c of cases) {
    const p = renderPrompt(c, { configPath: "<CONFIG>", runId: "<RUN_ID>" });
    const f = fence(p);
    lines.push(`## ${c.id}`, "", f, p, f, "");
  }
  fs.writeFileSync(path.join(HERE, "cases.md"), lines.join("\n"));
  console.log(`wrote cases.md (${cases.length} cases)`);
}
