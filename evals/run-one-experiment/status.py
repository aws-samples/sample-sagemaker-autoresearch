"""Hillclimb status table for the run-one-experiment flow, computed from the rows.

Writes cost_usd onto every results.jsonl row (model x usage at the _state.json
prices; cache read 0.1x, write 1.25x input) so the report and status table read
the same number, then prints the per-variant table with the adoption gates.

  python3 evals/run-one-experiment/status.py [.claude/hillclimb/run-one-experiment]
"""
import json
import math
import pathlib
import re
import statistics as st
import sys
from collections import defaultdict

FLOW = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".claude/hillclimb/run-one-experiment")


def row_cost(r, prices):
    p = prices[r["model"]]
    u = r["usage"]
    return (u["input_tokens"] * p["in"] + u["output_tokens"] * p["out"]
            + u["cache_read_input_tokens"] * p["in"] * 0.1
            + u["cache_creation_input_tokens"] * p["in"] * 1.25) / 1e6


def case_mean(rows, ids, key):
    per = defaultdict(list)
    for r in rows:
        if r["prompt_id"] in ids and key(r) is not None:
            per[r["prompt_id"]].append(key(r))
    means = [st.mean(v) for v in per.values()]
    if not means:
        return None, None
    hw = 1.96 * st.stdev(means) / math.sqrt(len(means)) if len(means) > 1 else 0.0
    return st.mean(means), hw


def main():
    state = json.loads((FLOW / "_state.json").read_text())
    prices, gates = state["prices"], state["gates"]
    train, test = set(state["train_ids"]), set(state["test_ids"])
    variants = sorted((d for d in FLOW.iterdir() if d.is_dir() and re.fullmatch(r"baseline|v[1-9]\d*", d.name)),
                      key=lambda d: -1 if d.name == "baseline" else int(d.name[1:]))
    spend = 0.0
    lines = ["| round | change | config | test $/exp | train $/exp | test valid | test F1 | allowlist | "
             "turns | out tok | errors | gates | spend |", "|" + "---|" * 13]
    base_cost = None
    for v in variants:
        path = v / "results.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        for r in rows:
            r["cost_usd"] = round(row_cost(r, prices), 4)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        errs = [json.loads(l) for l in (v / "errors.jsonl").read_text().splitlines()
                if l.strip()] if (v / "errors.jsonl").exists() else []
        err_cost = sum(row_cost(e, prices) for e in errs if e.get("usage") and e.get("model") in prices)
        spend += sum(r["cost_usd"] + (r.get("judge_cost_usd") or 0) + (r.get("aws_cost_usd") or 0)
                     for r in rows) + err_cost
        tc, tch = case_mean(rows, test, lambda r: r["cost_usd"])
        trc, _ = case_mean(rows, train, lambda r: r["cost_usd"])
        tv, _ = case_mean(rows, test, lambda r: r["grade"]["valid_exp"])
        tf, _ = case_mean(rows, test, lambda r: r["grade"]["f1"])
        allow = min(r["grade"]["allowlist"] * r["grade"]["guardrails"] for r in rows)
        turns = st.mean(r["turns"] for r in rows if r["prompt_id"] in test)
        out = st.mean(r["usage"]["output_tokens"] for r in rows if r["prompt_id"] in test)
        if v.name == "baseline":
            base_cost, base_spend = tc, spend
        cut = 1 - tc / base_cost if base_cost else 0
        ok = (allow >= gates["allowlist_min"] and tv >= gates["valid_exp_test_min"]
              and tf >= gates["f1_test_min"] and (v.name == "baseline" or cut >= gates["cost_cut_min"]))
        change = ((v / "change.md").read_text().splitlines()[0].lstrip("# ").strip()
                  if (v / "change.md").exists() else "baseline")
        r0 = rows[0]
        trc_s = "-" if trc is None else f"${trc:.3f}"
        cfg = f"{r0['model'].replace('claude-', '')} @ {(r0.get('run') or {}).get('effort', 'default')}"
        n_expected = len(train | test) * state["reps"]
        lines.append(
            f"| {v.name} | {change} | {cfg} | ${tc:.3f} ±{tch:.3f} ({1 - cut:.2f}×) | {trc_s} | {tv:.3f} | "
            f"{tf:.3f} | {allow:.0f} | {turns:.1f} | {out:,.0f} | {len(errs)} | "
            f"{'baseline' if v.name == 'baseline' else ('PASS' if ok else 'fail')} | ${spend:.2f} |"
            + ("" if len(rows) >= n_expected else f" PARTIAL {len(rows)}/{n_expected}"))
    print("\n".join(lines))
    loop = spend - base_spend
    print(f"\nloop spend ${loop:.2f} of ${state['budget_usd']} ceiling (excludes the ${base_spend:.2f} baseline; "
          "Claude table + judge + SageMaker + failed attempts; Redshift not measured)")


if __name__ == "__main__":
    main()
