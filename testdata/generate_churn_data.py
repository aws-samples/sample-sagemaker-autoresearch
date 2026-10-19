#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.5"]
# ///
"""Generate a multi-table synthetic churn dataset for testing SageMaker Auto Research.

The signal is deliberately SPLIT ACROSS TABLES via interaction terms, so that a
model built on any single table plateaus well below the goal, while a model built
on the joined + aggregated data can reach it. That forces the auto-research loop
to actually discover the join and the aggregation instead of getting lucky.

Tables written as CSV to --outdir:
  customers      (customer_id, age, region, signup_channel, tenure_months)
  subscriptions  (customer_id, plan_tier, monthly_price, contract_type, autopay)
  usage_monthly  (customer_id, month_num, avg_minutes, support_tickets,
                  late_payments, data_gb)   -- 12 rows per customer
  labels         (customer_id, churned)

Run with --report to fit reference models and print the achievable F1 per feature
set (single-table vs. joined), which is how the research.yaml threshold is chosen.
"""
import argparse
import pathlib

import numpy as np
import pandas as pd

MONTHS = 12

REGIONS = ["us-west", "us-east", "us-central", "us-south"]
CHANNELS = ["organic", "paid_search", "referral", "partner"]
TIERS = ["basic", "standard", "premium"]
CONTRACTS = ["month_to_month", "one_year", "two_year"]

# Marginal (single-table-visible) log-odds contributions.
CONTRACT_EFFECT = {"month_to_month": 0.85, "one_year": -0.25, "two_year": -0.80}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(n_customers: int, seed: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    cid = np.arange(1, n_customers + 1)

    # --- customers -------------------------------------------------------
    age = rng.integers(18, 80, n_customers)
    region = rng.choice(REGIONS, n_customers)
    channel = rng.choice(CHANNELS, n_customers, p=[0.4, 0.3, 0.2, 0.1])
    tenure = rng.integers(1, 72, n_customers)

    customers = pd.DataFrame({
        "customer_id": cid,
        "age": age,
        "region": region,
        "signup_channel": channel,
        "tenure_months": tenure,
    })

    # --- subscriptions ---------------------------------------------------
    tier = rng.choice(TIERS, n_customers, p=[0.45, 0.35, 0.20])
    base_price = {"basic": 25.0, "standard": 55.0, "premium": 95.0}
    price = np.array([base_price[t] for t in tier]) + rng.normal(0, 4, n_customers)
    price = np.round(np.clip(price, 10, None), 2)
    contract = rng.choice(CONTRACTS, n_customers, p=[0.55, 0.28, 0.17])
    # autopay correlates with longer contracts but is far from determined by them.
    autopay_p = np.where(contract == "month_to_month", 0.45, 0.78)
    autopay = rng.random(n_customers) < autopay_p

    subscriptions = pd.DataFrame({
        "customer_id": cid,
        "plan_tier": tier,
        "monthly_price": price,
        "contract_type": contract,
        "autopay": autopay,
    })

    # --- usage_monthly (panel: MONTHS rows per customer) -----------------
    # Per-customer latent usage intensity, then per-month noise around it.
    cust_minutes = rng.normal(420, 130, n_customers).clip(20, None)
    cust_ticket_rate = rng.gamma(1.4, 0.55, n_customers)     # tickets per month
    cust_late_rate = rng.beta(1.3, 9.0, n_customers)         # P(late) per month
    cust_data = rng.normal(14, 6, n_customers).clip(0.2, None)

    rep_cid = np.repeat(cid, MONTHS)
    month_num = np.tile(np.arange(1, MONTHS + 1), n_customers)
    avg_minutes = np.round(
        rng.normal(np.repeat(cust_minutes, MONTHS), 55).clip(0, None), 1)
    support_tickets = rng.poisson(np.repeat(cust_ticket_rate, MONTHS))
    late_payments = (rng.random(n_customers * MONTHS)
                     < np.repeat(cust_late_rate, MONTHS)).astype(int)
    data_gb = np.round(
        rng.normal(np.repeat(cust_data, MONTHS), 3.0).clip(0, None), 2)

    usage_monthly = pd.DataFrame({
        "customer_id": rep_cid,
        "month_num": month_num,
        "avg_minutes": avg_minutes,
        "support_tickets": support_tickets,
        "late_payments": late_payments,
        "data_gb": data_gb,
    })

    # Aggregates that drive the label (recoverable via GROUP BY).
    agg = usage_monthly.groupby("customer_id", sort=True).agg(
        total_tickets=("support_tickets", "sum"),
        total_late=("late_payments", "sum"),
        mean_minutes=("avg_minutes", "mean"),
    ).reindex(cid)

    total_tickets = agg["total_tickets"].to_numpy()
    total_late = agg["total_late"].to_numpy()
    mean_minutes = agg["mean_minutes"].to_numpy()

    # --- label -----------------------------------------------------------
    contract_eff = np.array([CONTRACT_EFFECT[c] for c in contract])
    tenure_eff = -0.045 * tenure
    ticket_eff = 0.16 * total_tickets
    late_eff = 0.22 * total_late
    # Weak, genuinely-present marginal signal on usage volume.
    minutes_eff = -0.0012 * (mean_minutes - 420)

    # CROSS-TABLE INTERACTIONS — invisible to any single-table model.
    # (1) month-to-month AND heavy support load  [subscriptions x usage]
    inter_1 = 1.75 * ((contract == "month_to_month") & (total_tickets >= 20))
    # (2) short tenure AND no autopay            [customers x subscriptions]
    inter_2 = 1.55 * ((tenure < 12) & (~autopay))
    # (3) long tenure AND clean payment record -> strongly sticky [customers x usage]
    inter_3 = -1.35 * ((tenure >= 36) & (total_late <= 1))

    logit = (-1.15 + tenure_eff + contract_eff + ticket_eff + late_eff
             + minutes_eff + inter_1 + inter_2 + inter_3)
    # Irreducible noise sets the Bayes ceiling; keeps the task non-trivial.
    logit += rng.normal(0, 0.55, n_customers)
    churned = (rng.random(n_customers) < _sigmoid(logit)).astype(int)

    labels = pd.DataFrame({"customer_id": cid, "churned": churned})

    return {
        "customers": customers,
        "subscriptions": subscriptions,
        "usage_monthly": usage_monthly,
        "labels": labels,
    }


# --------------------------------------------------------------------------
# Reference models: measure the F1 actually achievable per feature set.
# --------------------------------------------------------------------------
def _fit_report(frames: dict[str, pd.DataFrame], seed: int) -> list[tuple[str, float]]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    cust, sub, usage, lab = (frames["customers"], frames["subscriptions"],
                             frames["usage_monthly"], frames["labels"])
    agg = usage.groupby("customer_id").agg(
        total_tickets=("support_tickets", "sum"),
        total_late=("late_payments", "sum"),
        mean_minutes=("avg_minutes", "mean"),
        mean_data=("data_gb", "mean"),
        max_tickets=("support_tickets", "max"),
    ).reset_index()

    joined = (lab.merge(cust, on="customer_id")
                 .merge(sub, on="customer_id")
                 .merge(agg, on="customer_id"))

    feature_sets = {
        "customers only": ["age", "region", "signup_channel", "tenure_months"],
        "subscriptions only": ["plan_tier", "monthly_price", "contract_type", "autopay"],
        "usage only (aggregated)": ["total_tickets", "total_late", "mean_minutes",
                                    "mean_data", "max_tickets"],
        "customers + subscriptions": ["age", "region", "signup_channel", "tenure_months",
                                      "plan_tier", "monthly_price", "contract_type",
                                      "autopay"],
        "ALL THREE joined": ["age", "region", "signup_channel", "tenure_months",
                             "plan_tier", "monthly_price", "contract_type", "autopay",
                             "total_tickets", "total_late", "mean_minutes",
                             "mean_data", "max_tickets"],
    }

    y = joined["churned"].to_numpy()
    out = []
    for name, cols in feature_sets.items():
        X = pd.get_dummies(joined[cols], drop_first=False)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y)
        clf = HistGradientBoostingClassifier(random_state=seed, max_iter=300)
        clf.fit(Xtr, ytr)
        # Tune the decision threshold on train (as a real experiment would),
        # then score once on the held-out split.
        ptr = clf.predict_proba(Xtr)[:, 1]
        grid = np.linspace(0.2, 0.8, 61)
        best_t = max(grid, key=lambda t: f1_score(ytr, (ptr >= t).astype(int)))
        pte = clf.predict_proba(Xte)[:, 1]
        out.append((name, f1_score(yte, (pte >= best_t).astype(int))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-customers", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=str(pathlib.Path(__file__).parent / "csv"))
    ap.add_argument("--report", action="store_true",
                    help="fit reference models and print achievable F1 per feature set")
    args = ap.parse_args()

    frames = generate(args.n_customers, args.seed)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        path = outdir / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"wrote {path}  rows={len(df):,}  cols={len(df.columns)}")

    rate = frames["labels"]["churned"].mean()
    print(f"\nchurn base rate: {rate:.3f}")

    if args.report:
        print("\nachievable held-out F1 by feature set (HistGradientBoosting, tuned threshold):")
        for name, score in _fit_report(frames, args.seed):
            print(f"  {name:<28} F1 = {score:.4f}")


if __name__ == "__main__":
    main()
