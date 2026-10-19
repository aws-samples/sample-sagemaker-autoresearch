"""Load and validate research.yaml."""
import numbers

import yaml

REQUIRED_SECTIONS = ["aws", "data", "problem", "budget", "sagemaker", "model_registry"]
_METRIC_GOALS = {"maximize", "minimize"}


def load(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _is_number(v) -> bool:
    return isinstance(v, numbers.Number) and not isinstance(v, bool)


def validate(cfg: dict) -> list[str]:
    errs: list[str] = []
    for section in REQUIRED_SECTIONS:
        if section not in cfg or not isinstance(cfg[section], dict):
            errs.append(f"missing required section: {section}")
    if errs:
        return errs  # deeper checks assume sections exist and are mappings

    aws = cfg["aws"]
    for k in ("profile", "region"):
        if not aws.get(k):
            errs.append(f"aws.{k} is required")

    data = cfg["data"]
    if not data.get("allowed"):
        errs.append("data.allowed must list at least one table")
    if not data.get("target"):
        errs.append("data.target (label column) is required")

    prob = cfg["problem"]
    if not _is_number(prob.get("threshold")):
        errs.append("problem.threshold must be numeric")
    if prob.get("metric_goal") not in _METRIC_GOALS:
        errs.append("problem.metric_goal must be 'maximize' or 'minimize'")
    if not prob.get("success_metric"):
        errs.append("problem.success_metric is required")

    budget = cfg["budget"]
    for k in ("max_rounds", "experiments_per_round"):
        if not isinstance(budget.get(k), int) or budget.get(k) < 1:
            errs.append(f"budget.{k} must be a positive integer")
    for k in ("max_total_cost_usd", "max_wallclock_hours"):
        if not _is_number(budget.get(k)) or budget.get(k) <= 0:
            errs.append(f"budget.{k} must be a positive number")
    plateau = budget.get("plateau") or {}
    if not isinstance(plateau.get("patience_rounds"), int):
        errs.append("budget.plateau.patience_rounds must be an integer")

    sm = cfg["sagemaker"]
    for k in ("role_arn", "s3_bucket", "instance_type"):
        if not sm.get(k):
            errs.append(f"sagemaker.{k} is required")

    if not cfg["model_registry"].get("package_group"):
        errs.append("model_registry.package_group is required")

    return errs
