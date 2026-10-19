from sm_research import config


def _valid_cfg():
    return {
        "aws": {"profile": "p", "region": "us-east-1"},
        "data": {
            "redshift": {"workgroup_or_cluster": "wg", "database": "db"},
            "allowed": [{"schema": "public", "table": "t", "columns": ["a"]}],
            "target": "y",
        },
        "problem": {"type": "binary_classification", "success_metric": "f1",
                    "threshold": 0.9, "metric_goal": "maximize"},
        "budget": {"max_rounds": 5, "experiments_per_round": 4,
                   "max_total_cost_usd": 100, "max_wallclock_hours": 6,
                   "plateau": {"patience_rounds": 2, "min_delta": 0.005}},
        "sagemaker": {"role_arn": "arn:aws:iam::1:role/r", "s3_bucket": "b",
                      "instance_type": "ml.m5.xlarge",
                      "mlflow": {"tracking_server_arn": "", "experiment_name": "e"}},
        "model_registry": {"package_group": "g"},
    }


def test_valid_config_has_no_errors():
    assert config.validate(_valid_cfg()) == []


def test_missing_section_reported():
    cfg = _valid_cfg()
    del cfg["problem"]
    errs = config.validate(cfg)
    assert any("problem" in e for e in errs)


def test_empty_allowlist_rejected():
    cfg = _valid_cfg()
    cfg["data"]["allowed"] = []
    errs = config.validate(cfg)
    assert any("allowed" in e for e in errs)


def test_threshold_must_be_numeric():
    cfg = _valid_cfg()
    cfg["problem"]["threshold"] = "high"
    errs = config.validate(cfg)
    assert any("threshold" in e for e in errs)


def test_metric_goal_enum_enforced():
    cfg = _valid_cfg()
    cfg["problem"]["metric_goal"] = "sideways"
    errs = config.validate(cfg)
    assert any("metric_goal" in e for e in errs)


def test_missing_max_total_cost_usd_reported():
    cfg = _valid_cfg()
    cfg["budget"]["max_total_cost_usd"] = None
    errs = config.validate(cfg)
    assert any("max_total_cost_usd" in e for e in errs)


def test_missing_max_wallclock_hours_reported():
    cfg = _valid_cfg()
    cfg["budget"]["max_wallclock_hours"] = None
    errs = config.validate(cfg)
    assert any("max_wallclock_hours" in e for e in errs)


def test_none_valued_section_does_not_crash():
    cfg = _valid_cfg()
    cfg["aws"] = None            # empty "aws:" header in YAML
    errs = config.validate(cfg)  # must not raise
    assert any("aws" in e for e in errs)


def test_none_plateau_does_not_crash():
    cfg = _valid_cfg()
    cfg["budget"]["plateau"] = None
    errs = config.validate(cfg)  # must not raise
    assert isinstance(errs, list)


def test_load_reads_yaml(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text("aws:\n  profile: p\n  region: us-east-1\n")
    assert config.load(str(p))["aws"]["profile"] == "p"
