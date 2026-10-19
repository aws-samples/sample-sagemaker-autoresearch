import ast
import pathlib

import yaml

from sm_research import config

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_research_template_parses_and_validates():
    raw = (ROOT / "templates/research.yaml.tmpl").read_text()
    cfg = yaml.safe_load(raw)
    # Template ships with working sample values, so it must validate clean.
    assert config.validate(cfg) == []


def test_training_templates_are_valid_python():
    for name in ("xgboost_train.py", "sklearn_train.py"):
        src = (ROOT / "templates/train" / name).read_text()
        ast.parse(src)  # raises SyntaxError if invalid


def test_pyproject_declares_package_location():
    """The skill runs `uv run python -m sm_research.<module>` from the plugin root.

    Regression: without a build-system + wheel packages entry, sm_research was
    importable only via the pytest-only `pythonpath`, so every documented
    command failed on a clean install with ModuleNotFoundError.
    """
    import tomllib

    raw = (ROOT / "pyproject.toml").read_bytes()
    cfg = tomllib.loads(raw.decode())
    assert "build-system" in cfg, "pyproject must declare a build-system"
    packages = cfg["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert "scripts/sm_research" in packages


def _load_prepare(name, tmp_path):
    # Import only pandas, the constants, and _prepare, so the test runs
    # without xgboost/sklearn installed.
    src = (ROOT / "templates/train" / name).read_text()
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)) and
            any(a.name == "pandas" for a in n.names)]
    keep += [n for n in tree.body if isinstance(n, ast.Assign) or
             (isinstance(n, ast.FunctionDef) and n.name == "_prepare")]
    subset = tmp_path / f"prepare_{name}"
    subset.write_text(ast.unparse(ast.Module(body=keep, type_ignores=[])))
    return _load_module(subset)._prepare


def test_prepare_converts_decimals_and_encodes_categoricals(tmp_path):
    """Regression: Decimal-valued object columns were one-hot encoded per value
    and ran SageMaker jobs out of memory with no traceback."""
    from decimal import Decimal
    import pandas as pd
    for name in ("xgboost_train.py", "sklearn_train.py"):
        prepare = _load_prepare(name, tmp_path)
        train = pd.DataFrame({"price": [Decimal("9.99"), Decimal("19.5"), Decimal("4.25")] * 40,
                              "plan": ["basic", "pro", "basic"] * 40,
                              "big": [f"v{i}" for i in range(120)],
                              "autopay": [True, False, True] * 40})
        test = pd.DataFrame({"price": [Decimal("1.00")], "plan": ["enterprise"],
                             "big": ["unseen"], "autopay": [False]})
        tr, te = prepare(train, test)
        assert tr["price"].dtype == "float64" and te["price"].iloc[0] == 1.0
        assert {"plan=basic", "plan=pro"} <= set(tr.columns) and "plan" not in tr.columns
        assert tr["big"].dtype == "int32" and te["big"].iloc[0] == -1
        assert list(tr.columns) == list(te.columns)
        assert all(str(d) != "object" for d in tr.dtypes)


def _load_module(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_template(name):
    """Import a training template as a module without running main()."""
    return _load_module(ROOT / "templates/train" / name)


def test_splits_handle_a_flat_unload_slice(tmp_path):
    """Regression: the templates read train/ and test/ subdirectories, but
    redshift_unload writes one flat slice, so every experiment rewrote the
    loader (or failed with "No objects to concatenate")."""
    import pandas as pd
    import pytest
    pytest.importorskip("sklearn")
    df = pd.DataFrame({"customer_id": range(200), "x": [i % 7 for i in range(200)],
                       "churned": [i % 4 == 0 for i in range(200)]})
    df.iloc[:120].to_parquet(tmp_path / "000.parquet")
    df.iloc[120:].to_parquet(tmp_path / "001.parquet")
    mod = _load_template("sklearn_train.py")
    train, val, test = mod._splits(str(tmp_path), "churned")
    assert (len(train), len(val), len(test)) == (120, 40, 40)
    assert set(train.customer_id) | set(val.customer_id) | set(test.customer_id) == set(range(200))
    # Deterministic regardless of file/row order.
    shuffled = tmp_path / "shuffled"
    shuffled.mkdir()
    df.sample(frac=1, random_state=7).to_parquet(shuffled / "000.parquet")
    assert sorted(mod._splits(str(shuffled), "churned")[2].customer_id) == sorted(test.customer_id)
    assert "customer_id" not in mod._features(train).columns


def test_splits_use_presplit_channel(tmp_path):
    import pandas as pd
    import pytest
    pytest.importorskip("sklearn")
    for split, n in (("train", 80), ("test", 20)):
        (tmp_path / split).mkdir()
        pd.DataFrame({"x": range(n), "y": [i % 2 for i in range(n)]}).to_parquet(
            tmp_path / split / "part.parquet")
    train, val, test = _load_template("sklearn_train.py")._splits(str(tmp_path), "y")
    assert (len(train), len(val), len(test)) == (60, 20, 20)
