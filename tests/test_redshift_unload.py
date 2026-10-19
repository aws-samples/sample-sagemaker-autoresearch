from unittest.mock import MagicMock

import pytest

from sm_research import redshift_unload, slicing


def _cfg():
    return {
        "aws": {"region": "us-east-1"},
        "data": {
            "redshift": {"workgroup_or_cluster": "wg", "database": "db"},
            "allowed": [{"schema": "public", "table": "events"}],
        },
        "sagemaker": {"s3_bucket": "bk", "role_arn": "arn:aws:iam::1:role/r"},
    }


def _spec():
    return {"tables": ["public.events"], "columns": ["a", "b"], "filter": None,
            "seed": 1, "select_sql": "select a, b from public.events"}


def test_build_unload_sql_wraps_select_and_targets_prefix():
    sql = redshift_unload.build_unload_sql(_spec(), "s3://bk/x/", "arn:aws:iam::1:role/r")
    assert "unload" in sql.lower()
    assert "s3://bk/x/" in sql
    assert "parquet" in sql.lower()


def test_unload_skips_when_cached():
    cfg, spec = _cfg(), _spec()
    s3 = MagicMock()
    s3.list_objects_v2.return_value = {"KeyCount": 3}  # slice already present
    client = MagicMock()
    out = redshift_unload.unload(cfg, client, s3, spec, run_id="run1")
    assert out["cached"] is True
    client.execute_statement.assert_not_called()
    assert out["s3_prefix"] == slicing.s3_prefix("bk", "run1", spec)


def test_unload_runs_when_not_cached():
    cfg, spec = _cfg(), _spec()
    s3 = MagicMock()
    s3.list_objects_v2.return_value = {"KeyCount": 0}
    client = MagicMock()
    client.execute_statement.return_value = {"Id": "sid"}
    client.describe_statement.return_value = {"Status": "FINISHED"}
    client.get_statement_result.return_value = {"Records": [], "ColumnMetadata": []}
    out = redshift_unload.unload(cfg, client, s3, spec, run_id="run1")
    assert out["cached"] is False
    client.execute_statement.assert_called_once()


def test_unload_does_not_fetch_results():
    """UNLOAD produces no result set; GetStatementResult raises on success.

    Regression: the real Data API returned ResourceNotFoundException here, so
    every successful unload was reported as an error.
    """
    cfg, spec = _cfg(), _spec()
    s3 = MagicMock()
    s3.list_objects_v2.return_value = {"KeyCount": 0}
    client = MagicMock()
    client.execute_statement.return_value = {"Id": "sid"}
    client.describe_statement.return_value = {"Status": "FINISHED"}
    client.get_statement_result.side_effect = AssertionError(
        "GetStatementResult must not be called for UNLOAD")
    out = redshift_unload.unload(cfg, client, s3, spec, run_id="run1")
    assert out["cached"] is False
    client.get_statement_result.assert_not_called()


def test_build_unload_sql_wraps_select_so_limit_is_legal():
    """Redshift rejects a top-level LIMIT in UNLOAD; nesting makes it legal."""
    spec = _spec()
    spec["select_sql"] = "select a, b from public.events limit 500"
    sql = redshift_unload.build_unload_sql(spec, "s3://bk/x/", "arn:aws:iam::1:role/r")
    # The caller's SELECT must be nested, not spliced in at top level.
    assert "select * from (select a, b from public.events limit 500)" in sql


def test_build_unload_sql_escapes_quotes_after_wrapping():
    spec = _spec()
    spec["select_sql"] = "select a from public.events where region = 'us-west'"
    sql = redshift_unload.build_unload_sql(spec, "s3://bk/x/", "arn:aws:iam::1:role/r")
    # Single quotes must be doubled so the UNLOAD string literal stays valid.
    assert "''us-west''" in sql


def test_unload_rejects_table_outside_allowlist():
    cfg = _cfg()  # data.allowed only lists public.events
    spec = _spec()
    spec["tables"] = ["public.forbidden"]
    s3 = MagicMock()
    client = MagicMock()
    with pytest.raises(ValueError):
        redshift_unload.unload(cfg, client, s3, spec, run_id="run1")
    client.execute_statement.assert_not_called()


# --- allowlist enforcement against the SQL itself -------------------------

_ALLOWED = {"churn.customers", "churn.labels", "churn.usage_monthly"}


def test_sql_allowlist_blocks_undeclared_table():
    """Regression: spec["tables"] is caller-supplied, so it alone is not a guard.

    A slice could declare an allowed table while select_sql read something else,
    and UNLOAD would happily write that data to S3.
    """
    with pytest.raises(ValueError, match="outside data.allowed"):
        redshift_unload.assert_sql_within_allowlist(
            "select * from secrets.salaries", _ALLOWED)


def test_unload_blocks_sql_reading_outside_allowlist():
    cfg = _cfg()
    cfg["data"]["allowed"] = [{"schema": "churn", "table": "labels"}]
    spec = _spec()
    spec["tables"] = ["churn.labels"]          # declared: allowed
    spec["select_sql"] = "select * from secrets.salaries"   # actual: not allowed
    s3 = MagicMock()
    s3.list_objects_v2.return_value = {"KeyCount": 0}
    client = MagicMock()
    with pytest.raises(ValueError, match="outside data.allowed"):
        redshift_unload.unload(cfg, client, s3, spec, run_id="r1")
    client.execute_statement.assert_not_called()


def test_sql_allowlist_permits_joins_and_subquery_aggregates():
    redshift_unload.assert_sql_within_allowlist(
        "select c.customer_id, u.t from churn.customers c "
        "join (select customer_id, sum(support_tickets) t "
        "from churn.usage_monthly group by customer_id) u "
        "on c.customer_id = u.customer_id", _ALLOWED)


def test_sql_allowlist_permits_ctes_without_demanding_them_as_tables():
    # A WITH alias is not a physical table and must not be checked as one.
    redshift_unload.assert_sql_within_allowlist(
        "with agg as (select customer_id from churn.usage_monthly) "
        "select * from churn.labels l join agg on l.customer_id = agg.customer_id",
        _ALLOWED)


def test_sql_allowlist_blocks_union_exfiltration():
    with pytest.raises(ValueError, match="outside data.allowed"):
        redshift_unload.assert_sql_within_allowlist(
            "select * from churn.labels union all select * from secrets.q", _ALLOWED)


def test_sql_allowlist_ignores_tables_hidden_in_comments_and_literals():
    # Neither of these actually reads secrets.*, so both must be allowed.
    redshift_unload.assert_sql_within_allowlist(
        "select * from churn.labels -- from secrets.z", _ALLOWED)
    redshift_unload.assert_sql_within_allowlist(
        "select * from churn.labels /* from secrets.y */", _ALLOWED)
    redshift_unload.assert_sql_within_allowlist(
        "select * from churn.labels where region = 'from secrets.x'", _ALLOWED)


def test_sql_allowlist_rejects_unqualified_tables():
    # A bare name cannot be matched against schema.table, so fail closed.
    with pytest.raises(ValueError, match="unqualified"):
        redshift_unload.assert_sql_within_allowlist("select * from labels", _ALLOWED)


def test_sql_allowlist_rejects_non_select_statements():
    with pytest.raises(ValueError, match="read-only SELECT"):
        redshift_unload.assert_sql_within_allowlist(
            "select * from churn.labels; drop table churn.labels", _ALLOWED)


def test_sql_allowlist_rejects_sql_with_no_tables():
    with pytest.raises(ValueError, match="no tables"):
        redshift_unload.assert_sql_within_allowlist("select 1", _ALLOWED)


def test_sql_allowlist_is_case_insensitive():
    redshift_unload.assert_sql_within_allowlist("SELECT * FROM CHURN.LABELS", _ALLOWED)


def _restricted_cfg():
    cfg = _cfg()
    cfg["data"]["allowed"] = [
        {"schema": "public", "table": "events", "columns": ["id", "a"]},
        {"schema": "public", "table": "labels"},
    ]
    return cfg


def _describe(columns):
    client = MagicMock()
    client.execute_statement.return_value = {"Id": "sid"}
    client.describe_statement.return_value = {"Status": "FINISHED"}
    client.get_statement_result.return_value = {
        "Records": [[{"stringValue": "public"}, {"stringValue": "events"}, {"stringValue": c}]
                    for c in columns], "ColumnMetadata": []}
    return client


def test_forbidden_columns_only_for_column_restricted_tables():
    client = _describe(["id", "a", "secret"])
    rules = redshift_unload.forbidden_columns(_restricted_cfg(), client)
    assert rules == {"public.events": {"secret"}}
    client.execute_statement.assert_called_once()
    assert "information_schema.columns" in client.execute_statement.call_args.kwargs["Sql"]


ALLOWED = {"public.events", "public.labels"}
RULES = {"public.events": {"secret", "late_payments"}}


@pytest.mark.parametrize("sql", [
    "select e.id, e.a, count(*) as n from public.events e group by e.id, e.a",
    "select e.id, e.a * 2 as a2, l.y from public.events e join public.labels l on l.id = e.id",
    "select * from public.labels",  # unrestricted table: star is fine
])
def test_column_rules_allow_explicit_allowed_columns(sql):
    redshift_unload.assert_sql_within_allowlist(sql, ALLOWED, RULES)


@pytest.mark.parametrize("sql,needle", [
    # Regression: a bare star inside a window-function subquery read forbidden
    # columns even though the outer query projected only allowed ones.
    ("select id, max(a) from (select *, max(a) over (partition by id) m "
     "from public.events) s group by id", r"\* projection"),
    ("select e.* from public.events e", r"\* projection"),
    ("select e.id, l.* from public.events e join public.labels l on l.id = e.id", r"\* projection"),
    ("select id, sum(late_payments) from public.events group by id", "late_payments"),
    ('select "SECRET" from public.events', "secret"),
])
def test_column_rules_reject_star_and_forbidden_columns(sql, needle):
    with pytest.raises(ValueError, match=needle):
        redshift_unload.assert_sql_within_allowlist(sql, ALLOWED, RULES)


def test_forbidden_columns_skips_tables_the_sql_does_not_read():
    client = _describe(["id", "a", "secret"])
    assert redshift_unload.forbidden_columns(_restricted_cfg(), client, {"public.labels"}) == {}
    client.execute_statement.assert_not_called()


def test_unload_checks_columns_before_running():
    cfg = _restricted_cfg()
    spec = {"tables": ["public.events"], "seed": 1,
            "select_sql": "select * from (select * from public.events) s"}
    s3 = MagicMock()
    s3.list_objects_v2.return_value = {"KeyCount": 0}
    client = _describe(["id", "a", "secret"])
    with pytest.raises(ValueError, match="projection"):
        redshift_unload.unload(cfg, client, s3, spec, run_id="r")
    # Only the column lookup ran; the UNLOAD itself never did.
    assert all("unload" not in c.kwargs["Sql"].lower()
               for c in client.execute_statement.call_args_list)
