from unittest.mock import MagicMock

from sm_research import redshift_discover


def test_build_discovery_sql_scopes_to_allowlist():
    allowed = [{"schema": "public", "table": "events", "columns": ["a", "b"]}]
    specs = redshift_discover.build_discovery_sql(allowed, "y", sample_rows=5)
    assert len(specs) == 1
    assert 'public"."events"' in specs[0]["count_sql"]
    assert "limit 5" in specs[0]["sample_sql"].lower()


def test_sample_sql_projects_only_allowed_columns():
    allowed = [{"schema": "public", "table": "events", "columns": ["a", "b"]}]
    specs = redshift_discover.build_discovery_sql(allowed, "y", sample_rows=5)
    s = specs[0]["sample_sql"]
    assert '"a"' in s and '"b"' in s
    assert "*" not in s


def test_sample_sql_wildcard_columns_selects_all():
    allowed = [{"schema": "public", "table": "events", "columns": "*"}]
    specs = redshift_discover.build_discovery_sql(allowed, "y", sample_rows=5)
    assert "select *" in specs[0]["sample_sql"].lower()


def test_identifiers_escape_embedded_quotes():
    allowed = [{"schema": 'pub"lic', "table": 'ev"; drop table x; --', "columns": ['a"b']}]
    specs = redshift_discover.build_discovery_sql(allowed, "y", sample_rows=5)
    assert specs[0]["count_sql"] == 'select count(*) as count from "pub""lic"."ev""; drop table x; --"'
    assert specs[0]["sample_sql"].startswith('select "a""b" from ')


def test_run_aggregates_counts_and_samples():
    cfg = {
        "aws": {"region": "us-east-1"},
        "data": {
            "redshift": {"workgroup_or_cluster": "wg", "database": "db"},
            "allowed": [{"schema": "public", "table": "events", "columns": ["a"]}],
            "target": "y",
        },
    }
    client = MagicMock()
    client.execute_statement.return_value = {"Id": "sid"}
    client.describe_statement.return_value = {"Status": "FINISHED"}
    # First get_statement_result = count query; second = sample query
    client.get_statement_result.side_effect = [
        {"Records": [[{"longValue": 100}]],
         "ColumnMetadata": [{"name": "count"}]},
        {"Records": [[{"stringValue": "v1"}]],
         "ColumnMetadata": [{"name": "a"}]},
    ]
    out = redshift_discover.run(cfg, client, sample_rows=5)
    assert out["target"] == "y"
    assert out["tables"][0]["row_count"] == 100
    assert out["tables"][0]["sample"] == [{"a": "v1"}]
