"""UNLOAD an allowed data slice to S3 as Parquet, cached by content hash."""
import argparse
import json
import re

import boto3

from sm_research import config, contract, slicing
from sm_research.redshift_discover import _execute

# Table references after FROM / JOIN. Matches an optional schema qualifier and
# tolerates double-quoted identifiers, which Redshift accepts.
_TABLE_REF = re.compile(
    r'\b(?:from|join)\s+("?[A-Za-z_][\w$]*"?)\s*(?:\.\s*("?[A-Za-z_][\w$]*"?))?',
    re.IGNORECASE,
)
# Statements other than a single read. UNLOAD only accepts a SELECT, but a
# rejected statement here is a clearer failure than a Redshift syntax error.
_FORBIDDEN = re.compile(
    r'\b(insert|update|delete|drop|alter|create|grant|revoke|truncate|copy|unload|vacuum)\b',
    re.IGNORECASE,
)


def _strip_sql_noise(sql: str) -> str:
    """Remove comments and string literals so they cannot hide table refs."""
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"'(?:''|[^'])*'", "''", sql)
    return sql


def referenced_tables(select_sql: str) -> tuple[set[str], set[str]]:
    """Return (qualified, unqualified) table names read by the SQL.

    Common table expressions are resolved and excluded: a WITH alias is not a
    physical table and must not be demanded of the allowlist.
    """
    cleaned = _strip_sql_noise(select_sql)
    cte_names = {
        m.group(1).strip('"').lower()
        for m in re.finditer(r'(?:\bwith\b|,)\s*("?[A-Za-z_][\w$]*"?)\s+as\s*\(',
                             cleaned, re.IGNORECASE)
    }
    qualified: set[str] = set()
    unqualified: set[str] = set()
    for schema_or_table, maybe_table in _TABLE_REF.findall(cleaned):
        first = schema_or_table.strip('"').lower()
        if maybe_table:
            qualified.add(f"{first}.{maybe_table.strip('\"').lower()}")
        elif first not in cte_names:
            # A bare name is either a CTE (fine) or an unqualified physical
            # table, which cannot be matched against schema.table safely.
            unqualified.add(first)
    return qualified, unqualified


# Projection stars (SELECT *, SELECT t.*, ", t.*") - not count(*) or multiplication.
_PROJECTION_STAR = re.compile(r'\bselect\s+(?:distinct\s+)?(?:"?[\w$]+"?\.)?\*|,\s*"?[\w$]+"?\.\*',
                              re.IGNORECASE)


def assert_sql_within_allowlist(select_sql: str, allowed: set[str],
                                column_rules: dict[str, set[str]] | None = None) -> None:
    """Fail closed unless every table the SQL reads is in the allowlist.

    Validating spec["tables"] alone is not enough: that list is supplied by the
    caller, so a slice could declare an allowed table while its select_sql read
    something else entirely and UNLOAD it to S3.

    column_rules maps a column-restricted table ("schema.table") to the names of
    its columns that are NOT allowlisted. For any such table the SQL reads, a
    projection star or any use of a forbidden column name is rejected: Redshift
    reads every column a star expands to, even inside a subquery whose outer
    query later drops them.
    """
    if _FORBIDDEN.search(_strip_sql_noise(select_sql)):
        raise ValueError(
            "select_sql must be a single read-only SELECT; found a forbidden "
            "statement keyword"
        )
    qualified, unqualified = referenced_tables(select_sql)
    if unqualified:
        raise ValueError(
            "select_sql references unqualified table(s) "
            f"{sorted(unqualified)}; use schema.table so they can be checked "
            "against data.allowed"
        )
    if not qualified:
        raise ValueError("select_sql references no tables; refusing to UNLOAD")
    outside = {t for t in qualified if t not in allowed}
    if outside:
        raise ValueError(
            f"select_sql reads table(s) outside data.allowed: {sorted(outside)}"
        )
    restricted = {t: cols for t, cols in (column_rules or {}).items() if t in qualified}
    if not restricted:
        return
    cleaned = _strip_sql_noise(select_sql)
    if _PROJECTION_STAR.search(cleaned):
        raise ValueError(
            f"select_sql uses a * projection while reading column-restricted table(s) "
            f"{sorted(restricted)}; list the allowed columns explicitly (a * reads every "
            "column, even inside a subquery)"
        )
    words = {w.lower() for w in re.findall(r'[A-Za-z_][\w$]*', cleaned.replace('"', ' '))}
    used = sorted({c for cols in restricted.values() for c in cols} & words)
    if used:
        raise ValueError(f"select_sql references column(s) outside data.allowed: {used}")


def forbidden_columns(cfg: dict, client, tables: set[str] | None = None) -> dict[str, set[str]]:
    """For each allowlisted table that lists specific columns (limited to `tables`
    when given), the names of its real columns that are not allowlisted.

    One ExecuteStatement on information_schema covers every such table: on a cold
    Serverless workgroup DescribeTable times out ("Metadata request took too
    long"), while _execute waits for the workgroup to resume.
    """
    restricted = {}
    for t in cfg["data"]["allowed"]:
        cols = t.get("columns", "*")
        name = f"{t['schema']}.{t['table']}".lower()
        if cols == "*" or not isinstance(cols, list) or "*" in cols:
            continue
        if tables is not None and name not in tables:
            continue
        restricted[name] = {c.lower() for c in cols}
    if not restricted:
        return {}
    conds = " or ".join(
        "(table_schema = '{}' and table_name = '{}')".format(
            *(v.replace("'", "''") for v in n.split(".", 1)))
        for n in sorted(restricted))
    # conds holds only quote-escaped literals from the allowlist.
    sql = f"select table_schema, table_name, column_name from information_schema.columns where {conds}"  # nosec B608
    result = _execute(client, cfg["data"]["redshift"], cfg["aws"]["region"], sql)
    actual: dict[str, set[str]] = {n: set() for n in restricted}
    for rec in result.get("Records", []):
        schema, table, col = (str(next(iter(c.values()))).lower() for c in rec)
        actual.setdefault(f"{schema}.{table}", set()).add(col)
    missing = [n for n in restricted if not actual[n]]
    if missing:
        # Fail closed: without the real column list the check cannot run.
        raise ValueError(f"could not read columns of {missing}")
    return {n: actual[n] - restricted[n] for n in restricted}


def build_unload_sql(spec: dict, s3_prefix: str, iam_role: str) -> str:
    # Redshift rejects a top-level LIMIT inside UNLOAD ("Limit clause is not
    # supported"), which would break any sampled slice. Wrapping the caller's
    # SELECT in a subquery makes the LIMIT legal and is a no-op otherwise.
    # select_sql is allowlist-checked by the caller and quote-escaped here.
    inner = f"select * from ({spec['select_sql']}) as _slice".replace("'", "''")  # nosec B608
    return (
        f"unload ('{inner}') "  # nosec B608
        f"to '{s3_prefix}' "
        f"iam_role '{iam_role}' "
        f"format as parquet allowoverwrite parallel off"
    )


def _prefix_has_objects(s3, s3_uri: str) -> bool:
    without = s3_uri[len("s3://"):]
    bucket, _, key = without.partition("/")
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1)
    return resp.get("KeyCount", 0) > 0


def unload(cfg: dict, client, s3, spec: dict, run_id: str = "default") -> dict:
    allowed = {f"{t['schema']}.{t['table']}".lower() for t in cfg["data"]["allowed"]}
    bad = {t for t in spec.get("tables", []) if t.lower() not in allowed}
    if bad:
        raise ValueError(f"slice references tables outside data.allowed: {sorted(bad)}")
    # The declared list above is caller-supplied, so also check the SQL itself.
    qualified, _ = referenced_tables(spec["select_sql"])
    assert_sql_within_allowlist(spec["select_sql"], allowed,
                                forbidden_columns(cfg, client, qualified & allowed))
    bucket = cfg["sagemaker"]["s3_bucket"]
    s3_prefix = slicing.s3_prefix(bucket, run_id, spec)
    h = slicing.slice_hash(spec)
    if _prefix_has_objects(s3, s3_prefix):
        return {"s3_prefix": s3_prefix, "cached": True, "slice_hash": h, "run_id": run_id}
    sql = build_unload_sql(spec, s3_prefix, cfg["sagemaker"]["role_arn"])
    # UNLOAD returns no result set — fetching one would raise on success.
    _execute(client, cfg["data"]["redshift"], cfg["aws"]["region"], sql,
             fetch_results=False)
    return {"s3_prefix": s3_prefix, "cached": False, "slice_hash": h, "run_id": run_id}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--spec-json", required=True)
    parser.add_argument("--run-id", default="default")
    args = parser.parse_args()
    try:
        cfg = config.load(args.config)
        spec = json.loads(args.spec_json)
        session = boto3.Session(profile_name=cfg["aws"]["profile"],
                                region_name=cfg["aws"]["region"])
        client = session.client("redshift-data")
        s3 = session.client("s3")
        contract.emit(contract.ok(unload(cfg, client, s3, spec, args.run_id)))
    except Exception as e:  # noqa: BLE001
        contract.emit(contract.err(str(e)))


if __name__ == "__main__":
    main()
