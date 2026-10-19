"""Redshift Data API discovery: row counts + samples, scoped to the allowlist."""
import argparse
import time

import boto3

from sm_research import config, contract


def _ident(name: str) -> str:
    # Quoted identifier; doubling embedded quotes keeps a name from closing it.
    return '"' + str(name).replace('"', '""') + '"'


def _qual(schema: str, table: str) -> str:
    return f"{_ident(schema)}.{_ident(table)}"


def _projection(columns) -> str:
    # "*" (or a list containing "*") means all columns are allowed → wildcard.
    if columns == "*":
        return "*"
    if isinstance(columns, list):
        if "*" in columns:
            return "*"
        return ", ".join(_ident(c) for c in columns)
    return "*"


def build_discovery_sql(allowed: list, target: str, sample_rows: int) -> list[dict]:
    specs = []
    sample_rows = int(sample_rows)
    for t in allowed:
        fq = _qual(t["schema"], t["table"])
        projection = _projection(t.get("columns", "*"))
        # fq and projection are escaped identifiers; sample_rows is an int.
        specs.append({
            "table": f'{t["schema"]}.{t["table"]}',
            "count_sql": f"select count(*) as count from {fq}",  # nosec B608
            "sample_sql": f"select {projection} from {fq} limit {sample_rows}",  # nosec B608
        })
    return specs


def _execute(client, cfg_rs: dict, region: str, sql: str,
             fetch_results: bool = True) -> dict | None:
    """Run a statement and wait for it to finish.

    Set fetch_results=False for statements that produce no result set (UNLOAD,
    COPY, DDL). Calling GetStatementResult on those raises
    ResourceNotFoundException even though the statement succeeded.
    """
    kwargs = {"Database": cfg_rs["database"], "Sql": sql}
    if "secret_arn" in cfg_rs and cfg_rs.get("secret_arn"):
        kwargs["SecretArn"] = cfg_rs["secret_arn"]
    wg = cfg_rs["workgroup_or_cluster"]
    # Serverless workgroups use WorkgroupName; provisioned clusters ClusterIdentifier.
    if cfg_rs.get("serverless", True):
        kwargs["WorkgroupName"] = wg
    else:
        kwargs["ClusterIdentifier"] = wg
    started = client.execute_statement(**kwargs)
    sid = started["Id"]
    while True:
        desc = client.describe_statement(Id=sid)
        status = desc["Status"]
        if status == "FINISHED":
            break
        if status in ("FAILED", "ABORTED"):
            raise RuntimeError(desc.get("Error", f"statement {status}"))
        time.sleep(1)
    if not fetch_results:
        return None
    return client.get_statement_result(Id=sid)


def _rows_to_dicts(result: dict) -> list[dict]:
    cols = [c["name"] for c in result["ColumnMetadata"]]
    out = []
    for rec in result["Records"]:
        row = {}
        for name, cell in zip(cols, rec):
            row[name] = next(iter(cell.values())) if cell else None
        out.append(row)
    return out


def run(cfg: dict, client, sample_rows: int) -> dict:
    rs = cfg["data"]["redshift"]
    region = cfg["aws"]["region"]
    allowed = cfg["data"]["allowed"]
    target = cfg["data"]["target"]
    tables = []
    for spec in build_discovery_sql(allowed, target, sample_rows):
        count_res = _execute(client, rs, region, spec["count_sql"])
        row_count = _rows_to_dicts(count_res)[0]["count"]
        sample_res = _execute(client, rs, region, spec["sample_sql"])
        sample = _rows_to_dicts(sample_res)
        columns = [c["name"] for c in sample_res["ColumnMetadata"]]
        tables.append({"table": spec["table"], "row_count": row_count,
                       "columns": columns, "sample": sample})
    return {"tables": tables, "target": target}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-rows", type=int, default=10)
    args = parser.parse_args()
    try:
        cfg = config.load(args.config)
        errs = config.validate(cfg)
        if errs:
            contract.emit(contract.err("; ".join(errs)))
        session = boto3.Session(profile_name=cfg["aws"]["profile"],
                                region_name=cfg["aws"]["region"])
        client = session.client("redshift-data")
        contract.emit(contract.ok(run(cfg, client, args.sample_rows)))
    except Exception as e:  # noqa: BLE001 — surface AWS error verbatim
        contract.emit(contract.err(str(e)))


if __name__ == "__main__":
    main()
