"""Content-addressable hashing for data slices so UNLOADs can be cached."""
import hashlib
import json


def slice_hash(spec: dict) -> str:
    # sort_keys + sorted lists make the digest order-independent
    normalized = _normalize(spec)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _normalize(value):
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    if isinstance(value, list):
        return sorted((_normalize(v) for v in value), key=lambda x: json.dumps(x, sort_keys=True))
    return value


def s3_prefix(bucket: str, run_id: str, spec: dict) -> str:
    return f"s3://{bucket}/sm-research/{run_id}/slices/{slice_hash(spec)}/"
