"""Standard JSON result envelope for all sm_research scripts."""
import json
import sys


def ok(data: dict) -> dict:
    return {"status": "ok", "data": data, "error": None}


def err(message: str) -> dict:
    return {"status": "error", "data": {}, "error": message}


def emit(result: dict):
    """Print the result as JSON and exit 0 on ok, 1 on error."""
    print(json.dumps(result))
    sys.exit(0 if result.get("status") == "ok" else 1)
