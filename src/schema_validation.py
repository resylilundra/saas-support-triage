import json
from pathlib import Path
from typing import Any

from jsonschema import ValidationError, validate


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "data" / "schema.json"

with SCHEMA_PATH.open(encoding="utf-8") as schema_file:
    SCHEMA: dict[str, Any] = json.load(schema_file)


def validate_extraction(payload: str) -> dict[str, Any]:
    """Parse a model response and return it only if it matches the schema."""
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("Model response is not valid JSON.") from error

    try:
        validate(instance=data, schema=SCHEMA)
    except ValidationError as error:
        raise ValueError(
            f"Model response does not match schema: {error.message}"
        ) from error

    return data
