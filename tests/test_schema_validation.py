import json

import pytest

from src.schema_validation import validate_extraction


VALID = {
    "language": "de",
    "issue_type": "billing",
    "priority": "normal",
    "product_area": "billing",
    "summary": "The August invoice shows an unexpected tax rate.",
}


def test_accepts_valid_extraction():
    assert validate_extraction(json.dumps(VALID)) == VALID


def test_rejects_invalid_json():
    with pytest.raises(ValueError, match="not valid JSON"):
        validate_extraction('{"language":')


@pytest.mark.parametrize(
    "invalid",
    [
        {**VALID, "priority": "very urgent"},
        {**VALID, "product_area": "billing_portal"},
        {**VALID, "summary": None},
        {**VALID, "summary": "x" * 201},
        {key: value for key, value in VALID.items() if key != "language"},
        {**VALID, "escalation_required": True},
    ],
)
def test_rejects_schema_violations(invalid):
    with pytest.raises(ValueError, match="does not match schema"):
        validate_extraction(json.dumps(invalid))
