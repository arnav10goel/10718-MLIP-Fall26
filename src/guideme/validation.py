"""Small shared validators for runtime metadata."""

import math
import re
from uuid import UUID


def require_keys(record, fields):
    if not isinstance(record, dict) or set(record) != set(fields):
        raise ValueError("Record has missing or unsupported fields")


def require_uuid(value):
    if not isinstance(value, str) or str(UUID(value)) != value:
        raise ValueError("Expected a canonical UUID identity")


def require_sha256(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected a lowercase SHA-256 identity")


def require_nonnegative_number(value):
    try:
        valid = type(value) in (int, float) and value >= 0 and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("Expected a finite nonnegative number")
    return value
