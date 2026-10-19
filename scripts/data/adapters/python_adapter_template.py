"""Copy this file and adapt ``extract`` for a complex dataset."""

from collections.abc import Mapping
from typing import Any


def extract(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return one prompt/messages payload plus optional source and category."""
    return {
        "prompt": row["prompt"],
        "source": row.get("source"),
        "category": row.get("category"),
    }
