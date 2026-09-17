"""Standalone check: verify the SQLite database, schema, and probe row.

Usage: .venv/bin/python -m scripts.check_db
"""

import json

from app.db import setup_schema, test_connection


def main() -> None:
    setup_schema()
    result = test_connection()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
