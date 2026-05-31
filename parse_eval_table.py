"""Parse a whitespace-formatted eval results table into a pandas DataFrame.

Usage:
    python parse_eval_table.py                  # reads from stdin
    python parse_eval_table.py table.txt        # reads from file

Or import and call `parse_table(text)` programmatically.
"""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

import pandas as pd


def parse_table(text: str) -> pd.DataFrame:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    header = lines[0].split()
    rows = [ln.split() for ln in lines[1:]]

    bad = [i for i, r in enumerate(rows) if len(r) != len(header)]
    if bad:
        raise ValueError(
            f"Row(s) {bad} have a different number of fields than the header "
            f"({len(header)}). First offending row has {len(rows[bad[0]])} fields."
        )

    df = pd.DataFrame(rows, columns=header)

    for col in df.columns:
        s = df[col]
        if s.str.endswith("%").all():
            df[col] = s.str.rstrip("%").astype(float) / 100.0
            continue
        try:
            df[col] = pd.to_numeric(s)
        except (ValueError, TypeError):
            pass

    return df


def main() -> None:
    if len(sys.argv) > 1:
        text = Path(sys.argv[1]).read_text()
    else:
        text = sys.stdin.read()
    df = parse_table(text)
    print(df)
    print(f"\nshape: {df.shape}")
    print(f"dtypes:\n{df.dtypes}")


if __name__ == "__main__":
    main()
