from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DATE = "Date"

def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_excel(path, sheet_name="Sheet1") if path.suffix.lower() == ".xlsx" else pd.read_csv(path)
    df[DATE] = pd.to_datetime(df[DATE])
    return df.sort_values(DATE).reset_index(drop=True)