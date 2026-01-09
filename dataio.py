
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple

__all__ = ["load_dataset"]

# Strings interpreted as missing values
NA_TOKENS = {"NaN", "nan", "NA", "N/A", "None", ""}


def _first_nonempty_cols(path: str) -> int:
    """Return the number of columns on the first non-empty line of a text file.

    This helper counts the whitespace-separated tokens on the first non-empty
    line.  It is used to infer the expected number of columns in the `.data`
    and `.solution` files.
    """
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                return len(line.split())
    return 0


def _read_types_safe(type_file: str, n_features: int) -> List[str]:
    """Read feature types from a `.type` file.

    If the file does not contain exactly ``n_features`` tokens, return a default
    list of length ``n_features`` with the value "Numerical" for each feature.
    """
    tokens: List[str] = []
    with open(type_file, "r") as f:
        for line in f:
            tokens.extend(line.split())
    tokens = [t.strip() for t in tokens if t.strip()]
    if len(tokens) != n_features:
        return ["Numerical"] * n_features
    return tokens


def _read_matrix_ragged(path: str, n_cols: int) -> pd.DataFrame:
    """Read a whitespace-delimited matrix from a text file with variable row lengths.

    Each line of ``path`` is split on whitespace.  Missing values encoded by
    tokens in ``NA_TOKENS`` are replaced with ``numpy.nan``.  If a row has
    fewer than ``n_cols`` elements, it is padded with ``numpy.nan``; if it has
    more, it is truncated.  The resulting matrix is returned as a DataFrame.
    ``pandas.to_numeric`` is used to convert each column to numeric dtype when
    possible.  Columns that cannot be converted remain of dtype ``object``.
    """
    rows: list[list[object]] = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                row = [np.nan] * n_cols
            else:
                # Replace missing tokens with NaN
                row = [np.nan if p in NA_TOKENS else p for p in parts]
                # Pad or truncate to the expected number of columns
                if len(row) < n_cols:
                    row += [np.nan] * (n_cols - len(row))
                elif len(row) > n_cols:
                    row = row[:n_cols]
            rows.append(row)
    df = pd.DataFrame(rows)
    # Convert columns to numeric when possible
    for j in range(df.shape[1]):
        df[j] = pd.to_numeric(df[j], errors="ignore")
    return df


def load_dataset(base_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Load a dataset consisting of `.data`, `.solution` and `.type` files.

    Parameters
    ----------
    base_path : str
        Path prefix to the dataset files (without extension).  For example,
        ``base_path="/data/dataset_A"`` will load ``/data/dataset_A.data``,
        ``/data/dataset_A.solution`` and ``/data/dataset_A.type``.

    Returns
    -------
    X : pandas.DataFrame
        The feature matrix.  Columns are indexed by integer position starting
        at zero.
    y : pandas.DataFrame
        The target matrix.  Columns are indexed by integer position starting at
        zero.  Even for a single-target task the return type is DataFrame.
    types : list of str
        A list describing the type of each feature.  Values are either
        "Numerical" or "Categorical" (case-insensitive).  If the `.type` file
        cannot be parsed, the default is to mark all features as numerical.
    """
    data_file = f"{base_path}.data"
    type_file = f"{base_path}.type"
    sol_file = f"{base_path}.solution"
    # Infer matrix dimensions from the first non-empty lines
    n_x = _first_nonempty_cols(data_file)
    n_y = _first_nonempty_cols(sol_file)
    types = _read_types_safe(type_file, n_x)
    X = _read_matrix_ragged(data_file, n_x)
    y = _read_matrix_ragged(sol_file, n_y)
    # Align lengths for safety
    if len(X) != len(y):
        m = min(len(X), len(y))
        X = X.iloc[:m].reset_index(drop=True)
        y = y.iloc[:m].reset_index(drop=True)
    return X, y, types
