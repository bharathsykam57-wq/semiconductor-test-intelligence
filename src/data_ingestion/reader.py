from pathlib import Path

import polars as pl


def load_parquet(file_path: str | Path) -> pl.DataFrame:
    """
    Load semiconductor test measurements from a Parquet file.

    Parameters
    ----------
    file_path:
        Path to the Parquet dataset.

    Returns
    -------
    pl.DataFrame
        Loaded semiconductor test measurements.
    """

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Parquet file not found: {path}"
        )

    return pl.read_parquet(path)