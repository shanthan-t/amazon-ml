"""Chunked TSV readers for the challenge source and ground-truth files."""

from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd


SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")
GROUND_TRUTH_COLUMNS = ("source1_entity_id", "matched_entity_ids")
DEFAULT_CHUNK_SIZE = 50_000


def iter_tsv_chunks(
    path: str | Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    expected_columns: Sequence[str] | None = None,
    usecols: Sequence[str] | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield UTF-8 TSV chunks without loading the complete file into memory.

    Empty fields remain empty strings. Callers retain the original column values;
    normalization should be stored separately when both forms are needed.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")

    path = Path(path)
    header = pd.read_csv(path, sep="\t", encoding="utf-8", nrows=0)
    actual_columns = tuple(header.columns)
    if expected_columns is not None and actual_columns != tuple(expected_columns):
        raise ValueError(
            f"Unexpected columns in {path}: {actual_columns}; "
            f"expected {tuple(expected_columns)}"
        )
    if usecols is not None:
        unknown = set(usecols) - set(actual_columns)
        if unknown:
            raise ValueError(f"Unknown columns in {path}: {sorted(unknown)}")

    with pd.read_csv(
        path,
        sep="\t",
        encoding="utf-8",
        dtype=str,
        keep_default_na=False,
        chunksize=chunk_size,
        usecols=usecols,
    ) as reader:
        for chunk in reader:
            yield chunk


def iter_source_chunks(
    path: str | Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    usecols: Sequence[str] | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield source-table chunks with the challenge's four-column schema."""
    return iter_tsv_chunks(
        path,
        chunk_size=chunk_size,
        expected_columns=SOURCE_COLUMNS,
        usecols=usecols,
    )


def iter_ground_truth_chunks(
    path: str | Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[pd.DataFrame]:
    """Yield ground-truth chunks using its two-column schema."""
    return iter_tsv_chunks(
        path,
        chunk_size=chunk_size,
        expected_columns=GROUND_TRUTH_COLUMNS,
    )
