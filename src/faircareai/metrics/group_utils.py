"""Utilities for per-group metric computation.

Provides common scaffolding for iterating over demographic groups
and computing metrics, reducing code duplication.
"""

from __future__ import annotations

from typing import Any

import polars as pl


def get_unique_groups(df: pl.DataFrame, group_col: str) -> list[Any]:
    """Get sorted list of unique groups from a column.

    Args:
        df: DataFrame containing the group column
        group_col: Column name for grouping variable

    Returns:
        Sorted list of unique non-null group values
    """
    return df[group_col].drop_nulls().unique().sort().to_list()


def filter_to_group(df: pl.DataFrame, group_col: str, group: Any) -> pl.DataFrame:
    """Filter DataFrame to a single group.

    Args:
        df: DataFrame to filter
        group_col: Column name for grouping variable
        group: Group value to filter to

    Returns:
        Filtered DataFrame containing only rows for the specified group
    """
    return df.filter(pl.col(group_col) == group)


def determine_reference_group(
    groups: list[Any],
    df: pl.DataFrame,
    group_col: str,
    reference: str | None = None,
) -> Any:
    """Determine the reference group for comparisons.

    Args:
        groups: List of unique groups
        df: DataFrame containing the group column
        group_col: Column name for grouping variable
        reference: User-specified reference group (if any)

    Returns:
        Reference group value (specified, or largest group by count)
    """
    if reference is not None and reference in groups:
        return reference

    # Default to largest group
    group_counts = df.group_by(group_col).len().sort("len", descending=True)
    return group_counts[group_col][0]
