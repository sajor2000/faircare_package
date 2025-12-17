"""
FairCareAI Great Tables Components

Publication-quality tables for governance reporting.

Methodology: Van Calster et al. (2025), CHAI RAIC Checkpoint 1.
"""

from typing import Any

import polars as pl

from .themes import SEMANTIC_COLORS, TYPOGRAPHY


def create_executive_scorecard(
    pass_count: int,
    warn_count: int,
    flag_count: int,
    n_samples: int,
    n_groups: int,
    model_name: str = "Model",
) -> Any:
    """Create executive summary scorecard table.

    Uses CHAI-compliant advisory terminology:
    - "Flag" instead of "Fail" (informational, not blocking)
    """
    try:
        from great_tables import GT
    except ImportError as err:
        raise ImportError("Install great-tables: pip install great-tables") from err

    df = pl.DataFrame(
        {
            "Metric": ["PASS", "REVIEW", "FLAG"],
            "Count": [pass_count, warn_count, flag_count],
        }
    )

    return (
        GT(df.to_pandas())
        .tab_header(title=f"Equity Audit: {model_name}")
        .tab_source_note(f"N = {n_samples:,} | {n_groups} groups | Advisory guidance")
    )


def create_plain_language_summary(
    pass_count: int,
    warn_count: int,
    flag_count: int,
    worst_group: str,
    worst_metric: str,
    worst_value: float,
) -> str:
    """Generate plain-language summary for stakeholders.

    Uses neutral terminology for threshold-based results.
    """
    if flag_count > 0:
        status = "REVIEW SUGGESTED"
        color = SEMANTIC_COLORS["fail"]
        advisory = "Disparities flagged for clinical review"
    elif warn_count > 0:
        status = "CONSIDERATIONS NOTED"
        color = SEMANTIC_COLORS["warn"]
        advisory = "Some metrics may warrant discussion"
    else:
        status = "NO FLAGS"
        color = SEMANTIC_COLORS["pass"]
        advisory = "No significant disparities detected at current thresholds"

    return f"""
    <div style="font-family: {TYPOGRAPHY["data_font"]}; padding: 24px;">
        <div style="background: {color}; color: white; padding: 12px 24px;
                    border-radius: 4px; display: inline-block; font-weight: bold;">
            {status}
        </div>
        <p style="color: #666; font-size: 16px; margin-top: 8px;">{advisory}</p>
        <p>Largest disparity: <b>{worst_metric}</b> for <b>{worst_group}</b>
           ({abs(worst_value) * 100:.1f}% difference from reference)</p>
        <p style="font-size: 16px; color: #888; font-style: italic;">
            Advisory guidance — final deployment decisions rest with clinical stakeholders
        </p>
    </div>
    """
