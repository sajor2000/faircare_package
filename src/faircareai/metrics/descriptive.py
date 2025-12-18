"""
FairCareAI Descriptive Statistics Module

Compute Table 1 cohort summaries for governance reports.
Provides standardized descriptive statistics by sensitive attribute.

Methodology: CHAI RAIC Checkpoint 1 (population characteristics).
"""

from typing import Any, cast

import numpy as np
import polars as pl
from scipy import stats


def _pivot_compat(
    df: pl.DataFrame,
    *,
    index: str,
    columns: str,
    values: str,
) -> pl.DataFrame:
    pivot = cast(Any, df).pivot
    try:
        result = pivot(columns=columns, index=index, values=values)
    except TypeError:
        result = pivot(on=columns, index=index, values=values)
    return cast(pl.DataFrame, result)


def compute_cohort_summary(
    df: pl.DataFrame,
    y_true_col: str,
    y_prob_col: str,
    sensitive_attrs: dict[str, dict],
) -> dict[str, Any]:
    """Compute comprehensive cohort summary for Table 1.

    Args:
        df: Polars DataFrame with patient data.
        y_true_col: Column name for true labels.
        y_prob_col: Column name for predicted probabilities.
        sensitive_attrs: Dict of sensitive attribute configurations.

    Returns:
        Dict containing:
        - cohort_overview: n_total, n_positive, prevalence
        - attribute_distributions: N, %, missing per group
        - outcome_by_attribute: rate, rate_ratio, CI per group
        - prediction_distribution: mean, SD, percentiles
        - prediction_by_attribute: score distribution per group
    """
    results: dict[str, Any] = {}

    # === Cohort Overview ===
    n_total = len(df)
    n_positive = df[y_true_col].sum()
    prevalence = n_positive / n_total if n_total > 0 else 0.0

    results["cohort_overview"] = {
        "n_total": int(n_total),
        "n_positive": int(n_positive),
        "n_negative": int(n_total - n_positive),
        "prevalence": float(prevalence),
        "prevalence_pct": f"{prevalence * 100:.1f}%",
    }

    # === Prediction Distribution (Overall) ===
    y_prob = df[y_prob_col].drop_nulls()
    if len(y_prob) > 0:
        prob_np = y_prob.to_numpy()
        results["prediction_distribution"] = {
            "mean": float(np.mean(prob_np)),
            "std": float(np.std(prob_np)),
            "median": float(np.median(prob_np)),
            "min": float(np.min(prob_np)),
            "max": float(np.max(prob_np)),
            "percentile_25": float(np.percentile(prob_np, 25)),
            "percentile_75": float(np.percentile(prob_np, 75)),
            "percentile_90": float(np.percentile(prob_np, 90)),
            "percentile_95": float(np.percentile(prob_np, 95)),
        }
    else:
        results["prediction_distribution"] = {}

    # === Attribute Distributions ===
    attr_distributions: dict[str, dict] = {}
    outcome_by_attr: dict[str, dict] = {}
    prediction_by_attr: dict[str, dict] = {}

    for attr_name, attr_config in sensitive_attrs.items():
        col = attr_config.get("column", attr_name)
        reference = attr_config.get("reference")

        if col not in df.columns:
            continue

        # Group counts and percentages
        group_counts = (
            df.group_by(col)
            .agg(
                pl.len().alias("n"),
                pl.col(y_true_col).sum().alias("n_positive"),
                pl.col(y_prob_col).mean().alias("mean_prob"),
                pl.col(y_prob_col).std().alias("std_prob"),
            )
            .sort(col)
        )

        # Calculate missing
        n_missing = df[col].null_count()
        missing_rate = n_missing / n_total if n_total > 0 else 0.0

        # Build distribution dict
        attr_dist: dict[str, Any] = {
            "n_missing": int(n_missing),
            "missing_rate": float(missing_rate),
            "groups": {},
        }

        # Get reference group outcome rate for ratio calculation
        ref_rate = None
        if reference:
            ref_row = group_counts.filter(pl.col(col) == reference)
            if len(ref_row) > 0:
                ref_n = ref_row["n"][0]
                ref_pos = ref_row["n_positive"][0]
                ref_rate = ref_pos / ref_n if ref_n > 0 else None

        for row in group_counts.iter_rows(named=True):
            group_name = str(row[col]) if row[col] is not None else "Unknown"
            n_group = row["n"]
            n_pos = row["n_positive"]
            pct = n_group / n_total if n_total > 0 else 0.0
            outcome_rate = n_pos / n_group if n_group > 0 else 0.0

            # Calculate rate ratio vs reference
            rate_ratio = None
            if ref_rate is not None and ref_rate > 0:
                rate_ratio = outcome_rate / ref_rate

            # Calculate 95% CI for outcome rate using Wilson score
            ci_low, ci_high = _wilson_ci(n_pos, n_group)

            attr_dist["groups"][group_name] = {
                "n": int(n_group),
                "pct": float(pct),
                "pct_fmt": f"{pct * 100:.1f}%",
            }

            # Outcome by attribute
            if attr_name not in outcome_by_attr:
                outcome_by_attr[attr_name] = {"reference": reference, "groups": {}}

            outcome_by_attr[attr_name]["groups"][group_name] = {
                "n": int(n_group),
                "n_positive": int(n_pos),
                "outcome_rate": float(outcome_rate),
                "outcome_rate_pct": f"{outcome_rate * 100:.1f}%",
                "ci_95_low": float(ci_low),
                "ci_95_high": float(ci_high),
                "ci_95_fmt": f"({ci_low * 100:.1f}%, {ci_high * 100:.1f}%)",
                "rate_ratio": float(rate_ratio) if rate_ratio is not None else None,
                "is_reference": group_name == reference,
            }

            # Prediction by attribute
            if attr_name not in prediction_by_attr:
                prediction_by_attr[attr_name] = {"groups": {}}

            prediction_by_attr[attr_name]["groups"][group_name] = {
                "n": int(n_group),
                "mean_prob": float(row["mean_prob"]) if row["mean_prob"] is not None else None,
                "std_prob": float(row["std_prob"]) if row["std_prob"] is not None else None,
            }

        attr_distributions[attr_name] = attr_dist

    results["attribute_distributions"] = attr_distributions
    results["outcome_by_attribute"] = outcome_by_attr
    results["prediction_by_attribute"] = prediction_by_attr

    return results


def _wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Calculate Wilson score confidence interval for a proportion.

    Args:
        successes: Number of successes.
        n: Total number of trials.
        alpha: Significance level (default 0.05 for 95% CI).

    Returns:
        Tuple of (lower bound, upper bound).
    """
    if n == 0:
        return 0.0, 0.0

    p = successes / n
    z = stats.norm.ppf(1 - alpha / 2)

    denominator = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)

    lower = (center - spread) / denominator
    upper = (center + spread) / denominator

    return max(0.0, lower), min(1.0, upper)


def compute_outcome_rate_statistics(
    df: pl.DataFrame,
    y_true_col: str,
    group_col: str,
    reference: str | None = None,
) -> dict[str, Any]:
    """Compute statistical tests for outcome rate differences.

    Args:
        df: Polars DataFrame with patient data.
        y_true_col: Column name for true labels.
        group_col: Column name for grouping variable.
        reference: Reference group for pairwise comparisons.

    Returns:
        Dict containing chi-square test results and effect sizes.
    """
    # Create contingency table
    grouped = df.group_by(group_col, y_true_col).len()
    crosstab = _pivot_compat(grouped, columns=y_true_col, index=group_col, values="len").fill_null(
        0
    )

    # Get column names for outcomes
    outcome_cols = [c for c in crosstab.columns if c != group_col]
    if len(outcome_cols) < 2:
        return {"error": "Need at least 2 outcome values for chi-square test"}

    # Build contingency matrix
    contingency = crosstab.select(outcome_cols).to_numpy()

    # Chi-square test
    try:
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        cramers_v = np.sqrt(chi2 / (contingency.sum() * (min(contingency.shape) - 1)))
    except ValueError:
        return {"error": "Could not compute chi-square test"}

    result = {
        "chi_square": float(chi2),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "cramers_v": float(cramers_v),
        "interpretation": _interpret_cramers_v(cramers_v),
    }

    # Pairwise comparisons vs reference if specified
    if reference:
        pairwise = {}
        ref_data = df.filter(pl.col(group_col) == reference)
        ref_pos = int(ref_data[y_true_col].sum())
        ref_neg = len(ref_data) - ref_pos

        groups = df[group_col].drop_nulls().unique().to_list()
        for group in groups:
            if group == reference:
                continue

            group_data = df.filter(pl.col(group_col) == group)
            group_pos = int(group_data[y_true_col].sum())
            group_neg = len(group_data) - group_pos

            # 2x2 contingency table
            table = np.array([[ref_pos, ref_neg], [group_pos, group_neg]])
            try:
                odds_ratio, p_val = stats.fisher_exact(table)
                pairwise[str(group)] = {
                    "odds_ratio": float(odds_ratio),
                    "p_value": float(p_val),
                    "significant": p_val < 0.05,
                }
            except ValueError:
                pairwise[str(group)] = {"error": "Could not compute Fisher's exact test"}

        result["pairwise_vs_reference"] = pairwise

    return result


def _interpret_cramers_v(v: float) -> str:
    """Interpret Cramer's V effect size."""
    if v < 0.1:
        return "negligible"
    elif v < 0.2:
        return "small"
    elif v < 0.4:
        return "medium"
    else:
        return "large"


def format_table1_text(summary: dict[str, Any]) -> str:
    """Format cohort summary as text for console display.

    Args:
        summary: Dict from compute_cohort_summary().

    Returns:
        Formatted string for display.
    """
    lines = [
        "=" * 70,
        "TABLE 1: COHORT CHARACTERISTICS",
        "=" * 70,
        "",
    ]

    # Cohort overview
    overview = summary.get("cohort_overview", {})
    lines.extend(
        [
            "COHORT OVERVIEW",
            "-" * 40,
            f"  Total N:           {overview.get('n_total', 'N/A'):,}",
            f"  Outcome positive:  {overview.get('n_positive', 'N/A'):,} ({overview.get('prevalence_pct', 'N/A')})",
            f"  Outcome negative:  {overview.get('n_negative', 'N/A'):,}",
            "",
        ]
    )

    # Prediction distribution
    pred_dist = summary.get("prediction_distribution", {})
    if pred_dist:
        lines.extend(
            [
                "PREDICTION SCORE DISTRIBUTION",
                "-" * 40,
                f"  Mean (SD):         {pred_dist.get('mean', 0):.3f} ({pred_dist.get('std', 0):.3f})",
                f"  Median [IQR]:      {pred_dist.get('median', 0):.3f} [{pred_dist.get('percentile_25', 0):.3f}-{pred_dist.get('percentile_75', 0):.3f}]",
                f"  Range:             {pred_dist.get('min', 0):.3f} - {pred_dist.get('max', 0):.3f}",
                "",
            ]
        )

    # Attribute distributions
    attr_dist = summary.get("attribute_distributions", {})
    outcome_by_attr = summary.get("outcome_by_attribute", {})

    for attr_name, attr_data in attr_dist.items():
        lines.extend(
            [
                f"{attr_name.upper()}",
                "-" * 40,
            ]
        )

        if attr_data.get("n_missing", 0) > 0:
            lines.append(f"  Missing: {attr_data['n_missing']:,} ({attr_data['missing_rate']:.1%})")

        groups = attr_data.get("groups", {})
        outcome_groups = outcome_by_attr.get(attr_name, {}).get("groups", {})
        reference = outcome_by_attr.get(attr_name, {}).get("reference")

        # Header
        lines.append(f"  {'Group':<20} {'N':>8} {'%':>8} {'Outcome':>10} {'RR':>8}")
        lines.append("  " + "-" * 56)

        for group_name, group_data in groups.items():
            outcome_data = outcome_groups.get(group_name, {})
            rr = outcome_data.get("rate_ratio")
            rr_str = f"{rr:.2f}" if rr is not None else "-"
            ref_marker = " (ref)" if group_name == reference else ""

            lines.append(
                f"  {group_name:<20} {group_data['n']:>8,} {group_data['pct_fmt']:>8} "
                f"{outcome_data.get('outcome_rate_pct', 'N/A'):>10} {rr_str:>8}{ref_marker}"
            )

        lines.append("")

    lines.extend(
        [
            "=" * 70,
            "Note: RR = Rate Ratio vs reference group",
            "=" * 70,
        ]
    )

    return "\n".join(lines)


def generate_table1_dataframe(summary: dict[str, Any]) -> pl.DataFrame:
    """Generate Table 1 as a Polars DataFrame for export.

    Args:
        summary: Dict from compute_cohort_summary().

    Returns:
        Polars DataFrame with Table 1 data.
    """
    rows = []

    # Cohort overview
    overview = summary.get("cohort_overview", {})
    rows.append(
        {
            "Category": "Overall",
            "Group": "Total",
            "N": overview.get("n_total"),
            "Percentage": 100.0,
            "Outcome_Rate": overview.get("prevalence"),
            "Rate_Ratio": None,
            "CI_95_Low": None,
            "CI_95_High": None,
        }
    )

    # Attribute groups
    attr_dist = summary.get("attribute_distributions", {})
    outcome_by_attr = summary.get("outcome_by_attribute", {})

    for attr_name, attr_data in attr_dist.items():
        groups = attr_data.get("groups", {})
        outcome_groups = outcome_by_attr.get(attr_name, {}).get("groups", {})

        for group_name, group_data in groups.items():
            outcome_data = outcome_groups.get(group_name, {})
            rows.append(
                {
                    "Category": attr_name,
                    "Group": group_name,
                    "N": group_data.get("n"),
                    "Percentage": group_data.get("pct", 0) * 100,
                    "Outcome_Rate": outcome_data.get("outcome_rate"),
                    "Rate_Ratio": outcome_data.get("rate_ratio"),
                    "CI_95_Low": outcome_data.get("ci_95_low"),
                    "CI_95_High": outcome_data.get("ci_95_high"),
                }
            )

    return pl.DataFrame(rows)


def compute_continuous_variable_summary(
    df: pl.DataFrame,
    col: str,
    group_col: str | None = None,
) -> dict[str, Any]:
    """Compute summary statistics for a continuous variable.

    Args:
        df: Polars DataFrame with patient data.
        col: Column name for continuous variable.
        group_col: Optional grouping variable.

    Returns:
        Dict with summary statistics.
    """
    if group_col is None:
        # Overall summary
        data = df[col].drop_nulls()
        if len(data) == 0:
            return {"error": "No valid data"}

        arr = data.to_numpy()
        return {
            "n": int(len(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "q1": float(np.percentile(arr, 25)),
            "q3": float(np.percentile(arr, 75)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n_missing": int(df[col].null_count()),
        }

    # Grouped summary
    result: dict[str, Any] = {"groups": {}}
    for group in df[group_col].drop_nulls().unique().to_list():
        group_data = df.filter(pl.col(group_col) == group)[col].drop_nulls()
        if len(group_data) == 0:
            continue

        arr = group_data.to_numpy()
        result["groups"][str(group)] = {
            "n": int(len(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "q1": float(np.percentile(arr, 25)),
            "q3": float(np.percentile(arr, 75)),
        }

    return result
