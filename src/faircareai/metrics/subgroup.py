"""
FairCareAI Subgroup and Intersectional Analysis Module

Compute performance and fairness metrics for subgroups and
intersectional combinations of sensitive attributes.

Methodology: CHAI RAIC AC1.CR95 (subgroup performance), Van Calster et al. (2025).
Note: Small subgroups may have unstable estimates.
"""

from itertools import combinations
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from faircareai.core.metrics import compute_confusion_metrics


def compute_subgroup_metrics(
    df: pl.DataFrame,
    y_prob_col: str,
    y_true_col: str,
    group_col: str,
    threshold: float = 0.5,
    reference: str | None = None,
    bootstrap_ci: bool = True,
    n_bootstrap: int = 500,
) -> dict[str, Any]:
    """Compute comprehensive metrics for each subgroup.

    Args:
        df: Polars DataFrame with patient data.
        y_prob_col: Column name for predicted probabilities.
        y_true_col: Column name for true labels.
        group_col: Column name for grouping variable.
        threshold: Decision threshold.
        reference: Reference group for comparisons.
        bootstrap_ci: Whether to compute bootstrap CI.
        n_bootstrap: Number of bootstrap iterations.

    Returns:
        Dict with per-subgroup performance and fairness metrics.
    """
    results: dict[str, Any] = {
        "attribute": group_col,
        "threshold": threshold,
        "groups": {},
    }

    groups = df[group_col].drop_nulls().unique().sort().to_list()

    # Determine reference group
    if reference is None:
        group_counts = df.group_by(group_col).len().sort("len", descending=True)
        reference = group_counts[group_col][0]

    results["reference"] = reference

    # Compute metrics for each group
    for group in groups:
        group_df = df.filter(pl.col(group_col) == group)
        y_true = group_df[y_true_col].to_numpy()
        y_prob = group_df[y_prob_col].to_numpy()
        y_pred = (y_prob >= threshold).astype(int)

        n = len(y_true)

        # Basic info
        group_result: dict[str, Any] = {
            "n": n,
            "is_reference": str(group) == str(reference),
            "small_sample_warning": n < 100,
        }

        if n < 10:
            group_result["error"] = "Insufficient sample size (n < 10)"
            results["groups"][str(group)] = group_result
            continue

        # Prevalence
        prevalence = np.mean(y_true)
        group_result["prevalence"] = float(prevalence)

        # Classification metrics from confusion matrix
        try:
            cm = compute_confusion_metrics(y_true, y_pred, fairness_naming=True)
        except ValueError:
            group_result["error"] = "Could not compute metrics"
            results["groups"][str(group)] = group_result
            continue

        group_result.update(
            {
                "tpr": float(cm["tpr"]),
                "fpr": float(cm["fpr"]),
                "ppv": float(cm["ppv"]),
                "npv": float(cm["npv"]),
                "selection_rate": float(cm["selection_rate"]),
                "tp": cm["tp"],
                "fp": cm["fp"],
                "tn": cm["tn"],
                "fn": cm["fn"],
            }
        )

        # AUROC
        if len(np.unique(y_true)) >= 2:
            auroc = roc_auc_score(y_true, y_prob)
            group_result["auroc"] = float(auroc)

            # Bootstrap CI for AUROC
            if bootstrap_ci and n >= 20:
                auroc_samples = _bootstrap_auroc(y_true, y_prob, n_bootstrap)
                if len(auroc_samples) > 10:
                    auroc_ci = np.percentile(auroc_samples, [2.5, 97.5])
                    group_result["auroc_ci_95"] = [float(auroc_ci[0]), float(auroc_ci[1])]

        # Mean prediction
        group_result["mean_predicted_prob"] = float(np.mean(y_prob))

        results["groups"][str(group)] = group_result

    # Compute disparities vs reference
    results["disparities"] = _compute_subgroup_disparities(results, reference)

    return results


def _bootstrap_auroc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int,
) -> list[float]:
    """Bootstrap AUROC samples.

    Note: This is a thin wrapper around faircareai.core.bootstrap.bootstrap_auroc
    for backward compatibility.
    """
    from faircareai.core.bootstrap import bootstrap_metric

    samples, _ = bootstrap_metric(
        y_true,
        y_prob,
        lambda yt, yp: roc_auc_score(yt, yp),
        n_bootstrap=n_bootstrap,
        seed=42,
        min_classes=2,
    )
    return samples


def _compute_subgroup_disparities(
    results: dict[str, Any],
    reference: str,
) -> dict[str, Any]:
    """Compute disparities relative to reference group."""
    disparities: dict[str, Any] = {}

    ref_data = results["groups"].get(str(reference), {})
    if "error" in ref_data:
        return {"error": f"Reference group '{reference}' has insufficient data"}

    ref_tpr = ref_data.get("tpr", 0)
    ref_fpr = ref_data.get("fpr", 0)
    ref_ppv = ref_data.get("ppv", 0)
    ref_auroc = ref_data.get("auroc")
    ref_selection = ref_data.get("selection_rate", 0)

    for group_name, group_data in results["groups"].items():
        if str(group_name) == str(reference):
            continue
        if "error" in group_data:
            continue

        group_disp: dict[str, Any] = {}

        # TPR difference (equal opportunity)
        tpr = group_data.get("tpr", 0)
        group_disp["tpr_diff"] = float(tpr - ref_tpr)

        # FPR difference
        fpr = group_data.get("fpr", 0)
        group_disp["fpr_diff"] = float(fpr - ref_fpr)

        # Equalized odds (max of TPR/FPR diff)
        group_disp["equalized_odds_diff"] = float(max(abs(tpr - ref_tpr), abs(fpr - ref_fpr)))

        # PPV ratio (predictive parity)
        ppv = group_data.get("ppv", 0)
        if ref_ppv > 0:
            group_disp["ppv_ratio"] = float(ppv / ref_ppv)

        # Selection rate ratio (demographic parity)
        selection = group_data.get("selection_rate", 0)
        if ref_selection > 0:
            group_disp["selection_rate_ratio"] = float(selection / ref_selection)
        group_disp["selection_rate_diff"] = float(selection - ref_selection)

        # AUROC difference
        auroc = group_data.get("auroc")
        if auroc is not None and ref_auroc is not None:
            group_disp["auroc_diff"] = float(auroc - ref_auroc)

        disparities[str(group_name)] = group_disp

    return disparities


def compute_intersectional(
    df: pl.DataFrame,
    y_prob_col: str,
    y_true_col: str,
    group_cols: list[str],
    threshold: float = 0.5,
    min_n: int = 30,
    bootstrap_ci: bool = False,
    n_bootstrap: int = 500,
) -> dict[str, Any]:
    """Compute metrics for intersectional subgroups.

    Args:
        df: Polars DataFrame with patient data.
        y_prob_col: Column name for predicted probabilities.
        y_true_col: Column name for true labels.
        group_cols: List of columns to intersect.
        threshold: Decision threshold.
        min_n: Minimum sample size for reporting.
        bootstrap_ci: Whether to compute bootstrap CI.
        n_bootstrap: Number of bootstrap iterations.

    Returns:
        Dict with intersectional analysis results.
    """
    results: dict[str, Any] = {
        "attributes": group_cols,
        "threshold": threshold,
        "min_n": min_n,
        "intersections": {},
    }

    # Create intersection column
    df = df.with_columns(
        pl.concat_str([pl.col(c).cast(str) for c in group_cols], separator=" x ").alias(
            "_intersection"
        )
    )

    # Get unique intersections
    intersections = (
        df.group_by("_intersection")
        .agg(
            pl.len().alias("n"),
            pl.col(y_true_col).sum().alias("n_positive"),
        )
        .filter(pl.col("n") >= min_n)
        .sort("n", descending=True)
    )

    # Track overall best and worst
    best_auroc: dict[str, str | float | None] = {"group": None, "value": 0.0}
    worst_auroc: dict[str, str | float | None] = {"group": None, "value": 1.0}

    for row in intersections.iter_rows(named=True):
        intersection_name = row["_intersection"]
        intersection_df = df.filter(pl.col("_intersection") == intersection_name)

        y_true = intersection_df[y_true_col].to_numpy()
        y_prob = intersection_df[y_prob_col].to_numpy()
        y_pred = (y_prob >= threshold).astype(int)

        n = len(y_true)

        group_result: dict[str, Any] = {
            "n": n,
            "prevalence": float(np.mean(y_true)),
            "small_sample_warning": n < 100,
        }

        # Confusion matrix metrics
        try:
            cm = compute_confusion_metrics(y_true, y_pred, fairness_naming=True)
            group_result.update(
                {
                    "tpr": float(cm["tpr"]),
                    "fpr": float(cm["fpr"]),
                    "ppv": float(cm["ppv"]),
                    "selection_rate": float(cm["selection_rate"]),
                }
            )
        except ValueError:
            pass

        # AUROC
        if len(np.unique(y_true)) >= 2:
            try:
                auroc = roc_auc_score(y_true, y_prob)
                group_result["auroc"] = float(auroc)

                # Track best/worst
                if auroc > best_auroc["value"]:
                    best_auroc = {"group": intersection_name, "value": float(auroc)}
                if auroc < worst_auroc["value"]:
                    worst_auroc = {"group": intersection_name, "value": float(auroc)}

                # Bootstrap CI
                if bootstrap_ci and n >= 20:
                    auroc_samples = _bootstrap_auroc(y_true, y_prob, n_bootstrap)
                    if len(auroc_samples) > 10:
                        auroc_ci = np.percentile(auroc_samples, [2.5, 97.5])
                        group_result["auroc_ci_95"] = [
                            float(auroc_ci[0]),
                            float(auroc_ci[1]),
                        ]
            except ValueError:
                pass

        results["intersections"][intersection_name] = group_result

    # Compute disparities between intersections
    if best_auroc["group"] and worst_auroc["group"]:
        best_val = float(best_auroc["value"]) if best_auroc["value"] is not None else 0.0
        worst_val = float(worst_auroc["value"]) if worst_auroc["value"] is not None else 0.0
        auroc_disparity = best_val - worst_val
        results["summary"] = {
            "n_intersections": len(results["intersections"]),
            "n_excluded_small": len(intersections) - len(results["intersections"]),
            "best_performing": best_auroc,
            "worst_performing": worst_auroc,
            "auroc_range": float(auroc_disparity),
            "concern_level": _interpret_auroc_range(auroc_disparity),
        }
    else:
        results["summary"] = {
            "n_intersections": len(results["intersections"]),
            "n_excluded_small": 0,
            "note": "Insufficient data for summary",
        }

    return results


def _interpret_auroc_range(range_val: float) -> str:
    """Interpret AUROC range across intersections."""
    if range_val < 0.05:
        return "LOW"
    elif range_val < 0.1:
        return "MODERATE"
    elif range_val < 0.15:
        return "HIGH"
    else:
        return "SEVERE"


def compute_pairwise_intersectional(
    df: pl.DataFrame,
    y_prob_col: str,
    y_true_col: str,
    sensitive_attrs: dict[str, dict],
    threshold: float = 0.5,
    min_n: int = 30,
) -> dict[str, Any]:
    """Compute intersectional analysis for all pairs of attributes.

    Args:
        df: Polars DataFrame with patient data.
        y_prob_col: Column name for predicted probabilities.
        y_true_col: Column name for true labels.
        sensitive_attrs: Dict of sensitive attribute configurations.
        threshold: Decision threshold.
        min_n: Minimum sample size.

    Returns:
        Dict with pairwise intersectional results.
    """
    results: dict[str, Any] = {"pairs": {}}

    attr_cols = [cfg.get("column", name) for name, cfg in sensitive_attrs.items()]
    attr_names = list(sensitive_attrs.keys())

    # Generate all pairs
    for (name1, name2), (col1, col2) in zip(
        combinations(attr_names, 2), combinations(attr_cols, 2), strict=False
    ):
        pair_key = f"{name1} x {name2}"

        pair_result = compute_intersectional(
            df, y_prob_col, y_true_col, [col1, col2], threshold, min_n
        )

        results["pairs"][pair_key] = {
            "attributes": [name1, name2],
            "summary": pair_result.get("summary", {}),
            "n_intersections": len(pair_result.get("intersections", {})),
        }

    # Find most concerning pair
    worst_pair = None
    worst_range = 0.0

    for pair_key, pair_data in results["pairs"].items():
        auroc_range = pair_data.get("summary", {}).get("auroc_range", 0)
        if auroc_range is not None and auroc_range > worst_range:
            worst_range = auroc_range
            worst_pair = pair_key

    results["summary"] = {
        "n_pairs_analyzed": len(results["pairs"]),
        "most_disparate_pair": worst_pair,
        "worst_auroc_range": float(worst_range) if worst_range is not None else None,
    }

    return results


def identify_vulnerable_subgroups(
    df: pl.DataFrame,
    y_prob_col: str,
    y_true_col: str,
    sensitive_attrs: dict[str, dict],
    threshold: float = 0.5,
    auroc_threshold: float = 0.7,
    min_n: int = 30,
) -> dict[str, Any]:
    """Identify subgroups with concerning performance.

    Args:
        df: Polars DataFrame with patient data.
        y_prob_col: Column name for predicted probabilities.
        y_true_col: Column name for true labels.
        sensitive_attrs: Dict of sensitive attribute configurations.
        threshold: Decision threshold.
        auroc_threshold: AUROC below this is flagged.
        min_n: Minimum sample size.

    Returns:
        Dict with vulnerable subgroups identified.
    """
    vulnerable: list[dict] = []

    # Check single attributes
    for attr_name, attr_config in sensitive_attrs.items():
        col = attr_config.get("column", attr_name)

        metrics = compute_subgroup_metrics(df, y_prob_col, y_true_col, col, threshold)

        for group_name, group_data in metrics.get("groups", {}).items():
            if "error" in group_data:
                continue

            auroc = group_data.get("auroc")
            n = group_data.get("n", 0)

            if auroc is not None and auroc < auroc_threshold and n >= min_n:
                vulnerable.append(
                    {
                        "type": "single",
                        "attribute": attr_name,
                        "group": group_name,
                        "n": n,
                        "auroc": float(auroc),
                        "concern": "Low discriminative performance",
                    }
                )

    # Check intersections
    if len(sensitive_attrs) >= 2:
        attr_cols = [cfg.get("column", name) for name, cfg in sensitive_attrs.items()]

        for pair in combinations(attr_cols, 2):
            intersectional = compute_intersectional(
                df, y_prob_col, y_true_col, list(pair), threshold, min_n
            )

            for group_name, group_data in intersectional.get("intersections", {}).items():
                auroc = group_data.get("auroc")
                n = group_data.get("n", 0)

                if auroc is not None and auroc < auroc_threshold:
                    vulnerable.append(
                        {
                            "type": "intersectional",
                            "attributes": list(pair),
                            "group": group_name,
                            "n": n,
                            "auroc": float(auroc),
                            "concern": "Low discriminative performance in intersection",
                        }
                    )

    # Sort by AUROC (lowest first)
    vulnerable.sort(key=lambda x: x.get("auroc", 1.0))

    return {
        "vulnerable_subgroups": vulnerable,
        "n_vulnerable": len(vulnerable),
        "auroc_threshold": auroc_threshold,
        "summary": _summarize_vulnerable(vulnerable),
    }


def _summarize_vulnerable(vulnerable: list[dict]) -> dict[str, Any]:
    """Summarize vulnerable subgroups."""
    if not vulnerable:
        return {
            "status": "PASS",
            "message": "No vulnerable subgroups identified.",
        }

    n_single = sum(1 for v in vulnerable if v.get("type") == "single")
    n_intersectional = sum(1 for v in vulnerable if v.get("type") == "intersectional")

    worst = vulnerable[0] if vulnerable else None

    return {
        "status": "REVIEW" if len(vulnerable) <= 2 else "CONCERN",
        "n_single_attribute": n_single,
        "n_intersectional": n_intersectional,
        "worst_subgroup": worst.get("group") if worst else None,
        "worst_auroc": worst.get("auroc") if worst else None,
        "message": f"Found {len(vulnerable)} subgroup(s) with AUROC below threshold.",
    }
