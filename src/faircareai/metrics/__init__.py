"""
FairCareAI Metrics Module

Compute descriptive statistics, overall performance, fairness metrics,
subgroup analysis, and Van Calster recommended performance measures.
"""

from faircareai.metrics.descriptive import (
    compute_cohort_summary,
    format_table1_text,
    generate_table1_dataframe,
)
from faircareai.metrics.fairness import compute_fairness_metrics
from faircareai.metrics.performance import compute_overall_performance
from faircareai.metrics.subgroup import (
    compute_intersectional,
    compute_subgroup_metrics,
)
from faircareai.metrics.vancalster import (
    compute_auroc_by_subgroup,
    compute_calibration_by_subgroup,
    compute_net_benefit_by_subgroup,
    compute_risk_distribution_by_subgroup,
    compute_vancalster_metrics,
)

__all__ = [
    # Descriptive statistics
    "compute_cohort_summary",
    "format_table1_text",
    "generate_table1_dataframe",
    # Performance metrics
    "compute_overall_performance",
    # Fairness metrics
    "compute_fairness_metrics",
    # Subgroup analysis
    "compute_subgroup_metrics",
    "compute_intersectional",
    # Van Calster (2025) recommended metrics
    "compute_vancalster_metrics",
    "compute_auroc_by_subgroup",
    "compute_calibration_by_subgroup",
    "compute_net_benefit_by_subgroup",
    "compute_risk_distribution_by_subgroup",
]
