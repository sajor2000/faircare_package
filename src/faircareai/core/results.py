"""
FairCareAI Audit Results Container

Container for fairness audit results with export and visualization capabilities.

Metrics computed per Van Calster et al. (2025) methodology. Healthcare
organizations interpret results based on their clinical context,
organizational values, and governance frameworks.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from faircareai.core.config import FairnessConfig
from faircareai.core.logging import get_logger
from faircareai.visualization.themes import GOVERNANCE_DISCLAIMER_SHORT

logger = get_logger(__name__)


@dataclass
class AuditResults:
    """Container for fairness audit results with export capabilities.

    This is the main output from FairCareAudit.run(). It contains all
    computed metrics, flags, and provides methods for visualization
    and report generation.

    Attributes:
        config: FairnessConfig used for the audit.
        descriptive_stats: Section 1 - Table 1 cohort summary.
        overall_performance: Section 2 - TRIPOD+AI metrics.
        subgroup_performance: Section 3 - Performance by sensitive attribute.
        fairness_metrics: Section 4 - Fairness metrics per attribute.
        intersectional: Intersectional analysis results.
        flags: List of metrics outside configured thresholds.
        governance_recommendation: Section 7 - Summary statistics.
    """

    config: FairnessConfig

    # Results - IN ORDER OF REPORT SECTIONS
    # Section 1: Descriptive Statistics (Table 1)
    descriptive_stats: dict = field(default_factory=dict)

    # Section 2: Overall Model Performance
    overall_performance: dict = field(default_factory=dict)

    # Section 3: Subgroup Performance
    subgroup_performance: dict = field(default_factory=dict)

    # Section 4: Fairness Metrics
    fairness_metrics: dict = field(default_factory=dict)
    intersectional: dict = field(default_factory=dict)

    # Section 5: Flags & Warnings
    flags: list[dict] = field(default_factory=list)

    # Section 7: Governance Advisory
    governance_recommendation: dict = field(default_factory=dict)

    # Internal reference to audit for raw data access
    _audit: Any = None

    def summary(self) -> str:
        """Print summary to console.

        Returns:
            Formatted summary string.
        """
        desc = self.descriptive_stats
        cohort = desc.get("cohort_overview", {})
        perf = self.overall_performance
        disc = perf.get("discrimination", {})
        cal = perf.get("calibration", {})
        cls = perf.get("classification_at_threshold", {})
        gov = self.governance_recommendation

        # Format metrics safely - handles None and numeric values
        def fmt(val, fmt_str=".3f"):
            if val is None:
                return "N/A"
            try:
                return f"{val:{fmt_str}}"
            except (TypeError, ValueError):
                return "N/A"

        # Safe percentage formatting - handles 0.0 correctly (0.0 is falsy but valid)
        def fmt_pct(val):
            if val is None:
                return "N/A"
            return f"{val * 100:.1f}"

        # Cache repeated dict access for efficiency
        n_total = cohort.get("n_total")
        n_positive = cohort.get("n_positive")
        prevalence_pct = cohort.get("prevalence_pct", "N/A")

        # Format N values
        n_total_str = f"{n_total:,}" if isinstance(n_total, (int, float)) else str(n_total or "N/A")
        n_positive_str = (
            f"{n_positive:,}" if isinstance(n_positive, (int, float)) else str(n_positive or "N/A")
        )

        # Get Brier score from calibration dict (correct location)
        brier_score = cal.get("brier_score")

        # Get classification metrics safely
        sensitivity = cls.get("sensitivity")
        specificity = cls.get("specificity")
        ppv = cls.get("ppv")
        pct_flagged = cls.get("pct_flagged")

        lines = [
            "=" * 70,
            "FairCareAI Fairness Analysis Results",
            "=" * 70,
            f"Model: {self.config.model_name} v{self.config.model_version}",
            "",
            "SECTION 1: COHORT SUMMARY",
            f"  N:              {n_total_str}",
            f"  Outcome:        {n_positive_str} ({prevalence_pct})",
            "",
            "SECTION 2: OVERALL MODEL PERFORMANCE (TRIPOD+AI)",
            "  Discrimination:",
            f"    AUROC:        {fmt(disc.get('auroc'))} {disc.get('auroc_ci_fmt', '')}",
            f"    AUPRC:        {fmt(disc.get('auprc'))} {disc.get('auprc_ci_fmt', '')}",
            "  Calibration:",
            f"    Brier Score:  {fmt(brier_score, '.4f')}",
            f"    Cal. Slope:   {fmt(cal.get('calibration_slope'), '.2f')} (ideal: 1.00)",
            f"  At Threshold = {cls.get('threshold', 'N/A')}:",
            f"    Sensitivity:  {fmt_pct(sensitivity)}%",
            f"    Specificity:  {fmt_pct(specificity)}%",
            f"    PPV:          {fmt_pct(ppv)}%",
            f"    % Flagged:    {fmt(pct_flagged, '.1f') if pct_flagged is not None else 'N/A'}%",
            "",
            "SECTION 4: FAIRNESS SUMMARY",
            f"  Primary metric: {self.config.primary_fairness_metric.value if self.config.primary_fairness_metric else 'Not set'}",
            "",
            "SECTION 7: RESULTS SUMMARY",
            f"  Within threshold: {gov.get('n_pass', gov.get('within_threshold_count', 0))}",
            f"  Near threshold: {gov.get('n_warnings', gov.get('near_threshold_count', 0))}",
            f"  Outside threshold: {gov.get('n_errors', gov.get('outside_threshold_count', 0))}",
            "",
            "=" * 70,
            f"  {GOVERNANCE_DISCLAIMER_SHORT}",
            "=" * 70,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    # === Table 1 Methods ===

    def print_table1(self) -> str:
        """Print Table 1 descriptive statistics.

        Returns:
            Formatted Table 1 string.
        """
        from faircareai.metrics.descriptive import format_table1_text

        text = format_table1_text(self.descriptive_stats)
        logger.info("Table 1:\n%s", text)
        return text

    def get_table1_dataframe(self) -> pl.DataFrame:
        """Get Table 1 as a Polars DataFrame for export.

        Returns:
            Polars DataFrame with Table 1 data.
        """
        from faircareai.metrics.descriptive import generate_table1_dataframe

        return generate_table1_dataframe(self.descriptive_stats)

    # === Visualization Methods ===

    def plot_discrimination(self):
        """Plot ROC and Precision-Recall curves (TRIPOD+AI 2.1).

        Returns:
            Plotly Figure with side-by-side curves.
        """
        from faircareai.visualization.performance_charts import plot_discrimination_curves

        return plot_discrimination_curves(self)

    def plot_overall_calibration(self):
        """Plot calibration curve for overall model (TRIPOD+AI 2.2).

        Returns:
            Plotly Figure with calibration curve.
        """
        from faircareai.visualization.performance_charts import plot_calibration_curve

        return plot_calibration_curve(self)

    def plot_threshold_analysis(self, selected_threshold: float | None = None):
        """Interactive threshold sensitivity analysis (TRIPOD+AI 2.4).

        Data scientist can TOGGLE threshold to see metric impacts.

        Args:
            selected_threshold: Threshold to highlight (default: primary threshold).

        Returns:
            Plotly Figure with threshold analysis.
        """
        from faircareai.visualization.performance_charts import plot_threshold_analysis

        thresh = selected_threshold or self.overall_performance.get("primary_threshold", 0.5)
        return plot_threshold_analysis(self, selected_threshold=thresh)

    def plot_decision_curve(self):
        """Plot Decision Curve Analysis for clinical utility (TRIPOD+AI 2.5).

        Returns:
            Plotly Figure with DCA curves.
        """
        from faircareai.visualization.performance_charts import plot_decision_curve

        return plot_decision_curve(self)

    def plot_calibration(self, by: str | None = None):
        """Plot calibration curve(s).

        Args:
            by: Sensitive attribute to stratify by (None for overall).

        Returns:
            Plotly Figure with calibration curve(s).
        """
        from faircareai.visualization.plots import create_calibration_plot

        if by:
            # Stratified calibration
            return create_calibration_plot(
                self._audit.df if self._audit else None,
                self._audit.y_prob_col if self._audit else "y_prob",
                self._audit.y_true_col if self._audit else "y_true",
                by,
            )
        else:
            return self.plot_overall_calibration()

    def plot_fairness_dashboard(self):
        """Plot comprehensive fairness dashboard.

        Returns:
            Plotly Figure with 4-panel fairness dashboard.
        """
        from faircareai.visualization.governance_dashboard import (
            create_fairness_dashboard,
        )

        return create_fairness_dashboard(self)

    def plot_subgroup_performance(self, metric: str = "auroc"):
        """Plot subgroup performance comparison.

        Args:
            metric: Metric to compare ('auroc', 'tpr', 'fpr', 'ppv').

        Returns:
            Plotly Figure with subgroup comparison.
        """
        from faircareai.visualization.governance_dashboard import (
            plot_subgroup_comparison,
        )

        return plot_subgroup_comparison(self, metric=metric)

    def plot_executive_summary(self):
        """Plot executive summary for governance committee.

        Single-page visual with:
        - Traffic light status
        - Key metrics at a glance
        - Worst disparity highlighted
        - Plain language interpretation

        Returns:
            Plotly Figure with executive summary.
        """
        from faircareai.visualization.governance_dashboard import (
            create_executive_summary,
        )

        return create_executive_summary(self)

    def plot_go_nogo_scorecard(self):
        """Plot scorecard for governance presentation.

        Returns:
            Plotly Figure with checklist-style scorecard.
        """
        from faircareai.visualization.governance_dashboard import (
            create_go_nogo_scorecard,
        )

        return create_go_nogo_scorecard(self)

    # === Export Methods ===

    def to_html(self, path: str | Path, open_browser: bool = False) -> Path:
        """Export interactive HTML report.

        Args:
            path: Output file path.
            open_browser: Open report in browser after generation.

        Returns:
            Path to generated report.
        """
        from faircareai.reports.generator import generate_html_report

        path = Path(path)
        generate_html_report(self, path)

        if open_browser:
            import webbrowser

            webbrowser.open(path.absolute().as_uri())

        return path

    def to_pdf(self, path: str | Path) -> Path:
        """Export PDF report for governance review.

        Args:
            path: Output file path.

        Returns:
            Path to generated report.
        """
        from faircareai.reports.generator import generate_pdf_report

        path = Path(path)
        # Convert AuditResults to AuditSummary for generator
        summary = self._to_audit_summary()
        return generate_pdf_report(summary, path)

    def to_pptx(self, path: str | Path) -> Path:
        """Export PowerPoint deck for governance review.

        Creates a presentation suitable for board meetings and
        governance committee presentations.

        Args:
            path: Output file path.

        Returns:
            Path to generated presentation.
        """
        from faircareai.reports.generator import generate_pptx_report

        path = Path(path)
        summary = self._to_audit_summary()
        return generate_pptx_report(summary, path)

    def to_json(self, path: str | Path) -> Path:
        """Export metrics as JSON for programmatic use.

        Args:
            path: Output file path.

        Returns:
            Path to generated JSON file.
        """
        path = Path(path)

        export_data = {
            "config": {
                "model_name": self.config.model_name,
                "model_version": self.config.model_version,
                "primary_fairness_metric": (
                    self.config.primary_fairness_metric.value
                    if self.config.primary_fairness_metric
                    else None
                ),
                "fairness_justification": self.config.fairness_justification,
                "use_case_type": (
                    self.config.use_case_type.value if self.config.use_case_type else None
                ),
                "thresholds": self.config.thresholds,
            },
            "descriptive_stats": self.descriptive_stats,
            "overall_performance": _make_json_serializable(self.overall_performance),
            "subgroup_performance": _make_json_serializable(self.subgroup_performance),
            "fairness_metrics": _make_json_serializable(self.fairness_metrics),
            "intersectional": _make_json_serializable(self.intersectional),
            "flags": self.flags,
            "governance_recommendation": self.governance_recommendation,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, default=str)

        return path

    def _to_audit_summary(self):
        """Convert to legacy AuditSummary for report generator compatibility."""
        from faircareai.reports.generator import AuditSummary

        # Get worst disparity
        worst_group = ""
        worst_metric = ""
        worst_value = 0.0

        for attr_name, metrics in self.fairness_metrics.items():
            if not isinstance(metrics, dict):
                continue
            if "equalized_odds_diff" in metrics:
                eo_diffs = metrics["equalized_odds_diff"]
                if not isinstance(eo_diffs, dict):
                    continue
                for group, diff in eo_diffs.items():
                    # Skip None values
                    if diff is None:
                        continue
                    if abs(diff) > abs(worst_value):
                        worst_group = f"{attr_name}:{group}"
                        worst_metric = "equalized_odds"
                        worst_value = diff

        # Count groups - handle nested structure
        n_groups = 0
        for _attr_name, metrics in self.subgroup_performance.items():
            if isinstance(metrics, dict):
                # Get groups from nested structure
                groups = metrics.get("groups", metrics)
                n_groups += len(
                    [k for k in groups if k not in ("reference", "attribute", "threshold")]
                )

        return AuditSummary(
            model_name=self.config.model_name,
            audit_date=self.config.report_date or "",
            n_samples=self.descriptive_stats.get("cohort_overview", {}).get("n_total", 0),
            n_groups=n_groups,
            threshold=self.config.decision_thresholds[0]
            if self.config.decision_thresholds
            else 0.5,
            pass_count=self.governance_recommendation.get("n_pass", 0),
            warn_count=self.governance_recommendation.get("n_warnings", 0),
            fail_count=self.governance_recommendation.get("n_errors", 0),
            worst_disparity_group=worst_group,
            worst_disparity_metric=worst_metric,
            worst_disparity_value=worst_value,
            metrics_df=pl.DataFrame(),  # Would need to reconstruct
            disparities_df=pl.DataFrame(),  # Would need to reconstruct
        )


def _make_json_serializable(obj):
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]
    elif isinstance(obj, pl.DataFrame):
        return obj.to_dicts()
    elif isinstance(obj, pl.Series):
        return obj.to_list()  # Series uses to_list(), not to_dicts()
    elif hasattr(obj, "tolist"):  # numpy arrays
        return obj.tolist()
    elif hasattr(obj, "item"):  # numpy scalars
        return obj.item()
    else:
        return obj
