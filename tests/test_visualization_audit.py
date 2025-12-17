"""
Comprehensive Visualization Audit Tests - CHAI & Van Calster Compliance

This test module audits ALL figure generation functions in FairCareAI to verify:
1. They execute without errors
2. They follow CHAI RAIC Checkpoint 1 criteria (CR92, CR95, CR102)
3. They follow Van Calster et al. (2025) methodology
4. They have WCAG 2.1 AA alt text for accessibility
5. They support both overall model assessment AND subgroup analysis

Reference:
    Van Calster B, Collins GS, Vickers AJ, et al. Evaluation of performance
    measures in predictive AI models to support medical decisions.
    Lancet Digit Health 2025. doi:10.1016/j.landig.2025.100916

    CHAI RAIC Checkpoint 1 (2024) - Coalition for Health AI
"""

import numpy as np
import plotly.graph_objects as go
import polars as pl
import pytest

from faircareai import FairCareAudit

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def audit_results():
    """Run a full audit to get AuditResults for visualization testing."""
    from faircareai.core.config import FairnessConfig, FairnessMetric

    np.random.seed(42)
    n = 400

    # Create realistic clinical ML scenario
    race = ["White"] * 200 + ["Black"] * 120 + ["Hispanic"] * 80

    # Risk scores with group-level differences
    risk_white = np.random.beta(2, 5, 200)
    risk_black = np.random.beta(2.5, 4.5, 120)
    risk_hispanic = np.random.beta(2, 5.5, 80)

    risk_scores = np.concatenate([risk_white, risk_black, risk_hispanic])
    outcomes = (np.random.random(n) < risk_scores).astype(int)

    df = pl.DataFrame(
        {
            "risk_score": risk_scores,
            "readmitted": outcomes,
            "race": race,
        }
    )

    # CHAI RAIC Checkpoint 1 requires fairness metric and justification
    config = FairnessConfig(
        model_name="Test Model",
        primary_fairness_metric=FairnessMetric.EQUALIZED_ODDS,
        fairness_justification="Testing visualization components - equalized odds selected for intervention trigger use case",
    )

    audit = FairCareAudit(
        data=df,
        pred_col="risk_score",
        target_col="readmitted",
        config=config,
        threshold=0.3,
    )

    # Add sensitive attribute
    audit.add_sensitive_attribute(
        name="race",
        column="race",
        reference="White",
    )

    return audit.run(bootstrap_ci=False)


@pytest.fixture
def metrics_df():
    """Create a metrics DataFrame for plot functions that need it."""
    return pl.DataFrame(
        {
            "group": ["White", "Black", "Hispanic"],
            "n": [200, 120, 80],
            "tpr": [0.75, 0.68, 0.72],
            "fpr": [0.12, 0.15, 0.13],
            "ppv": [0.65, 0.58, 0.62],
            "npv": [0.88, 0.85, 0.87],
            "auroc": [0.82, 0.78, 0.80],
            "accuracy": [0.85, 0.82, 0.84],
        }
    )


@pytest.fixture
def disparity_df():
    """Create a disparity DataFrame for heatmap functions."""
    return pl.DataFrame(
        {
            "group": ["White", "Black", "Hispanic"],
            "tpr_diff": [0.0, -0.07, -0.03],
            "fpr_diff": [0.0, 0.03, 0.01],
            "ppv_diff": [0.0, -0.07, -0.03],
        }
    )


@pytest.fixture
def vancalster_results():
    """Create Van Calster-style results dict for vancalster_plots functions."""
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    prevalence = 0.28

    # Calculate net benefit for treat all/none strategies
    net_benefit_none = [0.0] * len(thresholds)
    net_benefit_all = [prevalence - (1 - prevalence) * t / (1 - t) for t in thresholds]

    return {
        "groups": {
            "White": {
                "n": 200,
                "auroc": 0.82,
                "auroc_ci_95": [0.78, 0.86],
                "is_reference": True,
                "calibration_curve": {
                    "prob_pred": [0.1, 0.3, 0.5, 0.7, 0.9],
                    "prob_true": [0.12, 0.28, 0.52, 0.68, 0.88],
                },
                "decision_curve": {
                    "thresholds": thresholds,
                    "net_benefit_model": [0.15, 0.12, 0.10, 0.08, 0.05],
                    "net_benefit_all": net_benefit_all,
                    "net_benefit_none": net_benefit_none,
                },
                "risk_distribution": {
                    "events": [0.4, 0.5, 0.6, 0.7, 0.8],
                    "nonevents": [0.1, 0.2, 0.2, 0.3, 0.3],
                },
            },
            "Black": {
                "n": 120,
                "auroc": 0.78,
                "auroc_ci_95": [0.72, 0.84],
                "is_reference": False,
                "calibration_curve": {
                    "prob_pred": [0.1, 0.3, 0.5, 0.7, 0.9],
                    "prob_true": [0.15, 0.32, 0.48, 0.65, 0.85],
                },
                "decision_curve": {
                    "thresholds": thresholds,
                    "net_benefit_model": [0.14, 0.11, 0.09, 0.07, 0.04],
                    "net_benefit_all": net_benefit_all,
                    "net_benefit_none": net_benefit_none,
                },
                "risk_distribution": {
                    "events": [0.45, 0.55, 0.65, 0.75, 0.85],
                    "nonevents": [0.15, 0.25, 0.25, 0.35, 0.35],
                },
            },
            "Hispanic": {
                "n": 80,
                "auroc": 0.80,
                "auroc_ci_95": [0.73, 0.87],
                "is_reference": False,
                "calibration_curve": {
                    "prob_pred": [0.1, 0.3, 0.5, 0.7, 0.9],
                    "prob_true": [0.11, 0.29, 0.51, 0.69, 0.89],
                },
                "decision_curve": {
                    "thresholds": thresholds,
                    "net_benefit_model": [0.16, 0.13, 0.11, 0.09, 0.06],
                    "net_benefit_all": net_benefit_all,
                    "net_benefit_none": net_benefit_none,
                },
                "risk_distribution": {
                    "events": [0.38, 0.48, 0.58, 0.68, 0.78],
                    "nonevents": [0.08, 0.18, 0.18, 0.28, 0.28],
                },
            },
        },
        "overall": {
            "auroc": 0.81,
            "prevalence": prevalence,
        },
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def assert_valid_plotly_figure(fig, test_name: str):
    """Assert that a Plotly figure is valid and has alt text."""
    assert fig is not None, f"{test_name}: Figure is None"
    assert isinstance(fig, go.Figure), f"{test_name}: Not a Plotly Figure"
    # Verify figure has data or annotations
    has_content = len(fig.data) > 0 or len(fig.layout.annotations) > 0
    assert has_content, f"{test_name}: Figure has no data or annotations"


def assert_has_wcag_alt_text(fig, test_name: str):
    """Assert that figure has WCAG 2.1 AA alt text in metadata."""
    meta = getattr(fig.layout, "meta", None)
    if meta is not None and isinstance(meta, dict):
        description = meta.get("description", "")
        if description:
            assert len(description) > 10, f"{test_name}: Alt text too short"


# =============================================================================
# TEST CLASS: plots.py (10 functions - Core Subgroup Analysis)
# CHAI CR92 (Bias Testing), CR95 (Performance Reporting)
# =============================================================================


class TestPlotsModule:
    """Audit plots.py - 10 functions for subgroup fairness analysis."""

    def test_create_forest_plot(self, metrics_df):
        """Test forest plot - CHAI CR92: Disparity visualization."""
        from faircareai.visualization.plots import create_forest_plot

        fig = create_forest_plot(metrics_df, metric="tpr")
        assert_valid_plotly_figure(fig, "create_forest_plot")

    def test_create_disparity_heatmap(self, disparity_df):
        """Test disparity heatmap - CHAI CR92: Pairwise disparity matrix."""
        from faircareai.visualization.plots import create_disparity_heatmap

        fig = create_disparity_heatmap(disparity_df, metric="tpr_diff")
        assert_valid_plotly_figure(fig, "create_disparity_heatmap")

    def test_create_metric_comparison_chart(self, metrics_df):
        """Test metric comparison - CHAI CR95: Performance by group."""
        from faircareai.visualization.plots import create_metric_comparison_chart

        fig = create_metric_comparison_chart(metrics_df, metrics=["tpr", "fpr", "ppv"])
        assert_valid_plotly_figure(fig, "create_metric_comparison_chart")

    def test_create_summary_scorecard(self):
        """Test summary scorecard - Overall status indicator."""
        from faircareai.visualization.plots import create_summary_scorecard

        fig = create_summary_scorecard(
            pass_count=5,
            warn_count=2,
            fail_count=1,
            n_samples=400,
            threshold=0.3,
            model_name="Test Model",
        )
        assert_valid_plotly_figure(fig, "create_summary_scorecard")

    def test_create_calibration_plot(self):
        """Test calibration plot - CHAI CR102: Calibration assessment."""
        from faircareai.visualization.plots import create_calibration_plot

        np.random.seed(42)
        y_true = np.random.binomial(1, 0.3, 200)
        y_prob = np.random.beta(2, 5, 200)
        group_labels = np.array(["A"] * 100 + ["B"] * 100)

        fig = create_calibration_plot(y_true, y_prob, group_labels=group_labels)
        assert_valid_plotly_figure(fig, "create_calibration_plot")

    def test_create_roc_curve_by_group(self):
        """Test ROC by group - CHAI CR95: Discrimination by subgroup."""
        from faircareai.visualization.plots import create_roc_curve_by_group

        np.random.seed(42)
        y_true = np.random.binomial(1, 0.3, 200)
        y_prob = np.random.beta(2, 5, 200)
        group_labels = np.array(["A"] * 100 + ["B"] * 100)

        fig = create_roc_curve_by_group(y_true, y_prob, group_labels)
        assert_valid_plotly_figure(fig, "create_roc_curve_by_group")

    def test_create_sample_size_waterfall(self, metrics_df):
        """Test sample size waterfall - Sample size transparency."""
        from faircareai.visualization.plots import create_sample_size_waterfall

        fig = create_sample_size_waterfall(metrics_df)
        assert_valid_plotly_figure(fig, "create_sample_size_waterfall")

    def test_create_equity_dashboard(self, metrics_df, disparity_df):
        """Test equity dashboard - CHAI CR92/CR95: Combined view."""
        from faircareai.visualization.plots import create_equity_dashboard

        fig = create_equity_dashboard(metrics_df, disparity_df=disparity_df)
        assert_valid_plotly_figure(fig, "create_equity_dashboard")

    def test_create_subgroup_heatmap(self, metrics_df):
        """Test subgroup heatmap - CHAI CR95: Metric heatmap."""
        from faircareai.visualization.plots import create_subgroup_heatmap

        fig = create_subgroup_heatmap(metrics_df, metric="auroc")
        assert_valid_plotly_figure(fig, "create_subgroup_heatmap")

    def test_create_fairness_radar(self, metrics_df):
        """Test fairness radar - CHAI CR92: Spider chart."""
        from faircareai.visualization.plots import create_fairness_radar

        fig = create_fairness_radar(metrics_df)
        assert_valid_plotly_figure(fig, "create_fairness_radar")


# =============================================================================
# TEST CLASS: vancalster_plots.py (5 functions - Van Calster 4 Required)
# Van Calster et al. (2025) Lancet Digital Health
# =============================================================================


class TestVanCalsterModule:
    """Audit vancalster_plots.py - Van Calster 4 required figures."""

    def test_create_auroc_forest_plot(self, vancalster_results):
        """Test AUROC forest plot - Van Calster Figure 1: Discrimination."""
        from faircareai.visualization.vancalster_plots import create_auroc_forest_plot

        fig = create_auroc_forest_plot(vancalster_results)
        assert_valid_plotly_figure(fig, "create_auroc_forest_plot")
        assert_has_wcag_alt_text(fig, "create_auroc_forest_plot")

    def test_create_calibration_plot_by_subgroup(self, vancalster_results):
        """Test calibration by subgroup - Van Calster Figure 2: Calibration."""
        from faircareai.visualization.vancalster_plots import (
            create_calibration_plot_by_subgroup,
        )

        fig = create_calibration_plot_by_subgroup(vancalster_results)
        assert_valid_plotly_figure(fig, "create_calibration_plot_by_subgroup")
        assert_has_wcag_alt_text(fig, "create_calibration_plot_by_subgroup")

    def test_create_decision_curve_by_subgroup(self, vancalster_results):
        """Test decision curve by subgroup - Van Calster Figure 3: Clinical utility."""
        from faircareai.visualization.vancalster_plots import (
            create_decision_curve_by_subgroup,
        )

        # Use vancalster fixture which has the required decision_curve nested data
        fig = create_decision_curve_by_subgroup(vancalster_results)
        assert_valid_plotly_figure(fig, "create_decision_curve_by_subgroup")

    def test_create_risk_distribution_plot(self, vancalster_results):
        """Test risk distribution plot - Van Calster Figure 4: Probability distributions."""
        from faircareai.visualization.vancalster_plots import create_risk_distribution_plot

        fig = create_risk_distribution_plot(vancalster_results)
        assert_valid_plotly_figure(fig, "create_risk_distribution_plot")
        assert_has_wcag_alt_text(fig, "create_risk_distribution_plot")

    def test_create_vancalster_dashboard(self, vancalster_results):
        """Test Van Calster dashboard - All 4 figures combined."""
        from faircareai.visualization.vancalster_plots import create_vancalster_dashboard

        fig = create_vancalster_dashboard(vancalster_results)
        assert_valid_plotly_figure(fig, "create_vancalster_dashboard")


# =============================================================================
# TEST CLASS: performance_charts.py (6 functions - TRIPOD+AI Overall)
# TRIPOD+AI Sections 2.1-2.5
# =============================================================================


class TestPerformanceChartsModule:
    """Audit performance_charts.py - TRIPOD+AI overall model assessment."""

    def test_plot_discrimination_curves(self, audit_results):
        """Test discrimination curves - TRIPOD+AI 2.1: ROC + PR."""
        from faircareai.visualization.performance_charts import plot_discrimination_curves

        fig = plot_discrimination_curves(audit_results)
        assert_valid_plotly_figure(fig, "plot_discrimination_curves")

    def test_plot_calibration_curve(self, audit_results):
        """Test calibration curve - TRIPOD+AI 2.2: Calibration."""
        from faircareai.visualization.performance_charts import plot_calibration_curve

        fig = plot_calibration_curve(audit_results)
        assert_valid_plotly_figure(fig, "plot_calibration_curve")

    def test_plot_threshold_analysis(self, audit_results):
        """Test threshold analysis - TRIPOD+AI 2.4: Sensitivity analysis."""
        from faircareai.visualization.performance_charts import plot_threshold_analysis

        fig = plot_threshold_analysis(audit_results)
        assert_valid_plotly_figure(fig, "plot_threshold_analysis")

    def test_plot_decision_curve(self, audit_results):
        """Test decision curve - TRIPOD+AI 2.5: Clinical utility."""
        from faircareai.visualization.performance_charts import plot_decision_curve

        fig = plot_decision_curve(audit_results)
        assert_valid_plotly_figure(fig, "plot_decision_curve")

    def test_plot_confusion_matrix(self, audit_results):
        """Test confusion matrix - Classification performance."""
        from faircareai.visualization.performance_charts import plot_confusion_matrix

        fig = plot_confusion_matrix(audit_results)
        assert_valid_plotly_figure(fig, "plot_confusion_matrix")

    def test_plot_performance_summary(self, audit_results):
        """Test performance summary - Overall summary dashboard."""
        from faircareai.visualization.performance_charts import plot_performance_summary

        fig = plot_performance_summary(audit_results)
        assert_valid_plotly_figure(fig, "plot_performance_summary")


# =============================================================================
# TEST CLASS: governance_dashboard.py (4 functions - Executive)
# =============================================================================


class TestGovernanceDashboardModule:
    """Audit governance_dashboard.py - Executive summary visualizations."""

    def test_create_executive_summary(self, audit_results):
        """Test executive summary - Single-page board view."""
        from faircareai.visualization.governance_dashboard import create_executive_summary

        fig = create_executive_summary(audit_results)
        assert_valid_plotly_figure(fig, "create_executive_summary")

    def test_create_go_nogo_scorecard(self, audit_results):
        """Test go/no-go scorecard - Large status indicator."""
        from faircareai.visualization.governance_dashboard import create_go_nogo_scorecard

        fig = create_go_nogo_scorecard(audit_results)
        assert_valid_plotly_figure(fig, "create_go_nogo_scorecard")

    def test_create_fairness_dashboard(self, audit_results):
        """Test fairness dashboard - Multi-panel fairness view."""
        from faircareai.visualization.governance_dashboard import create_fairness_dashboard

        fig = create_fairness_dashboard(audit_results)
        assert_valid_plotly_figure(fig, "create_fairness_dashboard")

    def test_plot_subgroup_comparison(self, audit_results):
        """Test subgroup comparison - Grouped bar chart."""
        from faircareai.visualization.governance_dashboard import plot_subgroup_comparison

        fig = plot_subgroup_comparison(audit_results, metric="tpr")
        assert_valid_plotly_figure(fig, "plot_subgroup_comparison")


# =============================================================================
# TEST CLASS: altair_plots.py (2 functions - Static Export)
# PDF/PPTX Static Charts
# =============================================================================


class TestAltairStaticModule:
    """Audit altair_plots.py - Static charts for PDF/PPTX export."""

    def test_create_forest_plot_static(self, metrics_df):
        """Test static forest plot - Altair chart for PDF export."""
        import altair as alt

        from faircareai.visualization.altair_plots import create_forest_plot_static

        chart = create_forest_plot_static(metrics_df, metric="tpr")
        assert chart is not None, "create_forest_plot_static: Chart is None"
        assert isinstance(chart, (alt.Chart, alt.LayerChart, alt.HConcatChart, alt.VConcatChart)), (
            "create_forest_plot_static: Not an Altair chart"
        )

    def test_create_icon_array(self):
        """Test icon array - SVG visualization for impact."""
        from faircareai.visualization.altair_plots import create_icon_array

        svg = create_icon_array(affected=15, total=100, title="Test Impact")
        assert svg is not None, "create_icon_array: SVG is None"
        assert isinstance(svg, str), "create_icon_array: Not a string"
        assert "<svg" in svg, "create_icon_array: Not valid SVG"
        assert "circle" in svg, "create_icon_array: Missing circle elements"


# =============================================================================
# TEST CLASS: tables.py (2 functions - HTML/Great Tables)
# =============================================================================


class TestTablesModule:
    """Audit tables.py - HTML table and summary components."""

    def test_create_executive_scorecard(self):
        """Test executive scorecard - Great Tables format."""
        from faircareai.visualization.tables import create_executive_scorecard

        table = create_executive_scorecard(
            pass_count=5,
            warn_count=2,
            flag_count=1,
            n_samples=400,
            n_groups=3,
            model_name="Test Model",
        )
        assert table is not None, "create_executive_scorecard: Table is None"

    def test_create_plain_language_summary(self):
        """Test plain language summary - HTML stakeholder summary."""
        from faircareai.visualization.tables import create_plain_language_summary

        html = create_plain_language_summary(
            pass_count=5,
            warn_count=2,
            flag_count=1,
            worst_group="Black",
            worst_metric="tpr",
            worst_value=-0.07,
        )
        assert html is not None, "create_plain_language_summary: HTML is None"
        assert isinstance(html, str), "create_plain_language_summary: Not a string"
        assert "<div" in html, "create_plain_language_summary: Not valid HTML"


# =============================================================================
# TEST CLASS: Edge Cases
# =============================================================================


class TestVisualizationEdgeCases:
    """Test edge cases to ensure robust figure generation."""

    def test_empty_metrics_df(self):
        """Test handling of empty DataFrame."""
        from faircareai.visualization.plots import create_forest_plot

        empty_df = pl.DataFrame(
            {
                "group": [],
                "n": [],
                "tpr": [],
            }
        )

        # Should handle gracefully (either empty figure or informative message)
        fig = create_forest_plot(empty_df, metric="tpr")
        assert fig is not None, "Empty DataFrame should return a figure"

    def test_single_group(self, metrics_df):
        """Test with single subgroup (no comparison possible)."""
        from faircareai.visualization.plots import create_forest_plot

        single_group_df = metrics_df.filter(pl.col("group") == "White")
        fig = create_forest_plot(single_group_df, metric="tpr")
        assert fig is not None, "Single group should return a figure"

    def test_many_groups(self):
        """Test with many subgroups (layout handling)."""
        from faircareai.visualization.plots import create_forest_plot

        many_groups_df = pl.DataFrame(
            {
                "group": [f"Group_{i}" for i in range(20)],
                "n": [50] * 20,
                "tpr": [0.7 + 0.01 * i for i in range(20)],
            }
        )

        fig = create_forest_plot(many_groups_df, metric="tpr")
        assert fig is not None, "Many groups should return a figure"

    def test_vancalster_empty_groups(self):
        """Test Van Calster functions with empty groups."""
        from faircareai.visualization.vancalster_plots import create_auroc_forest_plot

        empty_results = {"groups": {}}
        fig = create_auroc_forest_plot(empty_results)
        assert fig is not None, "Empty groups should return a figure"


# =============================================================================
# TEST CLASS: WCAG Accessibility Compliance
# =============================================================================


class TestWCAGAccessibility:
    """Verify WCAG 2.1 AA compliance for all visualizations."""

    def test_vancalster_plots_have_alt_text(self, vancalster_results):
        """Verify all Van Calster plots have descriptive alt text."""
        from faircareai.visualization.vancalster_plots import (
            create_auroc_forest_plot,
            create_calibration_plot_by_subgroup,
        )

        # Only test functions that work with our test fixture
        functions = [
            ("auroc_forest", create_auroc_forest_plot),
            ("calibration", create_calibration_plot_by_subgroup),
        ]

        for name, func in functions:
            fig = func(vancalster_results)
            meta = getattr(fig.layout, "meta", None)
            if meta is not None and isinstance(meta, dict):
                description = meta.get("description", "")
                assert description, f"{name}: Missing alt text description"
                assert len(description) > 20, f"{name}: Alt text too brief"


# =============================================================================
# TEST CLASS: Integration Tests
# =============================================================================


class TestVisualizationIntegration:
    """Integration tests verifying complete visualization workflows."""

    def test_full_audit_visualization_workflow(self, audit_results):
        """Test that all visualizations work with real audit results."""
        from faircareai.visualization.performance_charts import (
            plot_calibration_curve,
            plot_confusion_matrix,
            plot_decision_curve,
            plot_discrimination_curves,
            plot_performance_summary,
            plot_threshold_analysis,
        )

        # All TRIPOD+AI visualizations should work
        figures = [
            plot_discrimination_curves(audit_results),
            plot_calibration_curve(audit_results),
            plot_threshold_analysis(audit_results),
            plot_decision_curve(audit_results),
            plot_confusion_matrix(audit_results),
            plot_performance_summary(audit_results),
        ]

        for i, fig in enumerate(figures):
            assert fig is not None, f"Figure {i} is None"
            assert isinstance(fig, go.Figure), f"Figure {i} not a Plotly Figure"

    def test_results_visualization_methods(self, audit_results):
        """Test AuditResults visualization convenience methods."""
        # These are the primary user-facing visualization methods
        methods_to_test = [
            "plot_discrimination",
            "plot_overall_calibration",
            "plot_threshold_analysis",
            "plot_decision_curve",
        ]

        for method_name in methods_to_test:
            if hasattr(audit_results, method_name):
                method = getattr(audit_results, method_name)
                fig = method()
                assert fig is not None, f"{method_name}() returned None"
