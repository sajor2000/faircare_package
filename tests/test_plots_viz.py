"""
Tests for FairCareAI visualization plots module.

Tests cover:
- Alt text generation functions
- add_source_annotation function
- _get_status_color helper
- create_forest_plot function
- create_disparity_heatmap function
- create_metric_comparison_chart function
- create_summary_scorecard function
- create_calibration_plot function
- create_roc_curve_by_group function
- create_sample_size_waterfall function
- create_equity_dashboard function
- create_subgroup_heatmap function
- create_fairness_radar function
"""

import numpy as np
import polars as pl
import pytest
from plotly.graph_objects import Figure

from faircareai.visualization.plots import (
    _get_status_color,
    add_source_annotation,
    create_calibration_plot,
    create_disparity_heatmap,
    create_equity_dashboard,
    create_fairness_radar,
    create_forest_plot,
    create_metric_comparison_chart,
    create_roc_curve_by_group,
    create_sample_size_waterfall,
    create_subgroup_heatmap,
    create_summary_scorecard,
    generate_calibration_alt_text,
    generate_forest_plot_alt_text,
    generate_heatmap_alt_text,
    generate_roc_alt_text,
)
from faircareai.visualization.themes import FAIRCAREAI_BRAND, SEMANTIC_COLORS


@pytest.fixture
def sample_metrics_df() -> pl.DataFrame:
    """Create sample metrics DataFrame for testing."""
    return pl.DataFrame(
        {
            "group": ["_overall", "Group A", "Group B", "Group C"],
            "tpr": [0.85, 0.90, 0.75, 0.82],
            "fpr": [0.15, 0.10, 0.20, 0.18],
            "ppv": [0.80, 0.85, 0.72, 0.78],
            "npv": [0.88, 0.92, 0.80, 0.85],
            "accuracy": [0.85, 0.88, 0.78, 0.82],
            "n": [1000, 400, 350, 250],
            "ci_lower": [0.82, 0.85, 0.68, 0.75],
            "ci_upper": [0.88, 0.95, 0.82, 0.89],
        }
    )


@pytest.fixture
def sample_disparity_df() -> pl.DataFrame:
    """Create sample disparity DataFrame for testing."""
    return pl.DataFrame(
        {
            "metric": ["tpr", "tpr", "fpr", "fpr"],
            "reference_group": ["Group A", "Group A", "Group A", "Group A"],
            "comparison_group": ["Group B", "Group C", "Group B", "Group C"],
            "difference": [-0.15, -0.08, 0.10, 0.08],
            "statistically_significant": [True, False, True, False],
        }
    )


class TestAddSourceAnnotation:
    """Tests for add_source_annotation function."""

    def test_adds_annotation(self) -> None:
        """Test that function returns figure unchanged (annotations moved to HTML)."""
        import plotly.graph_objects as go

        fig = go.Figure()
        result = add_source_annotation(fig)
        assert result is fig
        # Source annotations now added in HTML report footer, not Plotly inline

    def test_default_source_note(self) -> None:
        """Test that function accepts source note parameter."""
        import plotly.graph_objects as go

        fig = go.Figure()
        result = add_source_annotation(fig)
        assert result is not None
        assert isinstance(result, go.Figure)
        # Source note now handled in HTML report generation

    def test_custom_source_note(self) -> None:
        """Test custom source note parameter."""
        import plotly.graph_objects as go

        fig = go.Figure()
        result = add_source_annotation(fig, "Custom Source")
        assert result is not None
        assert isinstance(result, go.Figure)
        # Custom source notes now added in HTML report footer


class TestGetStatusColor:
    """Tests for _get_status_color helper function."""

    def test_pass_threshold(self) -> None:
        """Test pass color above threshold."""
        color = _get_status_color(0.85, threshold=0.8)
        assert color == SEMANTIC_COLORS["pass"]

    def test_warn_threshold(self) -> None:
        """Test warn color in between thresholds."""
        color = _get_status_color(0.75, threshold=0.8, warn_threshold=0.7)
        assert color == SEMANTIC_COLORS["warn"]

    def test_fail_threshold(self) -> None:
        """Test fail color below warn threshold."""
        color = _get_status_color(0.65, threshold=0.8, warn_threshold=0.7)
        assert color == SEMANTIC_COLORS["fail"]


class TestGenerateForestPlotAltText:
    """Tests for generate_forest_plot_alt_text function."""

    def test_empty_groups(self) -> None:
        """Test with DataFrame containing only _overall."""
        df = pl.DataFrame({"group": ["_overall"], "tpr": [0.85], "n": [100]})
        result = generate_forest_plot_alt_text(df, "tpr", "Test Title")
        assert "No data available" in result

    def test_normal_data(self, sample_metrics_df: pl.DataFrame) -> None:
        """Test with normal data."""
        result = generate_forest_plot_alt_text(sample_metrics_df, "tpr", "TPR by Group")
        assert "TPR by Group" in result
        assert "3 demographic groups" in result
        assert "75.0%" in result
        assert "90.0%" in result

    def test_flagged_groups(self, sample_metrics_df: pl.DataFrame) -> None:
        """Test when groups are flagged."""
        result = generate_forest_plot_alt_text(
            sample_metrics_df, "tpr", "Test", flagged_threshold=0.80
        )
        assert "Group B" in result

    def test_all_meet_threshold(self, sample_metrics_df: pl.DataFrame) -> None:
        """Test when all groups meet threshold."""
        result = generate_forest_plot_alt_text(
            sample_metrics_df, "tpr", "Test", flagged_threshold=0.5
        )
        assert "All groups meet" in result


class TestGenerateCalibrationAltText:
    """Tests for generate_calibration_alt_text function."""

    def test_no_groups(self) -> None:
        """Test without group labels."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.9])
        result = generate_calibration_alt_text(y_true, y_prob, title="Cal Curve")
        assert "Cal Curve" in result
        assert "4 samples" in result

    def test_with_groups(self) -> None:
        """Test with group labels."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.9])
        group_labels = np.array(["A", "A", "B", "B"])
        result = generate_calibration_alt_text(y_true, y_prob, group_labels, "Cal Curve")
        assert "2 demographic groups" in result


class TestGenerateRocAltText:
    """Tests for generate_roc_alt_text function."""

    def test_basic_alt_text(self) -> None:
        """Test ROC alt text generation."""
        np.random.seed(42)
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.2, 0.85])
        group_labels = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
        result = generate_roc_alt_text(y_true, y_prob, group_labels, "ROC Curves")
        assert "ROC Curves" in result
        assert "2 demographic groups" in result
        assert "AUC" in result


class TestGenerateHeatmapAltText:
    """Tests for generate_heatmap_alt_text function."""

    def test_basic_alt_text(self, sample_disparity_df: pl.DataFrame) -> None:
        """Test heatmap alt text generation."""
        result = generate_heatmap_alt_text(sample_disparity_df, "tpr", "Disparity Heatmap")
        assert "Disparity Heatmap" in result
        assert "TPR" in result

    def test_no_data(self) -> None:
        """Test with no data for metric."""
        df = pl.DataFrame(
            {
                "metric": ["fpr"],
                "reference_group": ["A"],
                "comparison_group": ["B"],
                "difference": [0.1],
                "statistically_significant": [True],
            }
        )
        result = generate_heatmap_alt_text(df, "tpr", "Test")
        assert "No disparity data" in result


class TestCreateForestPlot:
    """Tests for create_forest_plot function."""

    def test_returns_figure(self, sample_metrics_df: pl.DataFrame) -> None:
        """Test that function returns a Plotly Figure."""
        fig = create_forest_plot(sample_metrics_df)
        assert isinstance(fig, Figure)

    def test_empty_data(self) -> None:
        """Test with DataFrame containing only _overall."""
        df = pl.DataFrame({"group": ["_overall"], "tpr": [0.85], "n": [100]})
        fig = create_forest_plot(df)
        assert isinstance(fig, Figure)

    def test_missing_columns(self) -> None:
        """Test with missing required columns."""
        df = pl.DataFrame({"group": ["A", "B"], "other": [1, 2]})
        fig = create_forest_plot(df)
        assert isinstance(fig, Figure)
        # Should have annotation about missing columns
        assert len(fig.layout.annotations) >= 1

    def test_missing_metric(self) -> None:
        """Test with missing metric column."""
        df = pl.DataFrame({"group": ["A", "B"], "n": [100, 50], "other": [1, 2]})
        fig = create_forest_plot(df, metric="tpr")
        assert isinstance(fig, Figure)

    def test_custom_title(self, sample_metrics_df: pl.DataFrame) -> None:
        """Test with custom title."""
        fig = create_forest_plot(sample_metrics_df, title="Custom Title")
        assert "Custom Title" in fig.layout.title.text

    def test_with_subtitle(self, sample_metrics_df: pl.DataFrame) -> None:
        """Test with subtitle."""
        fig = create_forest_plot(sample_metrics_df, subtitle="My Subtitle")
        assert "My Subtitle" in fig.layout.title.text

    def test_ghosting_enabled(self, sample_metrics_df: pl.DataFrame) -> None:
        """Test with ghosting enabled."""
        fig = create_forest_plot(sample_metrics_df, enable_ghosting=True)
        assert isinstance(fig, Figure)

    def test_ghosting_disabled(self, sample_metrics_df: pl.DataFrame) -> None:
        """Test with ghosting disabled."""
        fig = create_forest_plot(sample_metrics_df, enable_ghosting=False)
        assert isinstance(fig, Figure)

    def test_safe_zone_shown(self, sample_metrics_df: pl.DataFrame) -> None:
        """Test that safe zone is shown."""
        fig = create_forest_plot(sample_metrics_df, show_safe_zone=True)
        assert isinstance(fig, Figure)

    def test_reference_line(self, sample_metrics_df: pl.DataFrame) -> None:
        """Test with reference line."""
        fig = create_forest_plot(sample_metrics_df, reference_line=0.8)
        assert isinstance(fig, Figure)

    def test_alt_text_in_meta(self, sample_metrics_df: pl.DataFrame) -> None:
        """Test that alt text is in metadata."""
        fig = create_forest_plot(sample_metrics_df)
        assert "description" in fig.layout.meta


class TestCreateDisparityHeatmap:
    """Tests for create_disparity_heatmap function."""

    def test_returns_figure(self, sample_disparity_df: pl.DataFrame) -> None:
        """Test that function returns a Plotly Figure."""
        fig = create_disparity_heatmap(sample_disparity_df)
        assert isinstance(fig, Figure)

    def test_missing_columns(self) -> None:
        """Test with missing required columns."""
        df = pl.DataFrame({"metric": ["tpr"], "other": [1]})
        fig = create_disparity_heatmap(df)
        assert isinstance(fig, Figure)

    def test_no_data_for_metric(self, sample_disparity_df: pl.DataFrame) -> None:
        """Test with no data for the specified metric."""
        fig = create_disparity_heatmap(sample_disparity_df, metric="nonexistent")
        assert isinstance(fig, Figure)

    def test_custom_title(self, sample_disparity_df: pl.DataFrame) -> None:
        """Test with custom title."""
        fig = create_disparity_heatmap(sample_disparity_df, title="Custom Title")
        assert "Custom Title" in fig.layout.title.text


class TestCreateMetricComparisonChart:
    """Tests for create_metric_comparison_chart function."""

    def test_returns_figure(self, sample_metrics_df: pl.DataFrame) -> None:
        """Test that function returns a Plotly Figure."""
        fig = create_metric_comparison_chart(sample_metrics_df)
        assert isinstance(fig, Figure)

    def test_missing_group_column(self) -> None:
        """Test with missing group column."""
        df = pl.DataFrame({"other": [1, 2], "tpr": [0.8, 0.9]})
        fig = create_metric_comparison_chart(df)
        assert isinstance(fig, Figure)

    def test_empty_data(self) -> None:
        """Test with only _overall group."""
        df = pl.DataFrame({"group": ["_overall"], "tpr": [0.85], "fpr": [0.15]})
        fig = create_metric_comparison_chart(df)
        assert isinstance(fig, Figure)

    def test_custom_metrics(self, sample_metrics_df: pl.DataFrame) -> None:
        """Test with custom metrics list."""
        fig = create_metric_comparison_chart(sample_metrics_df, metrics=["tpr", "ppv"])
        assert isinstance(fig, Figure)

    def test_custom_title(self, sample_metrics_df: pl.DataFrame) -> None:
        """Test with custom title."""
        fig = create_metric_comparison_chart(sample_metrics_df, title="Custom Title")
        assert "Custom Title" in fig.layout.title.text


class TestCreateSummaryScorecard:
    """Tests for create_summary_scorecard function."""

    def test_returns_figure(self) -> None:
        """Test that function returns a Plotly Figure."""
        fig = create_summary_scorecard(5, 3, 2, 1000, 0.8)
        assert isinstance(fig, Figure)

    def test_all_pass(self) -> None:
        """Test with all pass counts."""
        fig = create_summary_scorecard(10, 0, 0, 1000, 0.8)
        assert isinstance(fig, Figure)

    def test_custom_model_name(self) -> None:
        """Test with custom model name."""
        fig = create_summary_scorecard(5, 3, 2, 1000, 0.8, model_name="ICU Mortality")
        assert "ICU Mortality" in fig.layout.title.text

    def test_large_sample_size(self) -> None:
        """Test with large sample size."""
        fig = create_summary_scorecard(5, 3, 2, 1000000, 0.8)
        assert isinstance(fig, Figure)


class TestCreateCalibrationPlot:
    """Tests for create_calibration_plot function."""

    def test_returns_figure_no_groups(self) -> None:
        """Test that function returns a Plotly Figure without groups."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.random(100)
        fig = create_calibration_plot(y_true, y_prob)
        assert isinstance(fig, Figure)

    def test_returns_figure_with_groups(self) -> None:
        """Test that function returns a Plotly Figure with groups."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.random(100)
        group_labels = np.array(["A"] * 50 + ["B"] * 50)
        fig = create_calibration_plot(y_true, y_prob, group_labels)
        assert isinstance(fig, Figure)

    def test_custom_bins(self) -> None:
        """Test with custom number of bins."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.random(100)
        fig = create_calibration_plot(y_true, y_prob, n_bins=5)
        assert isinstance(fig, Figure)

    def test_custom_title(self) -> None:
        """Test with custom title."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.random(100)
        fig = create_calibration_plot(y_true, y_prob, title="Custom Title")
        assert "Custom Title" in fig.layout.title.text

    def test_empty_group_mask(self) -> None:
        """Test handling of empty group masks."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.9])
        # Create group labels where "C" doesn't exist
        group_labels = np.array(["A", "A", "B", "B"])
        fig = create_calibration_plot(y_true, y_prob, group_labels)
        assert isinstance(fig, Figure)


class TestCreateRocCurveByGroup:
    """Tests for create_roc_curve_by_group function."""

    def test_returns_figure(self) -> None:
        """Test that function returns a Plotly Figure."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.random(100)
        group_labels = np.array(["A"] * 50 + ["B"] * 50)
        fig = create_roc_curve_by_group(y_true, y_prob, group_labels)
        assert isinstance(fig, Figure)

    def test_custom_title(self) -> None:
        """Test with custom title."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.random(100)
        group_labels = np.array(["A"] * 50 + ["B"] * 50)
        fig = create_roc_curve_by_group(y_true, y_prob, group_labels, title="Custom Title")
        assert "Custom Title" in fig.layout.title.text

    def test_single_class_group(self) -> None:
        """Test handling of group with single class."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1])  # A has only 0s
        y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        group_labels = np.array(["A", "A", "A", "A", "A", "B", "B", "B"])
        # A has only negative, should be skipped
        y_true[:5] = 0  # Ensure A has single class
        fig = create_roc_curve_by_group(y_true, y_prob, group_labels)
        assert isinstance(fig, Figure)


class TestCreateSampleSizeWaterfall:
    """Tests for create_sample_size_waterfall function."""

    def test_returns_figure(self, sample_metrics_df: pl.DataFrame) -> None:
        """Test that function returns a Plotly Figure."""
        fig = create_sample_size_waterfall(sample_metrics_df)
        assert isinstance(fig, Figure)

    def test_custom_title(self, sample_metrics_df: pl.DataFrame) -> None:
        """Test with custom title."""
        fig = create_sample_size_waterfall(sample_metrics_df, title="Custom Title")
        assert "Custom Title" in fig.layout.title.text

    def test_color_coding(self, sample_metrics_df: pl.DataFrame) -> None:
        """Test that colors are coded by sample size."""
        fig = create_sample_size_waterfall(sample_metrics_df)
        assert isinstance(fig, Figure)


class TestCreateEquityDashboard:
    """Tests for create_equity_dashboard function."""

    def test_returns_figure(self, sample_metrics_df: pl.DataFrame) -> None:
        """Test that function returns a Plotly Figure."""
        fig = create_equity_dashboard(sample_metrics_df)
        assert isinstance(fig, Figure)

    def test_with_disparity_df(
        self, sample_metrics_df: pl.DataFrame, sample_disparity_df: pl.DataFrame
    ) -> None:
        """Test with disparity DataFrame provided."""
        fig = create_equity_dashboard(sample_metrics_df, sample_disparity_df)
        assert isinstance(fig, Figure)

    def test_custom_metric(self, sample_metrics_df: pl.DataFrame) -> None:
        """Test with custom metric."""
        fig = create_equity_dashboard(sample_metrics_df, metric="fpr")
        assert isinstance(fig, Figure)


class TestCreateSubgroupHeatmap:
    """Tests for create_subgroup_heatmap function."""

    def test_missing_columns(self, sample_metrics_df: pl.DataFrame) -> None:
        """Test with missing required columns (no attribute)."""
        fig = create_subgroup_heatmap(sample_metrics_df)
        assert isinstance(fig, Figure)

    def test_with_attribute(self) -> None:
        """Test with attribute column present."""
        df = pl.DataFrame(
            {
                "group": ["A", "B", "C", "D"],
                "attribute": ["Race", "Race", "Sex", "Sex"],
                "tpr": [0.85, 0.75, 0.90, 0.82],
                "n": [100, 50, 100, 80],
            }
        )
        fig = create_subgroup_heatmap(df)
        assert isinstance(fig, Figure)

    def test_empty_data(self) -> None:
        """Test with only _overall group."""
        df = pl.DataFrame(
            {
                "group": ["_overall"],
                "attribute": ["Overall"],
                "tpr": [0.85],
                "n": [100],
            }
        )
        fig = create_subgroup_heatmap(df)
        assert isinstance(fig, Figure)

    def test_custom_title(self) -> None:
        """Test with custom title."""
        df = pl.DataFrame(
            {
                "group": ["A", "B"],
                "attribute": ["Race", "Race"],
                "tpr": [0.85, 0.75],
                "n": [100, 50],
            }
        )
        fig = create_subgroup_heatmap(df, title="Custom Title")
        assert "Custom Title" in fig.layout.title.text


class TestCreateFairnessRadar:
    """Tests for create_fairness_radar function."""

    def test_returns_figure(self, sample_metrics_df: pl.DataFrame) -> None:
        """Test that function returns a Plotly Figure."""
        fig = create_fairness_radar(sample_metrics_df)
        assert isinstance(fig, Figure)

    def test_custom_title(self, sample_metrics_df: pl.DataFrame) -> None:
        """Test with custom title."""
        fig = create_fairness_radar(sample_metrics_df, title="Custom Title")
        assert "Custom Title" in fig.layout.title.text

    def test_handles_missing_metrics(self) -> None:
        """Test handling of missing metrics columns."""
        df = pl.DataFrame(
            {
                "group": ["A", "B"],
                "tpr": [0.85, 0.75],
                "n": [100, 50],
            }
        )
        fig = create_fairness_radar(df)
        assert isinstance(fig, Figure)
