"""
Tests for FairCareAI audit results module.

Tests cover:
- AuditResults dataclass construction
- summary() method formatting
- Table 1 methods
- Visualization method delegation
- Export methods (to_json)
- JSON serialization helpers
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from faircareai.core.config import FairnessConfig, FairnessMetric, UseCaseType
from faircareai.core.results import AuditResults, _make_json_serializable


class TestAuditResultsConstruction:
    """Tests for AuditResults dataclass construction."""

    @pytest.fixture
    def basic_config(self) -> FairnessConfig:
        """Create basic FairnessConfig for testing."""
        return FairnessConfig(
            model_name="Test Model",
            model_version="1.0.0",
        )

    def test_minimal_construction(self, basic_config: FairnessConfig) -> None:
        """Test that AuditResults can be constructed with minimal args."""
        results = AuditResults(config=basic_config)
        assert results.config == basic_config
        assert results.descriptive_stats == {}
        assert results.overall_performance == {}
        assert results.flags == []

    def test_with_all_fields(self, basic_config: FairnessConfig) -> None:
        """Test construction with all fields populated."""
        results = AuditResults(
            config=basic_config,
            descriptive_stats={"cohort_overview": {"n_total": 1000}},
            overall_performance={"discrimination": {"auroc": 0.85}},
            subgroup_performance={"race": {"groups": {}}},
            fairness_metrics={"race": {"equalized_odds_diff": {"Black": 0.05}}},
            intersectional={"race_sex": {}},
            flags=[{"metric": "tpr", "status": "fail"}],
            governance_recommendation={"n_pass": 5, "n_warnings": 2, "n_errors": 1},
        )
        assert results.descriptive_stats["cohort_overview"]["n_total"] == 1000
        assert results.overall_performance["discrimination"]["auroc"] == 0.85
        assert len(results.flags) == 1

    def test_default_factory_isolation(self, basic_config: FairnessConfig) -> None:
        """Test that default factories create independent instances."""
        results1 = AuditResults(config=basic_config)
        results2 = AuditResults(config=basic_config)

        results1.flags.append({"test": "flag"})
        assert results2.flags == []  # Should be independent


class TestSummaryMethod:
    """Tests for AuditResults.summary() method."""

    @pytest.fixture
    def populated_results(self) -> AuditResults:
        """Create AuditResults with populated data for summary testing."""
        config = FairnessConfig(
            model_name="ICU Mortality Model",
            model_version="2.0.0",
            primary_fairness_metric=FairnessMetric.EQUALIZED_ODDS,
        )
        return AuditResults(
            config=config,
            descriptive_stats={
                "cohort_overview": {
                    "n_total": 10000,
                    "n_positive": 1500,
                    "prevalence_pct": "15.0%",
                }
            },
            overall_performance={
                "discrimination": {
                    "auroc": 0.852,
                    "auroc_ci_fmt": "(0.83-0.87)",
                    "auprc": 0.623,
                    "auprc_ci_fmt": "(0.58-0.66)",
                },
                "calibration": {
                    "brier_score": 0.0823,
                    "calibration_slope": 0.98,
                },
                "classification_at_threshold": {
                    "threshold": 0.30,
                    "sensitivity": 0.85,
                    "specificity": 0.72,
                    "ppv": 0.45,
                    "pct_flagged": 28.5,
                },
            },
            governance_recommendation={
                "n_pass": 12,
                "n_warnings": 3,
                "n_errors": 1,
            },
        )

    def test_summary_returns_string(self, populated_results: AuditResults) -> None:
        """Test that summary() returns a string."""
        result = populated_results.summary()
        assert isinstance(result, str)

    def test_summary_contains_model_info(self, populated_results: AuditResults) -> None:
        """Test that summary contains model name and version."""
        result = populated_results.summary()
        assert "ICU Mortality Model" in result
        assert "2.0.0" in result

    def test_summary_contains_cohort_stats(self, populated_results: AuditResults) -> None:
        """Test that summary contains cohort statistics."""
        result = populated_results.summary()
        assert "10,000" in result
        assert "1,500" in result

    def test_summary_contains_discrimination_metrics(self, populated_results: AuditResults) -> None:
        """Test that summary contains discrimination metrics."""
        result = populated_results.summary()
        assert "AUROC" in result
        assert "0.852" in result
        assert "AUPRC" in result

    def test_summary_contains_calibration_metrics(self, populated_results: AuditResults) -> None:
        """Test that summary contains calibration metrics."""
        result = populated_results.summary()
        assert "Brier Score" in result
        assert "Cal. Slope" in result

    def test_summary_contains_classification_metrics(self, populated_results: AuditResults) -> None:
        """Test that summary contains classification metrics."""
        result = populated_results.summary()
        assert "Sensitivity" in result
        assert "Specificity" in result
        assert "PPV" in result

    def test_summary_contains_governance_counts(self, populated_results: AuditResults) -> None:
        """Test that summary contains governance counts."""
        result = populated_results.summary()
        assert "12" in result  # n_pass
        assert "3" in result  # n_warnings
        assert "1" in result  # n_errors

    def test_summary_handles_missing_data(self) -> None:
        """Test that summary handles missing/None values gracefully."""
        config = FairnessConfig(
            model_name="Test",
            model_version="1.0",
        )
        results = AuditResults(config=config)
        result = results.summary()
        assert "N/A" in result
        assert isinstance(result, str)

    def test_summary_handles_zero_values(self) -> None:
        """Test that summary correctly displays zero values."""
        config = FairnessConfig(
            model_name="Test",
            model_version="1.0",
        )
        results = AuditResults(
            config=config,
            overall_performance={
                "classification_at_threshold": {
                    "sensitivity": 0.0,  # Zero is valid
                    "specificity": 0.0,
                }
            },
        )
        result = results.summary()
        assert "0.0%" in result

    def test_repr_returns_summary(self, populated_results: AuditResults) -> None:
        """Test that __repr__ returns the summary."""
        result = repr(populated_results)
        assert result == populated_results.summary()


class TestFormatHelpers:
    """Tests for internal format helper functions in summary()."""

    def test_fmt_handles_none(self) -> None:
        """Test fmt helper handles None values."""
        config = FairnessConfig(
            model_name="Test",
            model_version="1.0",
        )
        results = AuditResults(
            config=config,
            overall_performance={"discrimination": {"auroc": None}},
        )
        result = results.summary()
        assert "N/A" in result

    def test_fmt_handles_numeric(self) -> None:
        """Test fmt helper handles numeric values."""
        config = FairnessConfig(
            model_name="Test",
            model_version="1.0",
        )
        results = AuditResults(
            config=config,
            overall_performance={"discrimination": {"auroc": 0.85}},
        )
        result = results.summary()
        assert "0.850" in result


class TestTable1Methods:
    """Tests for Table 1 related methods."""

    @pytest.fixture
    def results_with_descriptive(self) -> AuditResults:
        """Create AuditResults with descriptive stats."""
        config = FairnessConfig(
            model_name="Test",
            model_version="1.0",
        )
        return AuditResults(
            config=config,
            descriptive_stats={
                "cohort_overview": {"n_total": 1000, "n_positive": 100},
                "attribute_distributions": {"race": {"White": 600, "Black": 400}},
            },
        )

    @patch("faircareai.metrics.descriptive.format_table1_text")
    def test_print_table1_calls_formatter(
        self, mock_format: MagicMock, results_with_descriptive: AuditResults
    ) -> None:
        """Test that print_table1 calls the formatter."""
        mock_format.return_value = "Table 1 text"
        result = results_with_descriptive.print_table1()
        mock_format.assert_called_once_with(results_with_descriptive.descriptive_stats)
        assert result == "Table 1 text"

    @patch("faircareai.metrics.descriptive.generate_table1_dataframe")
    def test_get_table1_dataframe_calls_generator(
        self, mock_generate: MagicMock, results_with_descriptive: AuditResults
    ) -> None:
        """Test that get_table1_dataframe calls the generator."""
        mock_df = pl.DataFrame({"col": [1, 2, 3]})
        mock_generate.return_value = mock_df
        result = results_with_descriptive.get_table1_dataframe()
        mock_generate.assert_called_once()
        assert isinstance(result, pl.DataFrame)


class TestVisualizationMethods:
    """Tests for visualization method delegation."""

    @pytest.fixture
    def basic_results(self) -> AuditResults:
        """Create basic AuditResults for viz testing."""
        config = FairnessConfig(
            model_name="Test",
            model_version="1.0",
        )
        return AuditResults(
            config=config,
            overall_performance={"primary_threshold": 0.5},
        )

    @patch("faircareai.visualization.performance_charts.plot_discrimination_curves")
    def test_plot_discrimination_delegates(
        self, mock_plot: MagicMock, basic_results: AuditResults
    ) -> None:
        """Test that plot_discrimination delegates correctly."""
        mock_plot.return_value = MagicMock()
        basic_results.plot_discrimination()
        mock_plot.assert_called_once_with(basic_results)

    @patch("faircareai.visualization.performance_charts.plot_calibration_curve")
    def test_plot_overall_calibration_delegates(
        self, mock_plot: MagicMock, basic_results: AuditResults
    ) -> None:
        """Test that plot_overall_calibration delegates correctly."""
        mock_plot.return_value = MagicMock()
        basic_results.plot_overall_calibration()
        mock_plot.assert_called_once_with(basic_results)

    @patch("faircareai.visualization.performance_charts.plot_threshold_analysis")
    def test_plot_threshold_analysis_uses_default(
        self, mock_plot: MagicMock, basic_results: AuditResults
    ) -> None:
        """Test that plot_threshold_analysis uses default threshold."""
        mock_plot.return_value = MagicMock()
        basic_results.plot_threshold_analysis()
        mock_plot.assert_called_once_with(basic_results, selected_threshold=0.5)

    @patch("faircareai.visualization.performance_charts.plot_threshold_analysis")
    def test_plot_threshold_analysis_custom_threshold(
        self, mock_plot: MagicMock, basic_results: AuditResults
    ) -> None:
        """Test that plot_threshold_analysis accepts custom threshold."""
        mock_plot.return_value = MagicMock()
        basic_results.plot_threshold_analysis(selected_threshold=0.7)
        mock_plot.assert_called_once_with(basic_results, selected_threshold=0.7)

    @patch("faircareai.visualization.performance_charts.plot_decision_curve")
    def test_plot_decision_curve_delegates(
        self, mock_plot: MagicMock, basic_results: AuditResults
    ) -> None:
        """Test that plot_decision_curve delegates correctly."""
        mock_plot.return_value = MagicMock()
        basic_results.plot_decision_curve()
        mock_plot.assert_called_once_with(basic_results)

    @patch("faircareai.visualization.governance_dashboard.create_fairness_dashboard")
    def test_plot_fairness_dashboard_delegates(
        self, mock_plot: MagicMock, basic_results: AuditResults
    ) -> None:
        """Test that plot_fairness_dashboard delegates correctly."""
        mock_plot.return_value = MagicMock()
        basic_results.plot_fairness_dashboard()
        mock_plot.assert_called_once_with(basic_results)

    @patch("faircareai.visualization.governance_dashboard.plot_subgroup_comparison")
    def test_plot_subgroup_performance_delegates(
        self, mock_plot: MagicMock, basic_results: AuditResults
    ) -> None:
        """Test that plot_subgroup_performance delegates correctly."""
        mock_plot.return_value = MagicMock()
        basic_results.plot_subgroup_performance(metric="tpr")
        mock_plot.assert_called_once_with(basic_results, metric="tpr")

    @patch("faircareai.visualization.governance_dashboard.create_executive_summary")
    def test_plot_executive_summary_delegates(
        self, mock_plot: MagicMock, basic_results: AuditResults
    ) -> None:
        """Test that plot_executive_summary delegates correctly."""
        mock_plot.return_value = MagicMock()
        basic_results.plot_executive_summary()
        mock_plot.assert_called_once_with(basic_results)

    @patch("faircareai.visualization.governance_dashboard.create_go_nogo_scorecard")
    def test_plot_go_nogo_scorecard_delegates(
        self, mock_plot: MagicMock, basic_results: AuditResults
    ) -> None:
        """Test that plot_go_nogo_scorecard delegates correctly."""
        mock_plot.return_value = MagicMock()
        basic_results.plot_go_nogo_scorecard()
        mock_plot.assert_called_once_with(basic_results)


class TestPlotCalibration:
    """Tests for plot_calibration with stratification."""

    @pytest.fixture
    def results_with_audit(self) -> AuditResults:
        """Create results with mock audit for stratified calibration."""
        config = FairnessConfig(
            model_name="Test",
            model_version="1.0",
        )
        results = AuditResults(config=config)

        # Mock audit object
        mock_audit = MagicMock()
        mock_audit.df = pl.DataFrame({"y_true": [0, 1], "y_prob": [0.3, 0.7], "race": ["A", "B"]})
        mock_audit.y_prob_col = "y_prob"
        mock_audit.y_true_col = "y_true"
        results._audit = mock_audit

        return results

    @patch("faircareai.visualization.plots.create_calibration_plot")
    def test_plot_calibration_with_stratification(
        self, mock_plot: MagicMock, results_with_audit: AuditResults
    ) -> None:
        """Test plot_calibration with by parameter."""
        mock_plot.return_value = MagicMock()
        results_with_audit.plot_calibration(by="race")
        mock_plot.assert_called_once()

    @patch("faircareai.visualization.performance_charts.plot_calibration_curve")
    def test_plot_calibration_overall(
        self, mock_plot: MagicMock, results_with_audit: AuditResults
    ) -> None:
        """Test plot_calibration without stratification falls back to overall."""
        mock_plot.return_value = MagicMock()
        results_with_audit.plot_calibration()
        mock_plot.assert_called_once()


class TestToJson:
    """Tests for to_json export method."""

    @pytest.fixture
    def export_results(self) -> AuditResults:
        """Create AuditResults for export testing."""
        config = FairnessConfig(
            model_name="Export Test Model",
            model_version="3.0.0",
            primary_fairness_metric=FairnessMetric.CALIBRATION,
            use_case_type=UseCaseType.RISK_COMMUNICATION,
            fairness_justification="Test justification",
            thresholds={"warn": 0.05, "fail": 0.10},
        )
        return AuditResults(
            config=config,
            descriptive_stats={"n_total": 1000},
            overall_performance={"auroc": 0.85},
            subgroup_performance={"race": {"White": 0.86}},
            fairness_metrics={"race": {"tpr_diff": 0.02}},
            intersectional={"race_sex": {}},
            flags=[{"metric": "fpr", "status": "warn"}],
            governance_recommendation={"n_pass": 10},
        )

    def test_to_json_creates_file(self, export_results: AuditResults) -> None:
        """Test that to_json creates a file."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            result = export_results.to_json(path)
            assert result == path
            assert path.exists()
        finally:
            path.unlink(missing_ok=True)

    def test_to_json_valid_content(self, export_results: AuditResults) -> None:
        """Test that to_json produces valid JSON."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            export_results.to_json(path)
            content = json.loads(path.read_text())
            assert "config" in content
            assert "descriptive_stats" in content
            assert "overall_performance" in content
            assert content["config"]["model_name"] == "Export Test Model"
        finally:
            path.unlink(missing_ok=True)

    def test_to_json_includes_config(self, export_results: AuditResults) -> None:
        """Test that JSON includes all config fields."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            export_results.to_json(path)
            content = json.loads(path.read_text())
            config = content["config"]
            assert config["model_name"] == "Export Test Model"
            assert config["model_version"] == "3.0.0"
            assert config["primary_fairness_metric"] == "calibration"
            assert config["use_case_type"] == "risk_communication"
        finally:
            path.unlink(missing_ok=True)


class TestMakeJsonSerializable:
    """Tests for _make_json_serializable helper function."""

    def test_dict_passthrough(self) -> None:
        """Test that dicts are processed recursively."""
        result = _make_json_serializable({"a": 1, "b": {"c": 2}})
        assert result == {"a": 1, "b": {"c": 2}}

    def test_list_passthrough(self) -> None:
        """Test that lists are processed recursively."""
        result = _make_json_serializable([1, 2, {"a": 3}])
        assert result == [1, 2, {"a": 3}]

    def test_polars_dataframe(self) -> None:
        """Test that Polars DataFrames are converted to dicts."""
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = _make_json_serializable(df)
        assert result == [{"a": 1, "b": 3}, {"a": 2, "b": 4}]

    def test_polars_series(self) -> None:
        """Test that Polars Series are converted to lists."""
        series = pl.Series("values", [1, 2, 3])
        result = _make_json_serializable(series)
        assert result == [1, 2, 3]

    def test_numpy_array(self) -> None:
        """Test that numpy arrays are converted to lists."""
        arr = np.array([1, 2, 3])
        result = _make_json_serializable(arr)
        assert result == [1, 2, 3]

    def test_numpy_scalar(self) -> None:
        """Test that numpy scalars are converted to Python types."""
        scalar = np.float64(0.5)
        result = _make_json_serializable(scalar)
        assert result == 0.5
        assert isinstance(result, float)

    def test_nested_structure(self) -> None:
        """Test complex nested structure with various types."""
        data = {
            "df": pl.DataFrame({"x": [1]}),
            "arr": np.array([2.0]),
            "nested": {"series": pl.Series("s", [3])},
        }
        result = _make_json_serializable(data)
        assert result["df"] == [{"x": 1}]
        assert result["arr"] == [2.0]
        assert result["nested"]["series"] == [3]

    def test_primitive_passthrough(self) -> None:
        """Test that primitives pass through unchanged."""
        assert _make_json_serializable(42) == 42
        assert _make_json_serializable("test") == "test"
        assert _make_json_serializable(True) is True
        assert _make_json_serializable(None) is None


class TestToAuditSummary:
    """Tests for _to_audit_summary conversion method."""

    @pytest.fixture
    def results_for_summary(self) -> AuditResults:
        """Create results for audit summary conversion."""
        config = FairnessConfig(
            model_name="Summary Test",
            model_version="1.0",
            decision_thresholds=[0.3],
            report_date="2024-01-15",
        )
        return AuditResults(
            config=config,
            descriptive_stats={"cohort_overview": {"n_total": 5000}},
            subgroup_performance={"race": {"groups": {"White": {}, "Black": {}}}},
            fairness_metrics={"race": {"equalized_odds_diff": {"White": 0.02, "Black": -0.08}}},
            governance_recommendation={"n_pass": 8, "n_warnings": 2, "n_errors": 0},
        )

    @patch("faircareai.reports.generator.AuditSummary")
    def test_conversion_populates_fields(
        self, mock_summary: MagicMock, results_for_summary: AuditResults
    ) -> None:
        """Test that conversion populates AuditSummary fields."""
        results_for_summary._to_audit_summary()
        mock_summary.assert_called_once()
        call_kwargs = mock_summary.call_args.kwargs
        assert call_kwargs["model_name"] == "Summary Test"
        assert call_kwargs["n_samples"] == 5000
        assert call_kwargs["threshold"] == 0.3

    @patch("faircareai.reports.generator.AuditSummary")
    def test_worst_disparity_detection(
        self, mock_summary: MagicMock, results_for_summary: AuditResults
    ) -> None:
        """Test that worst disparity is correctly identified."""
        results_for_summary._to_audit_summary()
        call_kwargs = mock_summary.call_args.kwargs
        # Black has -0.08, larger absolute value than White's 0.02
        assert "Black" in call_kwargs["worst_disparity_group"]
        assert call_kwargs["worst_disparity_value"] == -0.08

    def test_handles_none_disparities(self) -> None:
        """Test handling of None values in disparities."""
        config = FairnessConfig(
            model_name="Test",
            model_version="1.0",
        )
        results = AuditResults(
            config=config,
            fairness_metrics={"race": {"equalized_odds_diff": {"White": None, "Black": 0.05}}},
            governance_recommendation={"n_pass": 0, "n_warnings": 0, "n_errors": 0},
        )
        with patch("faircareai.reports.generator.AuditSummary"):
            # Should not raise with None values
            results._to_audit_summary()
