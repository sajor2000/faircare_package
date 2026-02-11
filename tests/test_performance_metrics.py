"""
Tests for FairCareAI performance metrics module.

Tests cover:
- compute_overall_performance function
- compute_discrimination_metrics function
- compute_calibration_metrics function
- compute_classification_at_threshold function
- compute_threshold_analysis function
- compute_decision_curve_analysis function
- compute_confusion_matrix function
- compute_subgroup_performance function
"""

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import roc_auc_score

from faircareai.metrics.performance import (
    _interpret_calibration,
    compute_calibration_metrics,
    compute_classification_at_threshold,
    compute_confusion_matrix,
    compute_decision_curve_analysis,
    compute_discrimination_metrics,
    compute_overall_performance,
    compute_subgroup_performance,
    compute_threshold_analysis,
)


class TestComputeOverallPerformance:
    """Tests for compute_overall_performance function."""

    @pytest.fixture
    def sample_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Create sample data for testing."""
        np.random.seed(42)
        n = 500
        y_true = np.random.binomial(1, 0.3, n)
        y_prob = np.where(
            y_true == 1,
            np.clip(np.random.normal(0.6, 0.2, n), 0.01, 0.99),
            np.clip(np.random.normal(0.4, 0.2, n), 0.01, 0.99),
        )
        return y_true, y_prob

    def test_returns_dict(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that function returns a dictionary."""
        y_true, y_prob = sample_data
        result = compute_overall_performance(y_true, y_prob, bootstrap_ci=False)
        assert isinstance(result, dict)

    def test_contains_primary_threshold(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that primary threshold is included."""
        y_true, y_prob = sample_data
        result = compute_overall_performance(y_true, y_prob, threshold=0.6, bootstrap_ci=False)
        assert result["primary_threshold"] == 0.6

    def test_contains_discrimination(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that discrimination metrics are included."""
        y_true, y_prob = sample_data
        result = compute_overall_performance(y_true, y_prob, bootstrap_ci=False)
        assert "discrimination" in result
        assert "auroc" in result["discrimination"]
        assert "auprc" in result["discrimination"]

    def test_contains_calibration(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that calibration metrics are included."""
        y_true, y_prob = sample_data
        result = compute_overall_performance(y_true, y_prob, bootstrap_ci=False)
        assert "calibration" in result
        assert "brier_score" in result["calibration"]
        assert "calibration_slope" in result["calibration"]

    def test_contains_classification_at_threshold(
        self, sample_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test that classification at threshold is included."""
        y_true, y_prob = sample_data
        result = compute_overall_performance(y_true, y_prob, bootstrap_ci=False)
        assert "classification_at_threshold" in result
        assert "sensitivity" in result["classification_at_threshold"]
        assert "specificity" in result["classification_at_threshold"]

    def test_contains_threshold_analysis(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that threshold analysis is included."""
        y_true, y_prob = sample_data
        result = compute_overall_performance(y_true, y_prob, bootstrap_ci=False)
        assert "threshold_analysis" in result
        assert "thresholds" in result["threshold_analysis"]

    def test_contains_decision_curve(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that decision curve is included."""
        y_true, y_prob = sample_data
        result = compute_overall_performance(y_true, y_prob, bootstrap_ci=False)
        assert "decision_curve" in result
        assert "net_benefit_model" in result["decision_curve"]

    def test_contains_confusion_matrix(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that confusion matrix is included."""
        y_true, y_prob = sample_data
        result = compute_overall_performance(y_true, y_prob, bootstrap_ci=False)
        assert "confusion_matrix" in result
        assert "tp" in result["confusion_matrix"]
        assert "fp" in result["confusion_matrix"]

    def test_custom_thresholds_to_evaluate(
        self, sample_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test custom thresholds for analysis."""
        y_true, y_prob = sample_data
        custom_thresholds = [0.25, 0.5, 0.75]
        result = compute_overall_performance(
            y_true, y_prob, thresholds_to_evaluate=custom_thresholds, bootstrap_ci=False
        )
        assert result["threshold_analysis"]["thresholds"] == custom_thresholds

    def test_default_thresholds(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test default thresholds when none provided."""
        y_true, y_prob = sample_data
        result = compute_overall_performance(y_true, y_prob, bootstrap_ci=False)
        expected = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        assert result["threshold_analysis"]["thresholds"] == expected


class TestComputeDiscriminationMetrics:
    """Tests for compute_discrimination_metrics function."""

    @pytest.fixture
    def sample_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Create sample data for testing."""
        np.random.seed(42)
        n = 300
        y_true = np.random.binomial(1, 0.3, n)
        y_prob = np.where(
            y_true == 1,
            np.clip(np.random.normal(0.65, 0.15, n), 0.01, 0.99),
            np.clip(np.random.normal(0.35, 0.15, n), 0.01, 0.99),
        )
        return y_true, y_prob

    def test_returns_dict(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that function returns a dictionary."""
        y_true, y_prob = sample_data
        result = compute_discrimination_metrics(y_true, y_prob, bootstrap_ci=False)
        assert isinstance(result, dict)

    def test_auroc_in_range(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that AUROC is in valid range."""
        y_true, y_prob = sample_data
        result = compute_discrimination_metrics(y_true, y_prob, bootstrap_ci=False)
        assert 0.0 <= result["auroc"] <= 1.0

    def test_auroc_matches_sklearn(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that AUROC matches sklearn."""
        y_true, y_prob = sample_data
        result = compute_discrimination_metrics(y_true, y_prob, bootstrap_ci=False)
        expected = roc_auc_score(y_true, y_prob)
        assert result["auroc"] == pytest.approx(expected, rel=1e-6)

    def test_auprc_in_range(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that AUPRC is in valid range."""
        y_true, y_prob = sample_data
        result = compute_discrimination_metrics(y_true, y_prob, bootstrap_ci=False)
        assert 0.0 <= result["auprc"] <= 1.0

    def test_brier_score_in_range(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that Brier score is in valid range."""
        y_true, y_prob = sample_data
        result = compute_discrimination_metrics(y_true, y_prob, bootstrap_ci=False)
        assert 0.0 <= result["brier_score"] <= 1.0

    def test_roc_curve_data(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that ROC curve data is included."""
        y_true, y_prob = sample_data
        result = compute_discrimination_metrics(y_true, y_prob, bootstrap_ci=False)
        assert "roc_curve" in result
        assert "fpr" in result["roc_curve"]
        assert "tpr" in result["roc_curve"]
        assert "thresholds" in result["roc_curve"]

    def test_pr_curve_data(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that PR curve data is included."""
        y_true, y_prob = sample_data
        result = compute_discrimination_metrics(y_true, y_prob, bootstrap_ci=False)
        assert "pr_curve" in result
        assert "precision" in result["pr_curve"]
        assert "recall" in result["pr_curve"]

    def test_prevalence_computed(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that prevalence is computed."""
        y_true, y_prob = sample_data
        result = compute_discrimination_metrics(y_true, y_prob, bootstrap_ci=False)
        assert "prevalence" in result
        assert result["prevalence"] == pytest.approx(np.mean(y_true), rel=1e-6)

    def test_bootstrap_ci_computed(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that bootstrap CI is computed when requested."""
        y_true, y_prob = sample_data
        result = compute_discrimination_metrics(y_true, y_prob, bootstrap_ci=True, n_bootstrap=100)
        assert "auroc_ci_95" in result
        assert "auprc_ci_95" in result
        assert len(result["auroc_ci_95"]) == 2
        assert result["auroc_ci_95"][0] < result["auroc_ci_95"][1]

    def test_bootstrap_ci_format(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that CI format string is included."""
        y_true, y_prob = sample_data
        result = compute_discrimination_metrics(y_true, y_prob, bootstrap_ci=True, n_bootstrap=100)
        assert "auroc_ci_fmt" in result
        assert "95% CI:" in result["auroc_ci_fmt"]

    def test_no_bootstrap_ci_small_sample(self) -> None:
        """Test that bootstrap is skipped for small samples."""
        np.random.seed(42)
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.2, 0.9])
        result = compute_discrimination_metrics(y_true, y_prob, bootstrap_ci=True, n_bootstrap=100)
        # Should not have CI since n <= 10
        assert "auroc_ci_95" not in result


class TestComputeCalibrationMetrics:
    """Tests for compute_calibration_metrics function."""

    @pytest.fixture
    def well_calibrated_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Create well-calibrated data for testing."""
        np.random.seed(42)
        n = 500
        # Generate well-calibrated probabilities
        y_prob = np.random.uniform(0.1, 0.9, n)
        # Generate y_true based on probabilities
        y_true = (np.random.uniform(0, 1, n) < y_prob).astype(int)
        return y_true, y_prob

    @pytest.fixture
    def poorly_calibrated_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Create poorly calibrated data for testing."""
        np.random.seed(42)
        n = 500
        y_true = np.random.binomial(1, 0.3, n)
        # Overconfident predictions
        y_prob = np.where(y_true == 1, 0.95, 0.05)
        return y_true, y_prob

    def test_returns_dict(self, well_calibrated_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that function returns a dictionary."""
        y_true, y_prob = well_calibrated_data
        result = compute_calibration_metrics(y_true, y_prob)
        assert isinstance(result, dict)

    def test_brier_score_in_range(
        self, well_calibrated_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test that Brier score is in valid range."""
        y_true, y_prob = well_calibrated_data
        result = compute_calibration_metrics(y_true, y_prob)
        assert 0.0 <= result["brier_score"] <= 1.0

    def test_calibration_slope_computed(
        self, well_calibrated_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test that calibration slope is computed."""
        y_true, y_prob = well_calibrated_data
        result = compute_calibration_metrics(y_true, y_prob)
        # Just verify slope is computed and is a float
        assert "calibration_slope" in result
        assert isinstance(result["calibration_slope"], float)

    def test_calibration_curve_data(
        self, well_calibrated_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test that calibration curve data is included."""
        y_true, y_prob = well_calibrated_data
        result = compute_calibration_metrics(y_true, y_prob)
        assert "calibration_curve" in result
        assert "prob_true" in result["calibration_curve"]
        assert "prob_pred" in result["calibration_curve"]
        assert "n_bins" in result["calibration_curve"]

    def test_oe_ratio_computed(self, well_calibrated_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that O:E ratio is computed."""
        y_true, y_prob = well_calibrated_data
        result = compute_calibration_metrics(y_true, y_prob)
        assert "oe_ratio" in result

    def test_ici_computed(self, well_calibrated_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that ICI is computed."""
        y_true, y_prob = well_calibrated_data
        result = compute_calibration_metrics(y_true, y_prob)
        assert "ici" in result
        assert result["ici"] >= 0

    def test_e_max_computed(self, well_calibrated_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that E_max is computed."""
        y_true, y_prob = well_calibrated_data
        result = compute_calibration_metrics(y_true, y_prob)
        assert "e_max" in result
        assert result["e_max"] >= 0

    def test_interpretation_included(
        self, well_calibrated_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test that interpretation is included."""
        y_true, y_prob = well_calibrated_data
        result = compute_calibration_metrics(y_true, y_prob)
        assert "interpretation" in result
        assert isinstance(result["interpretation"], str)

    def test_oe_ratio_zero_when_no_positives(self) -> None:
        """Test O:E ratio is zero when no positive outcomes but expected > 0."""
        y_true = np.zeros(100)
        y_prob = np.random.uniform(0.1, 0.3, 100)
        result = compute_calibration_metrics(y_true, y_prob)
        assert result["oe_ratio"] == 0.0

    def test_oe_ratio_none_when_expected_zero(self) -> None:
        """Test O:E ratio is None when expected events are zero."""
        y_true = np.zeros(100)
        y_prob = np.zeros(100)
        result = compute_calibration_metrics(y_true, y_prob)
        assert result["oe_ratio"] is None


class TestInterpretCalibration:
    """Tests for _interpret_calibration function."""

    def test_good_calibration(self) -> None:
        """Test interpretation for good calibration."""
        result = _interpret_calibration(slope=1.0, brier=0.1)
        assert result == "Good calibration"

    def test_overfitting_detected(self) -> None:
        """Test interpretation for overfitting."""
        result = _interpret_calibration(slope=0.5, brier=0.1)
        assert "overfitting" in result.lower()

    def test_underfitting_detected(self) -> None:
        """Test interpretation for underfitting."""
        result = _interpret_calibration(slope=1.5, brier=0.1)
        assert "underfitting" in result.lower()

    def test_poor_brier_detected(self) -> None:
        """Test interpretation for poor Brier score."""
        result = _interpret_calibration(slope=1.0, brier=0.35)
        assert "poor" in result.lower() or "brier" in result.lower()


class TestComputeClassificationAtThreshold:
    """Tests for compute_classification_at_threshold function."""

    @pytest.fixture
    def sample_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Create sample data for testing."""
        np.random.seed(42)
        n = 200
        y_true = np.random.binomial(1, 0.3, n)
        y_prob = np.where(
            y_true == 1,
            np.clip(np.random.normal(0.7, 0.15, n), 0.01, 0.99),
            np.clip(np.random.normal(0.3, 0.15, n), 0.01, 0.99),
        )
        return y_true, y_prob

    def test_returns_dict(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that function returns a dictionary."""
        y_true, y_prob = sample_data
        result = compute_classification_at_threshold(y_true, y_prob, 0.5, bootstrap_ci=False)
        assert isinstance(result, dict)

    def test_threshold_stored(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that threshold is stored."""
        y_true, y_prob = sample_data
        result = compute_classification_at_threshold(y_true, y_prob, 0.6, bootstrap_ci=False)
        assert result["threshold"] == 0.6

    def test_sensitivity_in_range(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that sensitivity is in valid range."""
        y_true, y_prob = sample_data
        result = compute_classification_at_threshold(y_true, y_prob, 0.5, bootstrap_ci=False)
        assert 0.0 <= result["sensitivity"] <= 1.0

    def test_specificity_in_range(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that specificity is in valid range."""
        y_true, y_prob = sample_data
        result = compute_classification_at_threshold(y_true, y_prob, 0.5, bootstrap_ci=False)
        assert 0.0 <= result["specificity"] <= 1.0

    def test_ppv_in_range(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that PPV is in valid range."""
        y_true, y_prob = sample_data
        result = compute_classification_at_threshold(y_true, y_prob, 0.5, bootstrap_ci=False)
        assert 0.0 <= result["ppv"] <= 1.0

    def test_npv_in_range(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that NPV is in valid range."""
        y_true, y_prob = sample_data
        result = compute_classification_at_threshold(y_true, y_prob, 0.5, bootstrap_ci=False)
        assert 0.0 <= result["npv"] <= 1.0

    def test_f1_in_range(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that F1 is in valid range."""
        y_true, y_prob = sample_data
        result = compute_classification_at_threshold(y_true, y_prob, 0.5, bootstrap_ci=False)
        assert 0.0 <= result["f1_score"] <= 1.0

    def test_pct_flagged_in_range(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that percent flagged is in valid range."""
        y_true, y_prob = sample_data
        result = compute_classification_at_threshold(y_true, y_prob, 0.5, bootstrap_ci=False)
        assert 0.0 <= result["pct_flagged"] <= 100.0

    def test_confusion_matrix_values(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that confusion matrix values are present."""
        y_true, y_prob = sample_data
        result = compute_classification_at_threshold(y_true, y_prob, 0.5, bootstrap_ci=False)
        assert "tp" in result
        assert "fp" in result
        assert "tn" in result
        assert "fn" in result
        assert result["tp"] + result["fp"] + result["tn"] + result["fn"] == len(y_true)

    def test_nne_computed(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that NNE is computed."""
        y_true, y_prob = sample_data
        result = compute_classification_at_threshold(y_true, y_prob, 0.5, bootstrap_ci=False)
        assert "nne" in result
        if result["ppv"] > 0:
            expected_nne = 1 / result["ppv"]
            assert result["nne"] == pytest.approx(expected_nne, rel=1e-6)

    def test_bootstrap_ci_computed(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that bootstrap CI is computed when requested."""
        y_true, y_prob = sample_data
        result = compute_classification_at_threshold(
            y_true, y_prob, 0.5, bootstrap_ci=True, n_bootstrap=100
        )
        assert "sensitivity_ci_95" in result
        assert "specificity_ci_95" in result
        assert "ppv_ci_95" in result

    def test_threshold_affects_sensitivity(
        self, sample_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test that threshold affects sensitivity."""
        y_true, y_prob = sample_data
        result_low = compute_classification_at_threshold(y_true, y_prob, 0.3, bootstrap_ci=False)
        result_high = compute_classification_at_threshold(y_true, y_prob, 0.7, bootstrap_ci=False)
        # Higher threshold should give lower sensitivity
        assert result_high["sensitivity"] <= result_low["sensitivity"]


class TestComputeThresholdAnalysis:
    """Tests for compute_threshold_analysis function."""

    @pytest.fixture
    def sample_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Create sample data for testing."""
        np.random.seed(42)
        n = 200
        y_true = np.random.binomial(1, 0.3, n)
        y_prob = np.where(
            y_true == 1,
            np.clip(np.random.normal(0.7, 0.15, n), 0.01, 0.99),
            np.clip(np.random.normal(0.3, 0.15, n), 0.01, 0.99),
        )
        return y_true, y_prob

    def test_returns_dict(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that function returns a dictionary."""
        y_true, y_prob = sample_data
        result = compute_threshold_analysis(y_true, y_prob, [0.3, 0.5, 0.7])
        assert isinstance(result, dict)

    def test_thresholds_stored(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that thresholds are stored."""
        y_true, y_prob = sample_data
        thresholds = [0.3, 0.5, 0.7]
        result = compute_threshold_analysis(y_true, y_prob, thresholds)
        assert result["thresholds"] == thresholds

    def test_metrics_for_each_threshold(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that metrics are computed for each threshold."""
        y_true, y_prob = sample_data
        thresholds = [0.3, 0.5, 0.7]
        result = compute_threshold_analysis(y_true, y_prob, thresholds)
        assert len(result["metrics"]) == len(thresholds)

    def test_plot_data_included(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that plot data is included."""
        y_true, y_prob = sample_data
        result = compute_threshold_analysis(y_true, y_prob, [0.3, 0.5, 0.7])
        assert "plot_data" in result
        assert "sensitivity" in result["plot_data"]
        assert "specificity" in result["plot_data"]
        assert "ppv" in result["plot_data"]
        assert "npv" in result["plot_data"]
        assert "f1" in result["plot_data"]
        assert "pct_flagged" in result["plot_data"]

    def test_plot_data_lengths(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that plot data has correct lengths."""
        y_true, y_prob = sample_data
        thresholds = [0.3, 0.5, 0.7]
        result = compute_threshold_analysis(y_true, y_prob, thresholds)
        for key in ["sensitivity", "specificity", "ppv", "npv", "f1", "pct_flagged"]:
            assert len(result["plot_data"][key]) == len(thresholds)


class TestComputeDecisionCurveAnalysis:
    """Tests for compute_decision_curve_analysis function."""

    @pytest.fixture
    def sample_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Create sample data for testing."""
        np.random.seed(42)
        n = 200
        y_true = np.random.binomial(1, 0.3, n)
        y_prob = np.where(
            y_true == 1,
            np.clip(np.random.normal(0.7, 0.15, n), 0.01, 0.99),
            np.clip(np.random.normal(0.3, 0.15, n), 0.01, 0.99),
        )
        return y_true, y_prob

    def test_returns_dict(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that function returns a dictionary."""
        y_true, y_prob = sample_data
        result = compute_decision_curve_analysis(y_true, y_prob)
        assert isinstance(result, dict)

    def test_thresholds_included(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that thresholds are included."""
        y_true, y_prob = sample_data
        result = compute_decision_curve_analysis(y_true, y_prob)
        assert "thresholds" in result
        assert len(result["thresholds"]) > 0

    def test_custom_thresholds(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test custom thresholds."""
        y_true, y_prob = sample_data
        custom = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        result = compute_decision_curve_analysis(y_true, y_prob, thresholds=custom)
        assert len(result["thresholds"]) == len(custom)

    def test_net_benefit_model(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that net benefit for model is included."""
        y_true, y_prob = sample_data
        result = compute_decision_curve_analysis(y_true, y_prob)
        assert "net_benefit_model" in result
        assert len(result["net_benefit_model"]) == len(result["thresholds"])

    def test_net_benefit_all(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that net benefit for treat all is included."""
        y_true, y_prob = sample_data
        result = compute_decision_curve_analysis(y_true, y_prob)
        assert "net_benefit_all" in result
        assert len(result["net_benefit_all"]) == len(result["thresholds"])

    def test_net_benefit_none(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that net benefit for treat none is included."""
        y_true, y_prob = sample_data
        result = compute_decision_curve_analysis(y_true, y_prob)
        assert "net_benefit_none" in result
        # All values should be 0
        assert all(v == 0.0 for v in result["net_benefit_none"])

    def test_useful_range_included(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that useful range is included."""
        y_true, y_prob = sample_data
        result = compute_decision_curve_analysis(y_true, y_prob)
        assert "useful_range" in result
        assert "useful_range_summary" in result

    def test_prevalence_included(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that prevalence is included."""
        y_true, y_prob = sample_data
        result = compute_decision_curve_analysis(y_true, y_prob)
        assert "prevalence" in result
        assert result["prevalence"] == pytest.approx(np.mean(y_true), rel=1e-6)


class TestComputeConfusionMatrix:
    """Tests for compute_confusion_matrix function."""

    @pytest.fixture
    def sample_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Create sample data for testing."""
        np.random.seed(42)
        n = 100
        y_true = np.random.binomial(1, 0.3, n)
        y_prob = np.where(
            y_true == 1,
            np.clip(np.random.normal(0.7, 0.1, n), 0.01, 0.99),
            np.clip(np.random.normal(0.3, 0.1, n), 0.01, 0.99),
        )
        return y_true, y_prob

    def test_returns_dict(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that function returns a dictionary."""
        y_true, y_prob = sample_data
        result = compute_confusion_matrix(y_true, y_prob, 0.5)
        assert isinstance(result, dict)

    def test_threshold_stored(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that threshold is stored."""
        y_true, y_prob = sample_data
        result = compute_confusion_matrix(y_true, y_prob, 0.6)
        assert result["threshold"] == 0.6

    def test_matrix_shape(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that matrix has correct shape."""
        y_true, y_prob = sample_data
        result = compute_confusion_matrix(y_true, y_prob, 0.5)
        assert "matrix" in result
        assert len(result["matrix"]) == 2
        assert len(result["matrix"][0]) == 2

    def test_labels_included(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that labels are included."""
        y_true, y_prob = sample_data
        result = compute_confusion_matrix(y_true, y_prob, 0.5)
        assert "labels" in result
        assert result["labels"] == ["Negative", "Positive"]

    def test_component_values(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that component values are present."""
        y_true, y_prob = sample_data
        result = compute_confusion_matrix(y_true, y_prob, 0.5)
        assert "tp" in result
        assert "fp" in result
        assert "tn" in result
        assert "fn" in result

    def test_values_sum_to_n(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that confusion matrix values sum to n."""
        y_true, y_prob = sample_data
        result = compute_confusion_matrix(y_true, y_prob, 0.5)
        total = result["tp"] + result["fp"] + result["tn"] + result["fn"]
        assert total == len(y_true)


class TestComputeSubgroupPerformance:
    """Tests for compute_subgroup_performance function."""

    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        """Create sample DataFrame for testing."""
        np.random.seed(42)
        n = 500
        groups = np.random.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
        y_true = np.random.binomial(1, 0.3, n)
        y_prob = np.where(
            y_true == 1,
            np.clip(np.random.normal(0.65, 0.15, n), 0.01, 0.99),
            np.clip(np.random.normal(0.35, 0.15, n), 0.01, 0.99),
        )
        return pl.DataFrame(
            {
                "y_true": y_true,
                "y_prob": y_prob,
                "group": groups,
            }
        )

    def test_returns_dict(self, sample_df: pl.DataFrame) -> None:
        """Test that function returns a dictionary."""
        result = compute_subgroup_performance(
            sample_df, "y_true", "y_prob", "group", bootstrap_ci=False
        )
        assert isinstance(result, dict)

    def test_groups_included(self, sample_df: pl.DataFrame) -> None:
        """Test that all groups are included."""
        result = compute_subgroup_performance(
            sample_df, "y_true", "y_prob", "group", bootstrap_ci=False
        )
        assert "groups" in result
        assert "A" in result["groups"]
        assert "B" in result["groups"]
        assert "C" in result["groups"]

    def test_reference_determined(self, sample_df: pl.DataFrame) -> None:
        """Test that reference group is determined."""
        result = compute_subgroup_performance(
            sample_df, "y_true", "y_prob", "group", bootstrap_ci=False
        )
        assert "reference" in result
        # Largest group should be reference
        assert result["reference"] == "A"

    def test_specified_reference(self, sample_df: pl.DataFrame) -> None:
        """Test that specified reference is used."""
        result = compute_subgroup_performance(
            sample_df, "y_true", "y_prob", "group", reference="B", bootstrap_ci=False
        )
        assert result["reference"] == "B"

    def test_group_metrics(self, sample_df: pl.DataFrame) -> None:
        """Test that group metrics are computed."""
        result = compute_subgroup_performance(
            sample_df, "y_true", "y_prob", "group", bootstrap_ci=False
        )
        for group in ["A", "B", "C"]:
            assert "n" in result["groups"][group]
            assert "prevalence" in result["groups"][group]
            assert "auroc" in result["groups"][group]
            assert "tpr" in result["groups"][group]

    def test_is_reference_flag(self, sample_df: pl.DataFrame) -> None:
        """Test that is_reference flag is set."""
        result = compute_subgroup_performance(
            sample_df, "y_true", "y_prob", "group", reference="B", bootstrap_ci=False
        )
        assert result["groups"]["B"]["is_reference"] is True
        assert result["groups"]["A"]["is_reference"] is False

    def test_small_group_handling(self) -> None:
        """Test handling of small groups."""
        df = pl.DataFrame(
            {
                "y_true": [1, 0, 1] + [0, 1] * 50,
                "y_prob": [0.8, 0.2, 0.7] + [0.3, 0.7] * 50,
                "group": ["Small"] * 3 + ["Large"] * 100,
            }
        )
        result = compute_subgroup_performance(df, "y_true", "y_prob", "group", bootstrap_ci=False)
        assert "error" in result["groups"]["Small"]
        assert "Insufficient" in result["groups"]["Small"]["error"]

    def test_bootstrap_ci_for_auroc(self, sample_df: pl.DataFrame) -> None:
        """Test that bootstrap CI is computed for AUROC."""
        result = compute_subgroup_performance(
            sample_df, "y_true", "y_prob", "group", bootstrap_ci=True, n_bootstrap=50
        )
        # At least one group should have CI
        has_ci = any(
            "auroc_ci_95" in result["groups"][g]
            for g in result["groups"]
            if "error" not in result["groups"][g]
        )
        assert has_ci

    def test_metrics_in_range(self, sample_df: pl.DataFrame) -> None:
        """Test that metrics are in valid ranges."""
        result = compute_subgroup_performance(
            sample_df, "y_true", "y_prob", "group", bootstrap_ci=False
        )
        for group, metrics in result["groups"].items():
            if "error" not in metrics:
                assert 0.0 <= metrics["prevalence"] <= 1.0
                assert 0.0 <= metrics["auroc"] <= 1.0
                assert 0.0 <= metrics["tpr"] <= 1.0
                assert 0.0 <= metrics["fpr"] <= 1.0
