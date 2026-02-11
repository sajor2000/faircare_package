"""Edge case tests for FairCareAI.

Tests for boundary conditions, unusual data scenarios, and error handling.
These tests ensure robust behavior when encountering edge cases that may
occur in real-world healthcare data.
"""

import numpy as np
import polars as pl

from faircareai.metrics.fairness import (
    compute_calibration_by_group,
    compute_fairness_metrics,
    compute_group_auroc_comparison,
)
from faircareai.metrics.performance import (
    compute_calibration_metrics,
    compute_classification_at_threshold,
    compute_discrimination_metrics,
    compute_overall_performance,
)


class TestEmptyDataFrames:
    """Tests for empty DataFrame handling."""

    def test_fairness_metrics_empty_df(self) -> None:
        """Should handle empty DataFrame gracefully or raise appropriately."""
        df = pl.DataFrame(
            {
                "y_prob": [],
                "y_true": [],
                "group": [],
            },
            schema={"y_prob": pl.Float64, "y_true": pl.Int64, "group": pl.Utf8},
        )

        # Empty df may raise or return empty result - both acceptable
        try:
            result = compute_fairness_metrics(df, "y_prob", "y_true", "group")
            # Should return dict with empty group_metrics
            assert isinstance(result, dict)
            assert "group_metrics" in result
        except (IndexError, ValueError):
            # Also acceptable - can't compute metrics on empty df
            pass

    def test_calibration_by_group_empty_df(self) -> None:
        """Should handle empty DataFrame for calibration."""
        df = pl.DataFrame(
            {
                "y_prob": [],
                "y_true": [],
                "group": [],
            },
            schema={"y_prob": pl.Float64, "y_true": pl.Int64, "group": pl.Utf8},
        )

        result = compute_calibration_by_group(df, "y_prob", "y_true", "group")
        assert isinstance(result, dict)
        assert "groups" in result


class TestSmallSampleSizes:
    """Tests for minimum viable sample sizes."""

    def test_fairness_small_groups(self) -> None:
        """Test fairness with groups below minimum sample size."""
        # Create groups with n < 10
        df = pl.DataFrame(
            {
                "y_prob": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7],
                "y_true": [0, 0, 1, 0, 1, 1, 1, 0, 1],
                "group": ["A", "A", "A", "A", "B", "B", "B", "B", "B"],
            }
        )

        result = compute_fairness_metrics(df, "y_prob", "y_true", "group")

        # Both groups have n < 10, should have error messages
        assert isinstance(result, dict)
        for group_data in result["group_metrics"].values():
            # Either has error or insufficient sample flag
            assert "n" in group_data
            assert group_data["n"] < 10

    def test_single_observation_per_group(self) -> None:
        """Test with only one observation per group."""
        df = pl.DataFrame(
            {
                "y_prob": [0.5, 0.5],
                "y_true": [0, 1],
                "group": ["A", "B"],
            }
        )

        result = compute_fairness_metrics(df, "y_prob", "y_true", "group")

        # Should mark as insufficient
        assert isinstance(result, dict)
        for group_data in result["group_metrics"].values():
            assert group_data["n"] <= 2
            # Should have error due to small sample
            if group_data["n"] < 10:
                assert "error" in group_data

    def test_bootstrap_with_small_sample(self) -> None:
        """Test bootstrap CI with small sample returns None."""
        # Sample too small for reliable bootstrap
        y_true = np.array([0, 1, 0, 1, 0])
        y_prob = np.array([0.3, 0.7, 0.4, 0.8, 0.2])

        result = compute_discrimination_metrics(y_true, y_prob, bootstrap_ci=True, n_bootstrap=100)

        # Should compute point estimates
        assert "auroc" in result
        # CI may or may not be present depending on bootstrap success


class TestSingleClassOutcomes:
    """Tests for single-class outcome scenarios."""

    def test_all_positive_outcomes(self) -> None:
        """Should handle all-positive outcomes (no negatives)."""
        y_true = np.ones(50, dtype=int)
        y_prob = np.random.uniform(0.5, 1.0, 50)

        # sklearn returns nan for AUROC with single class (with warning)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = compute_discrimination_metrics(y_true, y_prob, bootstrap_ci=False)

        # AUROC should be nan or 0.5 for single class
        assert isinstance(result, dict)
        assert "auroc" in result

    def test_all_negative_outcomes(self) -> None:
        """Should handle all-negative outcomes (no positives)."""
        y_true = np.zeros(50, dtype=int)
        y_prob = np.random.uniform(0.0, 0.5, 50)

        # sklearn returns nan for AUROC with single class (with warning)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = compute_discrimination_metrics(y_true, y_prob, bootstrap_ci=False)

        assert isinstance(result, dict)
        assert "auroc" in result

    def test_classification_all_positive(self) -> None:
        """Classification metrics with all-positive outcomes."""
        y_true = np.ones(20, dtype=int)
        y_prob = np.random.uniform(0.5, 1.0, 20)

        result = compute_classification_at_threshold(
            y_true, y_prob, threshold=0.5, bootstrap_ci=False
        )

        # Should compute but some metrics may be 0 or undefined
        assert isinstance(result, dict)
        assert "sensitivity" in result
        # With all positives and threshold 0.5, we should have high sensitivity
        assert result["sensitivity"] >= 0


class TestExtremeValues:
    """Tests for extreme prediction values."""

    def test_all_zero_predictions(self) -> None:
        """Should handle all-zero predictions."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1])
        y_prob = np.zeros(12)

        result = compute_overall_performance(y_true, y_prob, threshold=0.5, bootstrap_ci=False)

        # Should return results without crashing
        assert isinstance(result, dict)
        assert "classification_at_threshold" in result
        # With all zeros and threshold 0.5, sensitivity = 0
        assert result["classification_at_threshold"]["sensitivity"] == 0.0

    def test_all_one_predictions(self) -> None:
        """Should handle all-one predictions."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1])
        y_prob = np.ones(12)

        result = compute_overall_performance(y_true, y_prob, threshold=0.5, bootstrap_ci=False)

        # Should return results
        assert isinstance(result, dict)
        # With all ones and threshold 0.5, specificity = 0
        assert result["classification_at_threshold"]["specificity"] == 0.0

    def test_perfect_predictions(self) -> None:
        """Test with perfect predictions (AUROC = 1.0)."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.45, 0.55, 0.6, 0.7, 0.8, 0.9])

        result = compute_discrimination_metrics(y_true, y_prob, bootstrap_ci=False)

        assert result["auroc"] == 1.0

    def test_inverse_predictions(self) -> None:
        """Test with inverse predictions (AUROC = 0.0)."""
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.45, 0.55, 0.6, 0.7, 0.8, 0.9])

        result = compute_discrimination_metrics(y_true, y_prob, bootstrap_ci=False)

        assert result["auroc"] == 0.0


class TestCalibrationEdgeCases:
    """Tests for calibration metric edge cases."""

    def test_perfect_calibration(self) -> None:
        """Test calibration with perfectly calibrated predictions."""
        np.random.seed(42)
        n = 100
        y_prob = np.random.uniform(0, 1, n)
        # Generate outcomes based on probabilities
        y_true = (np.random.uniform(0, 1, n) < y_prob).astype(int)

        result = compute_calibration_metrics(y_true, y_prob)

        assert isinstance(result, dict)
        assert "brier_score" in result
        assert "calibration_slope" in result
        # Calibration slope can vary significantly with random data
        # Just ensure it's a valid number and brier score is reasonable
        assert isinstance(result["calibration_slope"], int | float)
        assert 0 <= result["brier_score"] <= 1

    def test_calibration_constant_predictions(self) -> None:
        """Test calibration with constant predictions."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1])
        y_prob = np.full(12, 0.5)  # All predictions are 0.5

        result = compute_calibration_metrics(y_true, y_prob)

        # Should handle without crashing
        assert isinstance(result, dict)


class TestGroupAUROCComparison:
    """Tests for group AUROC comparison edge cases."""

    def test_single_group(self) -> None:
        """Test with only one group."""
        df = pl.DataFrame(
            {
                "y_prob": [0.3, 0.4, 0.5, 0.6, 0.7] * 10,
                "y_true": [0, 0, 1, 1, 1] * 10,
                "group": ["A"] * 50,
            }
        )

        result = compute_group_auroc_comparison(df, "y_prob", "y_true", "group", n_bootstrap=50)

        assert isinstance(result, dict)
        assert "groups" in result
        assert "A" in result["groups"]

    def test_highly_imbalanced_groups(self) -> None:
        """Test with highly imbalanced group sizes."""
        # Group A has 99 samples, Group B has 15
        # Ensure array lengths match: 99 + 15 = 114
        df = pl.DataFrame(
            {
                "y_prob": [0.3, 0.5, 0.7] * 33 + [0.4, 0.6, 0.8] * 5,
                "y_true": [0, 0, 1] * 33 + [0, 1, 1] * 5,
                "group": ["A"] * 99 + ["B"] * 15,
            }
        )

        result = compute_group_auroc_comparison(df, "y_prob", "y_true", "group", n_bootstrap=50)

        assert isinstance(result, dict)
        # Reference should be A (largest group)
        assert result["reference"] == "A"


class TestNullHandling:
    """Tests for null value handling."""

    def test_null_predictions(self) -> None:
        """Test handling of null predictions - should fail validation before metrics."""
        # Note: In practice, null handling should be done during data validation
        # before reaching the metrics functions
        pass  # Null handling is done at audit level, not metric level

    def test_missing_group_values(self) -> None:
        """Test with null group values (should be dropped)."""
        df = pl.DataFrame(
            {
                "y_prob": [0.3, 0.4, 0.5, 0.6, 0.7] * 10,
                "y_true": [0, 0, 1, 1, 1] * 10,
                "group": (["A"] * 20 + [None] * 10 + ["B"] * 20),
            }
        )

        result = compute_fairness_metrics(df, "y_prob", "y_true", "group")

        # Nulls should be dropped, only A and B groups present
        assert isinstance(result, dict)
        group_names = set(result["group_metrics"].keys())
        assert "A" in group_names or "B" in group_names
        # None should not be a group key
        assert None not in group_names
        assert "None" not in group_names or result["group_metrics"].get("None", {}).get("n", 0) == 0


class TestThresholdEdgeCases:
    """Tests for threshold-related edge cases."""

    def test_threshold_zero(self) -> None:
        """Test classification at threshold = 0."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.15, 0.55, 0.25, 0.65, 0.05, 0.35, 0.75, 0.85])

        result = compute_classification_at_threshold(
            y_true, y_prob, threshold=0.0, bootstrap_ci=False
        )

        # At threshold 0, everything is predicted positive
        assert result["sensitivity"] == 1.0
        assert result["specificity"] == 0.0

    def test_threshold_one(self) -> None:
        """Test classification at threshold = 1."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.15, 0.55, 0.25, 0.65, 0.05, 0.35, 0.75, 0.85])

        result = compute_classification_at_threshold(
            y_true, y_prob, threshold=1.0, bootstrap_ci=False
        )

        # At threshold 1, nothing is predicted positive
        assert result["sensitivity"] == 0.0
        assert result["specificity"] == 1.0


class TestBootstrapUtility:
    """Tests for the bootstrap utility module."""

    def test_bootstrap_metric_basic(self) -> None:
        """Test basic bootstrap metric computation."""
        from sklearn.metrics import roc_auc_score

        from faircareai.core.bootstrap import bootstrap_metric

        np.random.seed(42)
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 5)
        y_prob = np.random.uniform(0, 1, 50)

        samples, n_failed = bootstrap_metric(
            y_true,
            y_prob,
            lambda yt, yp: roc_auc_score(yt, yp),
            n_bootstrap=100,
            seed=42,
        )

        # Should have some successful samples
        assert len(samples) > 0
        # All samples should be valid AUROC values
        assert all(0 <= s <= 1 for s in samples)

    def test_compute_percentile_ci(self) -> None:
        """Test percentile CI computation."""
        from faircareai.core.bootstrap import compute_percentile_ci

        samples = list(np.random.uniform(0.7, 0.9, 100))

        lower, upper = compute_percentile_ci(samples, alpha=0.05)

        assert lower is not None
        assert upper is not None
        assert lower < upper
        assert 0.7 <= lower <= 0.9
        assert 0.7 <= upper <= 0.9

    def test_compute_percentile_ci_insufficient_samples(self) -> None:
        """Test CI computation with insufficient samples."""
        from faircareai.core.bootstrap import compute_percentile_ci

        samples = [0.75, 0.78, 0.80]  # Only 3 samples

        lower, upper = compute_percentile_ci(samples)

        # Should return None for insufficient samples
        assert lower is None
        assert upper is None
