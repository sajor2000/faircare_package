"""
Tests for FairCareAI bootstrap confidence interval utilities.

Tests cover:
- bootstrap_metric function
- bootstrap_confusion_metrics function
- compute_percentile_ci function
- compute_ci_from_samples function
- bootstrap_auroc convenience function
"""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from faircareai.core.bootstrap import (
    bootstrap_auroc,
    bootstrap_confusion_metrics,
    bootstrap_metric,
    compute_ci_from_samples,
    compute_percentile_ci,
)


class TestBootstrapMetric:
    """Tests for bootstrap_metric function."""

    @pytest.fixture
    def sample_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Create sample data for testing."""
        np.random.seed(42)
        n = 200
        y_true = np.random.binomial(1, 0.3, n)
        y_prob = np.random.beta(2, 5, n)
        # Make y_prob somewhat correlated with y_true
        y_prob = np.where(y_true == 1, y_prob + 0.3, y_prob)
        y_prob = np.clip(y_prob, 0.01, 0.99)
        return y_true, y_prob

    def test_returns_samples_and_failures(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that function returns samples list and failure count."""
        y_true, y_prob = sample_data
        samples, n_failed = bootstrap_metric(
            y_true, y_prob, lambda yt, yp: roc_auc_score(yt, yp), n_bootstrap=100
        )
        assert isinstance(samples, list)
        assert isinstance(n_failed, int)
        assert len(samples) + n_failed <= 100

    def test_empty_arrays(self) -> None:
        """Test handling of empty arrays."""
        y_true = np.array([])
        y_prob = np.array([])
        samples, n_failed = bootstrap_metric(y_true, y_prob, lambda yt, yp: 0.0, n_bootstrap=10)
        assert samples == []
        assert n_failed == 0

    def test_single_class_handling(self) -> None:
        """Test handling when only one class in data."""
        y_true = np.ones(100)  # All positive
        y_prob = np.random.uniform(0, 1, 100)
        samples, n_failed = bootstrap_metric(
            y_true, y_prob, lambda yt, yp: roc_auc_score(yt, yp), n_bootstrap=50
        )
        # Should fail all due to single class
        assert n_failed == 50

    def test_respects_min_classes(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that min_classes parameter is respected."""
        y_true, y_prob = sample_data
        # With min_classes=3, all samples should fail for binary data
        samples, n_failed = bootstrap_metric(
            y_true, y_prob, lambda yt, yp: 0.5, n_bootstrap=20, min_classes=3
        )
        assert n_failed == 20

    def test_reproducibility_with_seed(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that same seed produces same results."""
        y_true, y_prob = sample_data
        samples1, _ = bootstrap_metric(
            y_true, y_prob, lambda yt, yp: yt.mean(), n_bootstrap=50, seed=123
        )
        samples2, _ = bootstrap_metric(
            y_true, y_prob, lambda yt, yp: yt.mean(), n_bootstrap=50, seed=123
        )
        assert samples1 == samples2

    def test_different_seeds_differ(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that different seeds produce different results."""
        y_true, y_prob = sample_data
        samples1, _ = bootstrap_metric(
            y_true, y_prob, lambda yt, yp: yt.mean(), n_bootstrap=50, seed=1
        )
        samples2, _ = bootstrap_metric(
            y_true, y_prob, lambda yt, yp: yt.mean(), n_bootstrap=50, seed=2
        )
        assert samples1 != samples2

    def test_metric_function_receives_correct_arrays(
        self, sample_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test that metric function receives proper array inputs."""
        y_true, y_prob = sample_data
        called_with_arrays = []

        def capture_metric(yt: np.ndarray, yp: np.ndarray) -> float:
            called_with_arrays.append((len(yt), len(yp)))
            return yt.mean()

        bootstrap_metric(y_true, y_prob, capture_metric, n_bootstrap=5)
        assert len(called_with_arrays) > 0
        for len_yt, len_yp in called_with_arrays:
            assert len_yt == len(y_true)  # Bootstrap samples same size
            assert len_yp == len(y_prob)


class TestBootstrapConfusionMetrics:
    """Tests for bootstrap_confusion_metrics function."""

    @pytest.fixture
    def binary_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Create binary classification data."""
        np.random.seed(42)
        n = 200
        y_true = np.random.binomial(1, 0.3, n)
        y_prob = np.where(
            y_true == 1, np.random.uniform(0.5, 1.0, n), np.random.uniform(0.0, 0.5, n)
        )
        return y_true, y_prob

    def test_all_metrics_returned(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that all confusion metrics are returned."""
        y_true, y_prob = binary_data
        results = bootstrap_confusion_metrics(y_true, y_prob, threshold=0.5, n_bootstrap=50)
        assert "sensitivity" in results
        assert "specificity" in results
        assert "ppv" in results
        assert "npv" in results

    def test_list_lengths(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that all metric lists have same length."""
        y_true, y_prob = binary_data
        results = bootstrap_confusion_metrics(y_true, y_prob, threshold=0.5, n_bootstrap=50)
        lengths = [len(results[k]) for k in results]
        assert all(l == lengths[0] for l in lengths)

    def test_values_in_range(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that metric values are in [0, 1]."""
        y_true, y_prob = binary_data
        results = bootstrap_confusion_metrics(y_true, y_prob, threshold=0.5, n_bootstrap=50)
        for metric_name, values in results.items():
            for v in values:
                assert 0.0 <= v <= 1.0, f"{metric_name} value {v} out of range"

    def test_threshold_affects_results(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that different thresholds produce different results."""
        y_true, y_prob = binary_data
        results_low = bootstrap_confusion_metrics(
            y_true, y_prob, threshold=0.3, n_bootstrap=50, seed=42
        )
        results_high = bootstrap_confusion_metrics(
            y_true, y_prob, threshold=0.7, n_bootstrap=50, seed=42
        )
        # Different thresholds should give different sensitivities
        assert np.mean(results_low["sensitivity"]) != np.mean(results_high["sensitivity"])


class TestComputePercentileCI:
    """Tests for compute_percentile_ci function."""

    def test_normal_samples(self) -> None:
        """Test CI computation with normal samples."""
        samples = list(np.random.normal(0.75, 0.05, 1000))
        ci_lower, ci_upper = compute_percentile_ci(samples)
        assert ci_lower is not None
        assert ci_upper is not None
        assert ci_lower < 0.75 < ci_upper
        assert ci_lower < ci_upper

    def test_empty_samples(self) -> None:
        """Test CI computation with empty samples."""
        ci_lower, ci_upper = compute_percentile_ci([])
        assert ci_lower is None
        assert ci_upper is None

    def test_too_few_samples(self) -> None:
        """Test CI computation with too few samples."""
        # MIN_BOOTSTRAP_SAMPLES is typically 20
        samples = [0.5, 0.6, 0.7]  # Only 3 samples
        ci_lower, ci_upper = compute_percentile_ci(samples)
        assert ci_lower is None
        assert ci_upper is None

    def test_alpha_affects_width(self) -> None:
        """Test that different alpha values affect CI width."""
        samples = list(np.random.normal(0.5, 0.1, 1000))
        ci_95_lower, ci_95_upper = compute_percentile_ci(samples, alpha=0.05)
        ci_99_lower, ci_99_upper = compute_percentile_ci(samples, alpha=0.01)
        # 99% CI should be wider than 95% CI
        if ci_95_lower is not None and ci_99_lower is not None:
            assert (ci_99_upper - ci_99_lower) > (ci_95_upper - ci_95_lower)

    def test_returns_floats(self) -> None:
        """Test that CI bounds are floats."""
        samples = list(np.random.uniform(0, 1, 100))
        ci_lower, ci_upper = compute_percentile_ci(samples)
        if ci_lower is not None:
            assert isinstance(ci_lower, float)
            assert isinstance(ci_upper, float)


class TestComputeCIFromSamples:
    """Tests for compute_ci_from_samples function."""

    def test_returns_dict(self) -> None:
        """Test that function returns a dictionary."""
        samples = list(np.random.uniform(0.6, 0.8, 100))
        result = compute_ci_from_samples(samples)
        assert isinstance(result, dict)

    def test_ci_fmt_format(self) -> None:
        """Test that ci_fmt has correct format."""
        samples = list(np.random.uniform(0.6, 0.8, 100))
        result = compute_ci_from_samples(samples)
        if result["ci_fmt"] is not None:
            assert "95% CI:" in result["ci_fmt"]
            assert "-" in result["ci_fmt"]

    def test_insufficient_samples(self) -> None:
        """Test handling of insufficient samples."""
        samples = [0.5, 0.6]  # Too few
        result = compute_ci_from_samples(samples)
        assert result["lower"] is None
        assert result["upper"] is None
        assert result["ci_fmt"] is None

    def test_different_alpha(self) -> None:
        """Test with different alpha value."""
        samples = list(np.random.uniform(0.6, 0.8, 100))
        result = compute_ci_from_samples(samples, alpha=0.10)
        if result["ci_fmt"] is not None:
            assert "90% CI:" in result["ci_fmt"]

    def test_dict_keys(self) -> None:
        """Test that dictionary has expected keys."""
        samples = list(np.random.uniform(0.6, 0.8, 100))
        result = compute_ci_from_samples(samples)
        assert "lower" in result
        assert "upper" in result
        assert "ci_fmt" in result


class TestBootstrapAUROC:
    """Tests for bootstrap_auroc convenience function."""

    @pytest.fixture
    def auroc_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Create data for AUROC computation with realistic separation."""
        np.random.seed(42)
        n = 300
        y_true = np.random.binomial(1, 0.3, n)
        # Add overlap for realistic AUROC (~0.75-0.85)
        y_prob = np.where(
            y_true == 1,
            np.clip(np.random.normal(0.6, 0.2, n), 0.01, 0.99),
            np.clip(np.random.normal(0.4, 0.2, n), 0.01, 0.99),
        )
        return y_true, y_prob

    def test_returns_tuple(self, auroc_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that function returns tuple of samples and CI bounds."""
        y_true, y_prob = auroc_data
        result = bootstrap_auroc(y_true, y_prob, n_bootstrap=50)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_ci_bounds_reasonable(self, auroc_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that CI bounds are reasonable for AUROC."""
        y_true, y_prob = auroc_data
        samples, ci_lower, ci_upper = bootstrap_auroc(y_true, y_prob, n_bootstrap=100)
        if ci_lower is not None:
            # AUROC should be between 0 and 1
            assert 0.0 <= ci_lower <= 1.0
            assert 0.0 <= ci_upper <= 1.0
            # Lower should be less than upper
            assert ci_lower < ci_upper
            # CI should contain most samples
            assert ci_lower < np.percentile(samples, 50) < ci_upper

    def test_samples_in_valid_range(self, auroc_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that all AUROC samples are in [0, 1]."""
        y_true, y_prob = auroc_data
        samples, _, _ = bootstrap_auroc(y_true, y_prob, n_bootstrap=50)
        for s in samples:
            assert 0.0 <= s <= 1.0

    def test_reproducibility_with_seed(self, auroc_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that same seed produces same results."""
        y_true, y_prob = auroc_data
        result1 = bootstrap_auroc(y_true, y_prob, n_bootstrap=50, seed=123)
        result2 = bootstrap_auroc(y_true, y_prob, n_bootstrap=50, seed=123)
        assert result1[0] == result2[0]  # Same samples
        assert result1[1] == result2[1]  # Same CI lower
        assert result1[2] == result2[2]  # Same CI upper
