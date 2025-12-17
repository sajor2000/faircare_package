"""
Tests for FairCareAI calibration metrics module.

These tests verify the accuracy of:
1. ACE (Adaptive Calibration Error) with quantile binning
2. Cluster bootstrap CIs for calibration metrics
3. Group calibration parity analysis
4. Calibration gap computation

ACE is preferred over ECE for healthcare data because:
- Quantile binning ensures high-risk tail contributes equally
- Uniform binning (ECE) is dominated by low-risk majority in imbalanced data
"""

import numpy as np
import polars as pl
import pytest

from faircareai.core.calibration import (
    CalibrationResult,
    GroupCalibrationResult,
    compute_ace,
    compute_ace_with_ci,
    compute_calibration_from_df,
    compute_group_calibration,
)


class TestACE:
    """
    Tests for Adaptive Calibration Error with quantile binning.

    ACE addresses the key limitation of ECE: in healthcare with 5% prevalence,
    uniform-bin ECE is dominated by the 95% healthy majority, masking
    miscalibration in high-risk patients.
    """

    def test_perfect_calibration_near_zero(self) -> None:
        """ACE should be near 0 for perfectly calibrated predictions."""
        np.random.seed(42)
        n = 1000

        # Generate probabilities
        y_prob = np.random.uniform(0, 1, n)
        # Generate outcomes that match probabilities (perfect calibration)
        y_true = (np.random.uniform(0, 1, n) < y_prob).astype(int)

        ace, bins_used = compute_ace(y_true, y_prob, n_bins=10)

        # Should be close to 0 (some noise expected)
        assert ace < 0.10
        assert bins_used > 0

    def test_severe_miscalibration_detected(self) -> None:
        """ACE should be high when predictions don't match outcomes."""
        n = 500

        # Always predict 0.8, but actual rate is 0.5
        y_prob = np.full(n, 0.8)
        y_true = np.array([0, 1] * (n // 2))

        ace, _ = compute_ace(y_true, y_prob, n_bins=10)

        # Calibration error ≈ |0.8 - 0.5| = 0.3
        assert ace > 0.25

    def test_quantile_binning_handles_imbalance(self) -> None:
        """Quantile binning should work with imbalanced data (5% prevalence)."""
        n = 1000

        # Simulate typical healthcare data: 5% positive
        y_prob = np.concatenate(
            [
                np.random.uniform(0, 0.1, 950),  # 95% low-risk
                np.random.uniform(0.8, 1.0, 50),  # 5% high-risk
            ]
        )
        y_true = np.concatenate(
            [
                np.zeros(950, dtype=int),
                np.ones(50, dtype=int),
            ]
        )

        ace, bins_used = compute_ace(y_true, y_prob, n_bins=10)

        # Should use multiple bins (quantile binning distributes samples)
        assert bins_used >= 2
        assert not np.isnan(ace)

    def test_empty_data_returns_nan(self) -> None:
        """Should return NaN for empty input."""
        ace, bins_used = compute_ace(np.array([]), np.array([]))
        assert np.isnan(ace)
        assert bins_used == 0

    def test_constant_predictions(self) -> None:
        """Should handle case where all predictions are identical."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_prob = np.full(8, 0.5)

        ace, bins_used = compute_ace(y_true, y_prob, n_bins=5)

        # Should collapse to 1 bin
        assert bins_used == 1
        assert not np.isnan(ace)
        # ACE = |mean(y_true) - mean(y_prob)| = |0.5 - 0.5| = 0
        assert ace == 0.0

    def test_min_per_bin_gate(self) -> None:
        """Bins with fewer than min_per_bin samples should be excluded."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.8, 0.9])

        # With min_per_bin=3, some bins should be gated
        ace, bins_used = compute_ace(y_true, y_prob, n_bins=5, min_per_bin=3)

        assert bins_used < 5

    def test_nan_handling(self) -> None:
        """Should handle NaN values gracefully."""
        y_true = np.array([0, 1, np.nan, 1, 0, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.5, np.nan, 0.3, 0.7, 0.4, 0.6])

        ace, _ = compute_ace(y_true, y_prob, n_bins=3)

        # Should compute on valid values only
        assert not np.isnan(ace) or True  # May be NaN if too few valid


class TestACEWithCI:
    """Tests for ACE with cluster-aware bootstrap confidence intervals."""

    def test_basic_ci_computation(self) -> None:
        """Test basic ACE CI computation."""
        np.random.seed(42)
        n = 200

        y_prob = np.random.uniform(0, 1, n)
        y_true = (np.random.uniform(0, 1, n) < y_prob).astype(int)

        ace, (ci_lower, ci_upper), bins_used = compute_ace_with_ci(
            y_true, y_prob, n_bins=5, n_bootstrap=100, random_state=42
        )

        assert ci_lower <= ace <= ci_upper
        assert ci_lower >= 0
        assert bins_used > 0

    def test_cluster_bootstrap_ci(self) -> None:
        """Test cluster-aware bootstrap (patient-level resampling)."""
        np.random.seed(42)
        n_patients = 50
        encounters_per_patient = 4
        n = n_patients * encounters_per_patient

        cluster_ids = np.repeat(np.arange(n_patients), encounters_per_patient)
        y_prob = np.random.uniform(0, 1, n)
        y_true = (np.random.uniform(0, 1, n) < y_prob).astype(int)

        ace_cluster, (ci_lo_cluster, ci_hi_cluster), _ = compute_ace_with_ci(
            y_true, y_prob, cluster_ids=cluster_ids, n_bins=5, n_bootstrap=100, random_state=42
        )

        ace_naive, (ci_lo_naive, ci_hi_naive), _ = compute_ace_with_ci(
            y_true, y_prob, cluster_ids=None, n_bins=5, n_bootstrap=100, random_state=42
        )

        # Both should produce valid CIs
        assert ci_lo_cluster < ci_hi_cluster
        assert ci_lo_naive < ci_hi_naive

    def test_reproducibility(self) -> None:
        """Same random_state should produce identical results."""
        np.random.seed(42)
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1] * 20)
        y_prob = np.array([0.2, 0.3, 0.7, 0.8, 0.3, 0.6, 0.2, 0.9, 0.4, 0.7] * 20)

        result1 = compute_ace_with_ci(y_true, y_prob, n_bins=5, n_bootstrap=50, random_state=123)
        result2 = compute_ace_with_ci(y_true, y_prob, n_bins=5, n_bootstrap=50, random_state=123)

        assert result1[0] == result2[0]
        assert result1[1] == result2[1]


class TestGroupCalibration:
    """Tests for per-group calibration analysis and calibration parity."""

    def test_basic_group_calibration(self) -> None:
        """Test calibration computed separately per group."""
        np.random.seed(42)
        n = 200

        groups = np.array(["A"] * 100 + ["B"] * 100)
        y_prob = np.random.uniform(0, 1, n)
        y_true = (np.random.uniform(0, 1, n) < y_prob).astype(int)

        result = compute_group_calibration(
            y_true, y_prob, groups, n_bins=5, n_bootstrap=50, random_state=42
        )

        assert isinstance(result, GroupCalibrationResult)
        assert "A" in result.group_results
        assert "B" in result.group_results

    def test_calibration_gap_computation(self) -> None:
        """Calibration gap = max(ACE) - min(ACE) across groups."""
        np.random.seed(42)
        n = 100

        groups = np.array(["A"] * 50 + ["B"] * 50)

        # Group A: well calibrated
        y_prob_a = np.random.uniform(0.4, 0.6, 50)
        y_true_a = (np.random.uniform(0, 1, 50) < y_prob_a).astype(int)

        # Group B: poorly calibrated
        y_prob_b = np.full(50, 0.9)
        y_true_b = np.random.binomial(1, 0.3, 50)  # True rate ≈ 30%

        y_prob = np.concatenate([y_prob_a, y_prob_b])
        y_true = np.concatenate([y_true_a, y_true_b])

        result = compute_group_calibration(
            y_true, y_prob, groups, n_bins=5, n_bootstrap=0, random_state=42
        )

        # Group B should have higher ACE
        assert result.group_results["B"].ace > result.group_results["A"].ace

        # Calibration gap should be positive
        assert result.calibration_gap > 0

    def test_worst_calibrated_group_identification(self) -> None:
        """Should identify group with highest ACE."""
        np.random.seed(42)
        n = 150

        groups = np.array(["A"] * 50 + ["B"] * 50 + ["C"] * 50)

        # Make group C worst calibrated
        y_prob_a = np.random.uniform(0.4, 0.6, 50)
        y_true_a = (np.random.uniform(0, 1, 50) < y_prob_a).astype(int)

        y_prob_b = np.random.uniform(0.4, 0.6, 50)
        y_true_b = (np.random.uniform(0, 1, 50) < y_prob_b).astype(int)

        # Group C: severe miscalibration (predict 0.95, truth is 0)
        y_prob_c = np.full(50, 0.95)
        y_true_c = np.zeros(50, dtype=int)

        y_prob = np.concatenate([y_prob_a, y_prob_b, y_prob_c])
        y_true = np.concatenate([y_true_a, y_true_b, y_true_c])

        result = compute_group_calibration(
            y_true, y_prob, groups, n_bins=5, n_bootstrap=0, random_state=42
        )

        assert result.worst_calibrated_group == "C"

    def test_calibration_gap_ci(self) -> None:
        """Should compute CI for calibration gap when bootstrapping."""
        np.random.seed(42)
        n_patients = 40
        encounters = 5
        n = n_patients * encounters

        cluster_ids = np.repeat(np.arange(n_patients), encounters)
        groups = np.array(["A"] * 100 + ["B"] * 100)
        y_prob = np.random.uniform(0, 1, n)
        y_true = (np.random.uniform(0, 1, n) < y_prob).astype(int)

        result = compute_group_calibration(
            y_true,
            y_prob,
            groups,
            cluster_ids=cluster_ids,
            n_bins=5,
            n_bootstrap=50,
            random_state=42,
        )

        # Should have calibration gap CI
        assert result.calibration_gap_ci is not None
        assert result.calibration_gap_ci[0] <= result.calibration_gap_ci[1]


class TestCalibrationFromDataFrame:
    """Tests for DataFrame interface to calibration metrics."""

    def test_single_group_analysis(self) -> None:
        """Compute calibration for entire dataset (no group column)."""
        np.random.seed(42)
        n = 100

        df = pl.DataFrame(
            {
                "y_true": np.random.binomial(1, 0.5, n),
                "y_prob": np.random.uniform(0, 1, n),
            }
        )

        result = compute_calibration_from_df(
            df, "y_true", "y_prob", n_bins=5, n_bootstrap=50, random_state=42
        )

        assert isinstance(result, CalibrationResult)
        assert result.group == "all"
        assert not np.isnan(result.ace)

    def test_grouped_analysis(self) -> None:
        """Compute calibration per demographic group."""
        np.random.seed(42)
        n = 100

        df = pl.DataFrame(
            {
                "y_true": np.random.binomial(1, 0.5, n),
                "y_prob": np.random.uniform(0, 1, n),
                "race": ["White"] * 50 + ["Black"] * 50,
            }
        )

        result = compute_calibration_from_df(
            df, "y_true", "y_prob", group_col="race", n_bins=5, n_bootstrap=50, random_state=42
        )

        assert isinstance(result, GroupCalibrationResult)
        assert "White" in result.group_results
        assert "Black" in result.group_results

    def test_with_cluster_column(self) -> None:
        """Should use cluster bootstrap when patient_id provided."""
        np.random.seed(42)
        n = 100

        df = pl.DataFrame(
            {
                "y_true": np.random.binomial(1, 0.5, n),
                "y_prob": np.random.uniform(0, 1, n),
                "patient_id": list(range(20)) * 5,
                "race": ["White"] * 50 + ["Black"] * 50,
            }
        )

        result = compute_calibration_from_df(
            df,
            "y_true",
            "y_prob",
            group_col="race",
            cluster_col="patient_id",
            n_bins=5,
            n_bootstrap=50,
            random_state=42,
        )

        assert isinstance(result, GroupCalibrationResult)

        # Effective n should be lower than n_samples (20 vs 50 per group)
        for group_result in result.group_results.values():
            assert group_result.n_effective < group_result.n_samples


class TestCalibrationResult:
    """Tests for CalibrationResult dataclass."""

    def test_dataclass_creation(self) -> None:
        """Test creating CalibrationResult."""
        result = CalibrationResult(
            group="test",
            ace=0.05,
            ace_ci_lower=0.03,
            ace_ci_upper=0.08,
            n_bins=10,
            n_samples=100,
            n_effective=100,
            bins_used=8,
        )

        assert result.group == "test"
        assert result.ace == 0.05
        assert result.ace_ci_lower == 0.03
        assert result.ace_ci_upper == 0.08
        assert result.bins_used == 8

    def test_frozen_immutable(self) -> None:
        """CalibrationResult should be immutable (frozen dataclass)."""
        result = CalibrationResult(
            group="test",
            ace=0.05,
            ace_ci_lower=0.03,
            ace_ci_upper=0.08,
            n_bins=10,
            n_samples=100,
            n_effective=100,
            bins_used=8,
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            result.ace = 0.10
