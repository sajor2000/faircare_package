"""
Tests for FairCareAI fairness metrics module.

Tests cover:
- compute_fairness_metrics function
- _compute_fairness_summary function
- compute_disparity_index function
- _interpret_disparity_index function
- compute_calibration_by_group function
- compute_threshold_fairness function
- compute_group_auroc_comparison function
- _interpret_auroc_diff function
"""

import numpy as np
import polars as pl
import pytest

from faircareai.metrics.fairness import (
    _compute_fairness_summary,
    _interpret_auroc_diff,
    _interpret_disparity_index,
    compute_calibration_by_group,
    compute_disparity_index,
    compute_fairness_metrics,
    compute_group_auroc_comparison,
    compute_threshold_fairness,
)


class TestComputeFairnessMetrics:
    """Tests for compute_fairness_metrics function."""

    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        """Create sample DataFrame for testing."""
        np.random.seed(42)
        n = 600
        groups = np.random.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
        y_true = np.random.binomial(1, 0.3, n)
        y_prob = np.where(
            y_true == 1,
            np.clip(np.random.normal(0.65, 0.15, n), 0.01, 0.99),
            np.clip(np.random.normal(0.35, 0.15, n), 0.01, 0.99),
        )
        return pl.DataFrame({
            "y_true": y_true,
            "y_prob": y_prob,
            "group": groups,
        })

    def test_returns_dict(self, sample_df: pl.DataFrame) -> None:
        """Test that function returns a dictionary."""
        result = compute_fairness_metrics(sample_df, "y_prob", "y_true", "group")
        assert isinstance(result, dict)

    def test_contains_group_col(self, sample_df: pl.DataFrame) -> None:
        """Test that result contains group column name."""
        result = compute_fairness_metrics(sample_df, "y_prob", "y_true", "group")
        assert result["group_col"] == "group"

    def test_contains_threshold(self, sample_df: pl.DataFrame) -> None:
        """Test that result contains threshold."""
        result = compute_fairness_metrics(
            sample_df, "y_prob", "y_true", "group", threshold=0.6
        )
        assert result["threshold"] == 0.6

    def test_contains_group_metrics(self, sample_df: pl.DataFrame) -> None:
        """Test that result contains group metrics."""
        result = compute_fairness_metrics(sample_df, "y_prob", "y_true", "group")
        assert "group_metrics" in result
        assert "A" in result["group_metrics"]
        assert "B" in result["group_metrics"]
        assert "C" in result["group_metrics"]

    def test_group_metrics_contain_required_fields(self, sample_df: pl.DataFrame) -> None:
        """Test that group metrics contain required fields."""
        result = compute_fairness_metrics(sample_df, "y_prob", "y_true", "group")
        required_fields = ["n", "prevalence", "selection_rate", "tpr", "fpr", "ppv", "npv"]
        for group in ["A", "B", "C"]:
            for field in required_fields:
                assert field in result["group_metrics"][group], f"Missing {field} for group {group}"

    def test_reference_group_determined(self, sample_df: pl.DataFrame) -> None:
        """Test that reference group is determined."""
        result = compute_fairness_metrics(sample_df, "y_prob", "y_true", "group")
        assert "reference" in result
        assert result["reference"] == "A"  # Largest group

    def test_specified_reference_used(self, sample_df: pl.DataFrame) -> None:
        """Test that specified reference is used."""
        result = compute_fairness_metrics(
            sample_df, "y_prob", "y_true", "group", reference="B"
        )
        assert result["reference"] == "B"

    def test_contains_demographic_parity_ratio(self, sample_df: pl.DataFrame) -> None:
        """Test that demographic parity ratio is computed."""
        result = compute_fairness_metrics(sample_df, "y_prob", "y_true", "group")
        assert "demographic_parity_ratio" in result
        # Should have ratios for non-reference groups
        assert "B" in result["demographic_parity_ratio"]
        assert "C" in result["demographic_parity_ratio"]

    def test_contains_tpr_diff(self, sample_df: pl.DataFrame) -> None:
        """Test that TPR difference is computed."""
        result = compute_fairness_metrics(sample_df, "y_prob", "y_true", "group")
        assert "tpr_diff" in result

    def test_contains_fpr_diff(self, sample_df: pl.DataFrame) -> None:
        """Test that FPR difference is computed."""
        result = compute_fairness_metrics(sample_df, "y_prob", "y_true", "group")
        assert "fpr_diff" in result

    def test_contains_equalized_odds_diff(self, sample_df: pl.DataFrame) -> None:
        """Test that equalized odds difference is computed."""
        result = compute_fairness_metrics(sample_df, "y_prob", "y_true", "group")
        assert "equalized_odds_diff" in result

    def test_contains_ppv_ratio(self, sample_df: pl.DataFrame) -> None:
        """Test that PPV ratio is computed."""
        result = compute_fairness_metrics(sample_df, "y_prob", "y_true", "group")
        assert "ppv_ratio" in result

    def test_contains_summary(self, sample_df: pl.DataFrame) -> None:
        """Test that summary is computed."""
        result = compute_fairness_metrics(sample_df, "y_prob", "y_true", "group")
        assert "summary" in result

    def test_is_reference_flag(self, sample_df: pl.DataFrame) -> None:
        """Test that is_reference flag is set correctly."""
        result = compute_fairness_metrics(
            sample_df, "y_prob", "y_true", "group", reference="B"
        )
        assert result["group_metrics"]["B"]["is_reference"] is True
        assert result["group_metrics"]["A"]["is_reference"] is False

    def test_small_group_handling(self) -> None:
        """Test handling of small groups."""
        df = pl.DataFrame({
            "y_true": [1, 0, 1] + [0, 1] * 100,
            "y_prob": [0.8, 0.2, 0.7] + [0.3, 0.7] * 100,
            "group": ["Small"] * 3 + ["Large"] * 200,
        })
        result = compute_fairness_metrics(df, "y_prob", "y_true", "group")
        assert "error" in result["group_metrics"]["Small"]

    def test_reference_error_propagated(self) -> None:
        """Test that error is returned when reference has insufficient data."""
        df = pl.DataFrame({
            "y_true": [1, 0, 1] + [0, 1] * 100,
            "y_prob": [0.8, 0.2, 0.7] + [0.3, 0.7] * 100,
            "group": ["Small"] * 3 + ["Large"] * 200,
        })
        result = compute_fairness_metrics(
            df, "y_prob", "y_true", "group", reference="Small"
        )
        assert "error" in result


class TestComputeFairnessSummary:
    """Tests for _compute_fairness_summary function."""

    def test_empty_metrics(self) -> None:
        """Test handling of empty metrics."""
        result = _compute_fairness_summary({})
        assert result == {}

    def test_demographic_parity_summary(self) -> None:
        """Test demographic parity summary."""
        metrics = {
            "demographic_parity_diff": {"B": 0.05, "C": -0.10},
        }
        result = _compute_fairness_summary(metrics)
        assert "demographic_parity" in result
        assert result["demographic_parity"]["worst_diff"] == -0.10

    def test_equal_opportunity_summary(self) -> None:
        """Test equal opportunity summary."""
        metrics = {
            "tpr_diff": {"B": 0.03, "C": 0.08},
        }
        result = _compute_fairness_summary(metrics)
        assert "equal_opportunity" in result
        assert result["equal_opportunity"]["worst_diff"] == 0.08

    def test_equalized_odds_summary(self) -> None:
        """Test equalized odds summary."""
        metrics = {
            "equalized_odds_diff": {"B": 0.05, "C": 0.12},
        }
        result = _compute_fairness_summary(metrics)
        assert "equalized_odds" in result
        assert result["equalized_odds"]["worst_diff"] == 0.12

    def test_predictive_parity_summary(self) -> None:
        """Test predictive parity summary."""
        metrics = {
            "ppv_ratio": {"B": 0.9, "C": 1.1},
        }
        result = _compute_fairness_summary(metrics)
        assert "predictive_parity" in result
        assert result["predictive_parity"]["worst_ratio"] == 0.9

    def test_filters_none_ppv_ratios(self) -> None:
        """Test that None values are filtered from PPV ratios."""
        metrics = {
            "ppv_ratio": {"B": 0.9, "C": None},
        }
        result = _compute_fairness_summary(metrics)
        assert "predictive_parity" in result
        assert result["predictive_parity"]["worst_ratio"] == 0.9


class TestComputeDisparityIndex:
    """Tests for compute_disparity_index function."""

    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        """Create sample DataFrame for testing."""
        np.random.seed(42)
        n = 600
        groups = np.random.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
        y_true = np.random.binomial(1, 0.3, n)
        y_prob = np.where(
            y_true == 1,
            np.clip(np.random.normal(0.65, 0.15, n), 0.01, 0.99),
            np.clip(np.random.normal(0.35, 0.15, n), 0.01, 0.99),
        )
        return pl.DataFrame({
            "y_true": y_true,
            "y_prob": y_prob,
            "group": groups,
        })

    def test_returns_dict(self, sample_df: pl.DataFrame) -> None:
        """Test that function returns a dictionary."""
        result = compute_disparity_index(sample_df, "y_prob", "y_true", "group")
        assert isinstance(result, dict)

    def test_contains_disparity_index(self, sample_df: pl.DataFrame) -> None:
        """Test that result contains disparity index."""
        result = compute_disparity_index(sample_df, "y_prob", "y_true", "group")
        assert "disparity_index" in result
        assert 0.0 <= result["disparity_index"] <= 1.0

    def test_contains_components(self, sample_df: pl.DataFrame) -> None:
        """Test that result contains component scores."""
        result = compute_disparity_index(sample_df, "y_prob", "y_true", "group")
        assert "components" in result

    def test_contains_interpretation(self, sample_df: pl.DataFrame) -> None:
        """Test that result contains interpretation."""
        result = compute_disparity_index(sample_df, "y_prob", "y_true", "group")
        assert "interpretation" in result
        assert "level" in result["interpretation"]

    def test_contains_raw_metrics(self, sample_df: pl.DataFrame) -> None:
        """Test that result contains raw metrics."""
        result = compute_disparity_index(sample_df, "y_prob", "y_true", "group")
        assert "raw_metrics" in result

    def test_error_returned_on_insufficient_data(self) -> None:
        """Test that error is returned when data is insufficient."""
        df = pl.DataFrame({
            "y_true": [1, 0, 1],
            "y_prob": [0.8, 0.2, 0.7],
            "group": ["Small"] * 3,
        })
        result = compute_disparity_index(df, "y_prob", "y_true", "group")
        assert "error" in result


class TestInterpretDisparityIndex:
    """Tests for _interpret_disparity_index function."""

    def test_low_disparity(self) -> None:
        """Test interpretation for low disparity."""
        result = _interpret_disparity_index(0.1)
        assert result["level"] == "LOW"
        assert result["color"] == "green"

    def test_moderate_disparity(self) -> None:
        """Test interpretation for moderate disparity."""
        result = _interpret_disparity_index(0.35)
        assert result["level"] == "MODERATE"
        assert result["color"] == "yellow"

    def test_high_disparity(self) -> None:
        """Test interpretation for high disparity."""
        result = _interpret_disparity_index(0.55)
        assert result["level"] == "HIGH"
        assert result["color"] == "orange"

    def test_severe_disparity(self) -> None:
        """Test interpretation for severe disparity."""
        result = _interpret_disparity_index(0.85)
        assert result["level"] == "SEVERE"
        assert result["color"] == "red"


class TestComputeCalibrationByGroup:
    """Tests for compute_calibration_by_group function."""

    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        """Create sample DataFrame for testing."""
        np.random.seed(42)
        n = 600
        groups = np.random.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
        y_true = np.random.binomial(1, 0.3, n)
        y_prob = np.where(
            y_true == 1,
            np.clip(np.random.normal(0.65, 0.15, n), 0.01, 0.99),
            np.clip(np.random.normal(0.35, 0.15, n), 0.01, 0.99),
        )
        return pl.DataFrame({
            "y_true": y_true,
            "y_prob": y_prob,
            "group": groups,
        })

    def test_returns_dict(self, sample_df: pl.DataFrame) -> None:
        """Test that function returns a dictionary."""
        result = compute_calibration_by_group(sample_df, "y_prob", "y_true", "group")
        assert isinstance(result, dict)

    def test_contains_groups(self, sample_df: pl.DataFrame) -> None:
        """Test that result contains groups."""
        result = compute_calibration_by_group(sample_df, "y_prob", "y_true", "group")
        assert "groups" in result

    def test_group_contains_calibration_data(self, sample_df: pl.DataFrame) -> None:
        """Test that group data contains calibration information."""
        result = compute_calibration_by_group(sample_df, "y_prob", "y_true", "group")
        for group in ["A", "B", "C"]:
            group_data = result["groups"][group]
            if "error" not in group_data:
                assert "prob_true" in group_data
                assert "prob_pred" in group_data
                assert "ece" in group_data
                assert "mce" in group_data

    def test_small_group_handling(self) -> None:
        """Test handling of small groups."""
        df = pl.DataFrame({
            "y_true": [1, 0, 1, 0, 1] + [0, 1] * 100,
            "y_prob": [0.8, 0.2, 0.7, 0.3, 0.6] + [0.3, 0.7] * 100,
            "group": ["Small"] * 5 + ["Large"] * 200,
        })
        result = compute_calibration_by_group(df, "y_prob", "y_true", "group")
        assert "error" in result["groups"]["Small"]


class TestComputeThresholdFairness:
    """Tests for compute_threshold_fairness function."""

    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        """Create sample DataFrame for testing."""
        np.random.seed(42)
        n = 600
        groups = np.random.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
        y_true = np.random.binomial(1, 0.3, n)
        y_prob = np.where(
            y_true == 1,
            np.clip(np.random.normal(0.65, 0.15, n), 0.01, 0.99),
            np.clip(np.random.normal(0.35, 0.15, n), 0.01, 0.99),
        )
        return pl.DataFrame({
            "y_true": y_true,
            "y_prob": y_prob,
            "group": groups,
        })

    def test_returns_dict(self, sample_df: pl.DataFrame) -> None:
        """Test that function returns a dictionary."""
        result = compute_threshold_fairness(sample_df, "y_prob", "y_true", "group")
        assert isinstance(result, dict)

    def test_contains_thresholds(self, sample_df: pl.DataFrame) -> None:
        """Test that result contains thresholds."""
        result = compute_threshold_fairness(sample_df, "y_prob", "y_true", "group")
        assert "thresholds" in result

    def test_default_thresholds(self, sample_df: pl.DataFrame) -> None:
        """Test default thresholds."""
        result = compute_threshold_fairness(sample_df, "y_prob", "y_true", "group")
        expected = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        assert result["thresholds"] == expected

    def test_custom_thresholds(self, sample_df: pl.DataFrame) -> None:
        """Test custom thresholds."""
        custom = [0.3, 0.5, 0.7]
        result = compute_threshold_fairness(
            sample_df, "y_prob", "y_true", "group", thresholds=custom
        )
        assert result["thresholds"] == custom

    def test_contains_metrics_by_threshold(self, sample_df: pl.DataFrame) -> None:
        """Test that result contains metrics by threshold."""
        result = compute_threshold_fairness(sample_df, "y_prob", "y_true", "group")
        assert "metrics_by_threshold" in result
        assert len(result["metrics_by_threshold"]) == len(result["thresholds"])

    def test_contains_recommended_threshold(self, sample_df: pl.DataFrame) -> None:
        """Test that result contains recommended threshold."""
        result = compute_threshold_fairness(sample_df, "y_prob", "y_true", "group")
        assert "recommended_threshold" in result
        assert "threshold" in result["recommended_threshold"]
        assert "note" in result["recommended_threshold"]


class TestComputeGroupAurocComparison:
    """Tests for compute_group_auroc_comparison function."""

    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        """Create sample DataFrame for testing."""
        np.random.seed(42)
        n = 600
        groups = np.random.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
        y_true = np.random.binomial(1, 0.3, n)
        y_prob = np.where(
            y_true == 1,
            np.clip(np.random.normal(0.65, 0.15, n), 0.01, 0.99),
            np.clip(np.random.normal(0.35, 0.15, n), 0.01, 0.99),
        )
        return pl.DataFrame({
            "y_true": y_true,
            "y_prob": y_prob,
            "group": groups,
        })

    def test_returns_dict(self, sample_df: pl.DataFrame) -> None:
        """Test that function returns a dictionary."""
        result = compute_group_auroc_comparison(
            sample_df, "y_prob", "y_true", "group", n_bootstrap=50
        )
        assert isinstance(result, dict)

    def test_contains_groups(self, sample_df: pl.DataFrame) -> None:
        """Test that result contains groups."""
        result = compute_group_auroc_comparison(
            sample_df, "y_prob", "y_true", "group", n_bootstrap=50
        )
        assert "groups" in result

    def test_group_contains_auroc(self, sample_df: pl.DataFrame) -> None:
        """Test that group data contains AUROC."""
        result = compute_group_auroc_comparison(
            sample_df, "y_prob", "y_true", "group", n_bootstrap=50
        )
        for group in ["A", "B", "C"]:
            if "error" not in result["groups"][group]:
                assert "auroc" in result["groups"][group]
                assert 0.0 <= result["groups"][group]["auroc"] <= 1.0

    def test_contains_comparisons(self, sample_df: pl.DataFrame) -> None:
        """Test that result contains comparisons."""
        result = compute_group_auroc_comparison(
            sample_df, "y_prob", "y_true", "group", n_bootstrap=50
        )
        assert "comparisons" in result

    def test_comparisons_have_auroc_diff(self, sample_df: pl.DataFrame) -> None:
        """Test that comparisons have AUROC difference."""
        result = compute_group_auroc_comparison(
            sample_df, "y_prob", "y_true", "group", n_bootstrap=50
        )
        for group, comp in result["comparisons"].items():
            assert "auroc_diff" in comp
            assert "interpretation" in comp

    def test_bootstrap_ci_computed(self, sample_df: pl.DataFrame) -> None:
        """Test that bootstrap CI is computed."""
        result = compute_group_auroc_comparison(
            sample_df, "y_prob", "y_true", "group", n_bootstrap=100
        )
        for group in ["A", "B", "C"]:
            if "error" not in result["groups"][group]:
                assert "auroc_ci_95" in result["groups"][group]


class TestInterpretAurocDiff:
    """Tests for _interpret_auroc_diff function."""

    def test_negligible(self) -> None:
        """Test interpretation for negligible difference."""
        result = _interpret_auroc_diff(0.01)
        assert result == "negligible"

    def test_small(self) -> None:
        """Test interpretation for small difference."""
        result = _interpret_auroc_diff(0.04)
        assert result == "small"

    def test_moderate(self) -> None:
        """Test interpretation for moderate difference."""
        result = _interpret_auroc_diff(0.08)
        assert result == "moderate"

    def test_large(self) -> None:
        """Test interpretation for large difference."""
        result = _interpret_auroc_diff(0.15)
        assert result == "large"

    def test_negative_difference(self) -> None:
        """Test interpretation for negative difference."""
        result = _interpret_auroc_diff(-0.08)
        assert result == "moderate"
