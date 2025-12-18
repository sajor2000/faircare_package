"""
Tests for FairCareAI disparity analysis module.

Tests cover:
- DisparityResult dataclass
- Single disparity computation
- Multiple group disparity computation
- Worst disparity identification
- Status counting
"""

import polars as pl
import pytest

from faircareai.core.disparity import (
    DisparityResult,
    compute_disparities,
    compute_disparity,
    count_by_status,
    get_worst_disparity,
)


class TestDisparityResult:
    """Tests for DisparityResult dataclass."""

    def test_dataclass_construction(self) -> None:
        """Test that DisparityResult can be constructed."""
        result = DisparityResult(
            reference_group="White",
            comparison_group="Black",
            metric="tpr",
            reference_value=0.85,
            comparison_value=0.75,
            difference=-0.10,
            diff_ci_lower=-0.18,
            diff_ci_upper=-0.02,
            ratio=0.88,
            status="warn",
            p_value=0.03,
            statistically_significant=True,
        )
        assert result.reference_group == "White"
        assert result.comparison_group == "Black"
        assert result.metric == "tpr"
        assert result.difference == -0.10
        assert result.status == "warn"

    def test_dataclass_attributes(self) -> None:
        """Test all required attributes exist."""
        result = DisparityResult(
            reference_group="A",
            comparison_group="B",
            metric="fpr",
            reference_value=0.1,
            comparison_value=0.2,
            difference=0.1,
            diff_ci_lower=0.05,
            diff_ci_upper=0.15,
            ratio=2.0,
            status="fail",
            p_value=0.001,
            statistically_significant=True,
        )
        assert hasattr(result, "reference_group")
        assert hasattr(result, "comparison_group")
        assert hasattr(result, "metric")
        assert hasattr(result, "reference_value")
        assert hasattr(result, "comparison_value")
        assert hasattr(result, "difference")
        assert hasattr(result, "diff_ci_lower")
        assert hasattr(result, "diff_ci_upper")
        assert hasattr(result, "ratio")
        assert hasattr(result, "status")
        assert hasattr(result, "p_value")
        assert hasattr(result, "statistically_significant")


class TestComputeDisparity:
    """Tests for compute_disparity function."""

    def test_basic_computation(self) -> None:
        """Test basic disparity computation."""
        result = compute_disparity(
            reference_group="White",
            reference_value=0.80,
            reference_successes=80,
            reference_trials=100,
            comparison_group="Black",
            comparison_value=0.70,
            comparison_successes=70,
            comparison_trials=100,
            metric="tpr",
        )
        assert isinstance(result, DisparityResult)
        assert result.reference_group == "White"
        assert result.comparison_group == "Black"
        assert result.difference == pytest.approx(-0.10, abs=0.01)

    def test_reference_value_zero(self) -> None:
        """Test ratio handling when reference value is zero."""
        result = compute_disparity(
            reference_group="A",
            reference_value=0.0,
            reference_successes=0,
            reference_trials=100,
            comparison_group="B",
            comparison_value=0.5,
            comparison_successes=50,
            comparison_trials=100,
            metric="tpr",
        )
        assert result.ratio == float("inf")

    def test_both_zero_values(self) -> None:
        """Test ratio when both values are zero."""
        result = compute_disparity(
            reference_group="A",
            reference_value=0.0,
            reference_successes=0,
            reference_trials=100,
            comparison_group="B",
            comparison_value=0.0,
            comparison_successes=0,
            comparison_trials=100,
            metric="tpr",
        )
        assert result.ratio == 1.0
        assert result.difference == 0.0

    def test_fail_status(self) -> None:
        """Test that large difference results in fail status."""
        result = compute_disparity(
            reference_group="A",
            reference_value=0.90,
            reference_successes=90,
            reference_trials=100,
            comparison_group="B",
            comparison_value=0.70,
            comparison_successes=70,
            comparison_trials=100,
            metric="tpr",
            fail_threshold=0.10,
        )
        assert result.status == "fail"

    def test_warn_status(self) -> None:
        """Test that moderate difference results in warn status."""
        result = compute_disparity(
            reference_group="A",
            reference_value=0.80,
            reference_successes=80,
            reference_trials=100,
            comparison_group="B",
            comparison_value=0.73,
            comparison_successes=73,
            comparison_trials=100,
            metric="tpr",
            warn_threshold=0.05,
            fail_threshold=0.10,
        )
        assert result.status == "warn"

    def test_pass_status(self) -> None:
        """Test that small difference results in pass status."""
        result = compute_disparity(
            reference_group="A",
            reference_value=0.80,
            reference_successes=80,
            reference_trials=100,
            comparison_group="B",
            comparison_value=0.78,
            comparison_successes=78,
            comparison_trials=100,
            metric="tpr",
            warn_threshold=0.05,
        )
        assert result.status == "pass"

    def test_confidence_interval_computed(self) -> None:
        """Test that CI bounds are computed."""
        result = compute_disparity(
            reference_group="A",
            reference_value=0.80,
            reference_successes=80,
            reference_trials=100,
            comparison_group="B",
            comparison_value=0.70,
            comparison_successes=70,
            comparison_trials=100,
            metric="tpr",
        )
        assert result.diff_ci_lower is not None
        assert result.diff_ci_upper is not None
        assert result.diff_ci_lower < result.diff_ci_upper

    def test_p_value_computed(self) -> None:
        """Test that p-value is computed."""
        result = compute_disparity(
            reference_group="A",
            reference_value=0.80,
            reference_successes=80,
            reference_trials=100,
            comparison_group="B",
            comparison_value=0.70,
            comparison_successes=70,
            comparison_trials=100,
            metric="tpr",
        )
        assert result.p_value is not None
        assert 0.0 <= result.p_value <= 1.0

    def test_statistical_significance(self) -> None:
        """Test statistical significance detection."""
        # Large difference should be significant
        result = compute_disparity(
            reference_group="A",
            reference_value=0.90,
            reference_successes=900,
            reference_trials=1000,
            comparison_group="B",
            comparison_value=0.70,
            comparison_successes=700,
            comparison_trials=1000,
            metric="tpr",
            alpha=0.05,
        )
        assert result.statistically_significant == True  # noqa: E712
        assert result.p_value < 0.05


class TestComputeDisparities:
    """Tests for compute_disparities function."""

    @pytest.fixture
    def metrics_df(self) -> pl.DataFrame:
        """Create sample metrics DataFrame."""
        return pl.DataFrame(
            {
                "group": ["White", "Black", "Hispanic", "_overall"],
                "n": [500, 300, 200, 1000],
                "n_positive": [100, 60, 40, 200],
                "tpr": [0.85, 0.75, 0.80, 0.80],
            }
        )

    def test_returns_dataframe(self, metrics_df: pl.DataFrame) -> None:
        """Test that function returns a DataFrame."""
        result = compute_disparities(metrics_df)
        assert isinstance(result, pl.DataFrame)

    def test_reference_strategy_largest(self, metrics_df: pl.DataFrame) -> None:
        """Test largest group is selected as reference."""
        result = compute_disparities(metrics_df, reference_strategy="largest")
        # White has largest n (500)
        if len(result) > 0:
            assert result["reference_group"][0] == "White"

    def test_reference_strategy_best(self, metrics_df: pl.DataFrame) -> None:
        """Test best performing group is selected as reference."""
        result = compute_disparities(metrics_df, reference_strategy="best")
        # White has best TPR (0.85)
        if len(result) > 0:
            assert result["reference_group"][0] == "White"

    def test_reference_strategy_specified(self, metrics_df: pl.DataFrame) -> None:
        """Test specified reference group is used."""
        result = compute_disparities(
            metrics_df, reference_strategy="specified", reference_group="Black"
        )
        if len(result) > 0:
            assert result["reference_group"][0] == "Black"

    def test_specified_nonexistent_raises(self, metrics_df: pl.DataFrame) -> None:
        """Test that nonexistent specified reference raises error."""
        with pytest.raises(ValueError, match="not found"):
            compute_disparities(metrics_df, reference_strategy="specified", reference_group="Asian")

    def test_specified_without_group_raises(self, metrics_df: pl.DataFrame) -> None:
        """Test that specified strategy without group raises error."""
        with pytest.raises(ValueError, match="reference_group required"):
            compute_disparities(metrics_df, reference_strategy="specified")

    def test_filters_overall_row(self, metrics_df: pl.DataFrame) -> None:
        """Test that _overall row is filtered out."""
        result = compute_disparities(metrics_df)
        if len(result) > 0:
            groups = result["comparison_group"].to_list()
            assert "_overall" not in groups

    def test_empty_dataframe(self) -> None:
        """Test handling of empty DataFrame."""
        df = pl.DataFrame({"group": [], "n": [], "n_positive": [], "tpr": []})
        result = compute_disparities(df)
        assert len(result) == 0

    def test_single_group(self) -> None:
        """Test handling of single group."""
        df = pl.DataFrame({"group": ["White"], "n": [100], "n_positive": [50], "tpr": [0.80]})
        result = compute_disparities(df)
        assert len(result) == 0

    def test_result_columns(self, metrics_df: pl.DataFrame) -> None:
        """Test that result has expected columns."""
        result = compute_disparities(metrics_df)
        if len(result) > 0:
            expected_columns = {
                "reference_group",
                "comparison_group",
                "metric",
                "reference_value",
                "comparison_value",
                "difference",
                "diff_ci_lower",
                "diff_ci_upper",
                "ratio",
                "status",
                "p_value",
                "statistically_significant",
            }
            assert set(result.columns) == expected_columns


class TestGetWorstDisparity:
    """Tests for get_worst_disparity function."""

    def test_empty_dataframe(self) -> None:
        """Test handling of empty DataFrame."""
        df = pl.DataFrame(
            {
                "comparison_group": [],
                "metric": [],
                "difference": [],
            }
        )
        result = get_worst_disparity(df)
        assert result is None

    def test_returns_largest_absolute(self) -> None:
        """Test that largest absolute difference is returned."""
        df = pl.DataFrame(
            {
                "comparison_group": ["A", "B", "C"],
                "metric": ["tpr", "tpr", "tpr"],
                "difference": [0.05, -0.15, 0.10],
            }
        )
        result = get_worst_disparity(df)
        assert result is not None
        group, metric, value = result
        assert group == "B"
        assert metric == "tpr"
        assert value == -0.15

    def test_returns_tuple(self) -> None:
        """Test that function returns a tuple."""
        df = pl.DataFrame(
            {
                "comparison_group": ["A"],
                "metric": ["tpr"],
                "difference": [0.10],
            }
        )
        result = get_worst_disparity(df)
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 3


class TestCountByStatus:
    """Tests for count_by_status function."""

    def test_empty_dataframe(self) -> None:
        """Test handling of empty DataFrame."""
        df = pl.DataFrame({"status": []})
        result = count_by_status(df)
        assert result == {"pass": 0, "warn": 0, "fail": 0}

    def test_mixed_statuses(self) -> None:
        """Test counting of mixed statuses."""
        df = pl.DataFrame(
            {
                "status": ["pass", "pass", "warn", "fail"],
            }
        )
        result = count_by_status(df)
        assert result["pass"] == 2
        assert result["warn"] == 1
        assert result["fail"] == 1

    def test_all_pass(self) -> None:
        """Test when all are pass."""
        df = pl.DataFrame({"status": ["pass", "pass", "pass"]})
        result = count_by_status(df)
        assert result["pass"] == 3
        assert result["warn"] == 0
        assert result["fail"] == 0

    def test_all_fail(self) -> None:
        """Test when all are fail."""
        df = pl.DataFrame({"status": ["fail", "fail"]})
        result = count_by_status(df)
        assert result["pass"] == 0
        assert result["warn"] == 0
        assert result["fail"] == 2
