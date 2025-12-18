"""
Tests for FairCareAI core metrics module.

Tests cover:
- GroupMetrics dataclass
- _compute_confusion_matrix function
- _safe_divide function
- compute_metrics_for_group function
- compute_metric_ci function
- compute_group_metrics function
"""

import polars as pl
import pytest

from faircareai.core.metrics import (
    GroupMetrics,
    _compute_confusion_matrix,
    _safe_divide,
    compute_group_metrics,
    compute_metric_ci,
    compute_metrics_for_group,
)


class TestGroupMetrics:
    """Tests for GroupMetrics dataclass."""

    def test_dataclass_construction(self) -> None:
        """Test that GroupMetrics can be constructed with all fields."""
        gm = GroupMetrics(
            group="White",
            n=100,
            n_positive=30,
            n_negative=70,
            tp=25,
            fp=10,
            tn=60,
            fn=5,
            tpr=0.833,
            fpr=0.143,
            tnr=0.857,
            fnr=0.167,
            ppv=0.714,
            npv=0.923,
            accuracy=0.85,
            ci_method="wilson",
            sample_status="ADEQUATE",
            warning=None,
        )
        assert gm.group == "White"
        assert gm.n == 100
        assert gm.tpr == 0.833

    def test_dataclass_with_warning(self) -> None:
        """Test GroupMetrics with warning message."""
        gm = GroupMetrics(
            group="Small Group",
            n=15,
            n_positive=5,
            n_negative=10,
            tp=4,
            fp=3,
            tn=7,
            fn=1,
            tpr=0.8,
            fpr=0.3,
            tnr=0.7,
            fnr=0.2,
            ppv=0.571,
            npv=0.875,
            accuracy=0.733,
            ci_method="wilson",
            sample_status="LOW",
            warning="Small sample size for Small Group",
        )
        assert gm.warning is not None
        assert "Small" in gm.warning


class TestComputeConfusionMatrix:
    """Tests for _compute_confusion_matrix function."""

    def test_basic_computation(self) -> None:
        """Test basic confusion matrix computation."""
        y_true = pl.Series([1, 1, 0, 0, 1, 0])
        y_pred = pl.Series([1, 0, 0, 1, 1, 0])
        tp, fp, tn, fn = _compute_confusion_matrix(y_true, y_pred)
        assert tp == 2  # 1,1 and 1,1
        assert fn == 1  # 1,0
        assert tn == 2  # 0,0 and 0,0
        assert fp == 1  # 0,1

    def test_all_positive(self) -> None:
        """Test confusion matrix with all positive predictions."""
        y_true = pl.Series([1, 0, 1, 0])
        y_pred = pl.Series([1, 1, 1, 1])
        tp, fp, tn, fn = _compute_confusion_matrix(y_true, y_pred)
        assert tp == 2
        assert fp == 2
        assert tn == 0
        assert fn == 0

    def test_all_negative(self) -> None:
        """Test confusion matrix with all negative predictions."""
        y_true = pl.Series([1, 0, 1, 0])
        y_pred = pl.Series([0, 0, 0, 0])
        tp, fp, tn, fn = _compute_confusion_matrix(y_true, y_pred)
        assert tp == 0
        assert fp == 0
        assert tn == 2
        assert fn == 2

    def test_perfect_predictions(self) -> None:
        """Test confusion matrix with perfect predictions."""
        y_true = pl.Series([1, 0, 1, 0, 1])
        y_pred = pl.Series([1, 0, 1, 0, 1])
        tp, fp, tn, fn = _compute_confusion_matrix(y_true, y_pred)
        assert tp == 3
        assert fp == 0
        assert tn == 2
        assert fn == 0

    def test_empty_arrays(self) -> None:
        """Test confusion matrix with empty arrays."""
        y_true = pl.Series([], dtype=pl.Int64)
        y_pred = pl.Series([], dtype=pl.Int64)
        tp, fp, tn, fn = _compute_confusion_matrix(y_true, y_pred)
        assert tp == 0
        assert fp == 0
        assert tn == 0
        assert fn == 0


class TestSafeDivide:
    """Tests for _safe_divide function."""

    def test_normal_division(self) -> None:
        """Test normal division."""
        result = _safe_divide(10, 5)
        assert result == 2.0

    def test_zero_denominator_default(self) -> None:
        """Test zero denominator returns default."""
        result = _safe_divide(10, 0)
        assert result == 0.0

    def test_zero_denominator_custom(self) -> None:
        """Test zero denominator with custom default."""
        result = _safe_divide(10, 0, default=1.0)
        assert result == 1.0

    def test_zero_numerator(self) -> None:
        """Test zero numerator."""
        result = _safe_divide(0, 10)
        assert result == 0.0

    def test_both_zero(self) -> None:
        """Test both numerator and denominator zero."""
        result = _safe_divide(0, 0)
        assert result == 0.0

    def test_float_result(self) -> None:
        """Test division producing float."""
        result = _safe_divide(1, 3)
        assert abs(result - 0.333) < 0.01


class TestComputeMetricsForGroup:
    """Tests for compute_metrics_for_group function."""

    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        """Create sample DataFrame for testing."""
        return pl.DataFrame(
            {
                "y_true": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0] * 10,  # 100 rows
                "y_pred": [1, 1, 0, 0, 0, 0, 0, 0, 1, 0] * 10,
            }
        )

    def test_returns_group_metrics(self, sample_df: pl.DataFrame) -> None:
        """Test that function returns GroupMetrics instance."""
        result = compute_metrics_for_group(sample_df, "TestGroup")
        assert isinstance(result, GroupMetrics)

    def test_group_name_set(self, sample_df: pl.DataFrame) -> None:
        """Test that group name is correctly set."""
        result = compute_metrics_for_group(sample_df, "MyGroup")
        assert result.group == "MyGroup"

    def test_sample_size_correct(self, sample_df: pl.DataFrame) -> None:
        """Test that sample size is correct."""
        result = compute_metrics_for_group(sample_df, "Test")
        assert result.n == 100
        assert result.n_positive == 40  # 4 per 10 rows * 10
        assert result.n_negative == 60

    def test_confusion_matrix_correct(self, sample_df: pl.DataFrame) -> None:
        """Test that confusion matrix values are correct."""
        result = compute_metrics_for_group(sample_df, "Test")
        # Per 10 rows: TP=2, FN=2, TN=5, FP=1
        assert result.tp == 20
        assert result.fn == 20
        assert result.tn == 50
        assert result.fp == 10

    def test_tpr_computed(self, sample_df: pl.DataFrame) -> None:
        """Test that TPR is correctly computed."""
        result = compute_metrics_for_group(sample_df, "Test")
        expected_tpr = 20 / 40  # tp / (tp + fn)
        assert result.tpr == pytest.approx(expected_tpr, abs=0.01)

    def test_fpr_computed(self, sample_df: pl.DataFrame) -> None:
        """Test that FPR is correctly computed."""
        result = compute_metrics_for_group(sample_df, "Test")
        expected_fpr = 10 / 60  # fp / (fp + tn)
        assert result.fpr == pytest.approx(expected_fpr, abs=0.01)

    def test_ppv_computed(self, sample_df: pl.DataFrame) -> None:
        """Test that PPV is correctly computed."""
        result = compute_metrics_for_group(sample_df, "Test")
        expected_ppv = 20 / 30  # tp / (tp + fp)
        assert result.ppv == pytest.approx(expected_ppv, abs=0.01)

    def test_accuracy_computed(self, sample_df: pl.DataFrame) -> None:
        """Test that accuracy is correctly computed."""
        result = compute_metrics_for_group(sample_df, "Test")
        expected_accuracy = 70 / 100  # (tp + tn) / n
        assert result.accuracy == pytest.approx(expected_accuracy, abs=0.01)

    def test_sample_status_set(self, sample_df: pl.DataFrame) -> None:
        """Test that sample status is set."""
        result = compute_metrics_for_group(sample_df, "Test")
        assert result.sample_status in ["ADEQUATE", "MODERATE", "LOW", "VERY_LOW"]

    def test_ci_method_default(self, sample_df: pl.DataFrame) -> None:
        """Test default CI method is wilson."""
        result = compute_metrics_for_group(sample_df, "Test")
        assert result.ci_method == "wilson"

    def test_extreme_proportion_uses_clopper_pearson(self) -> None:
        """Test that extreme proportions use Clopper-Pearson."""
        # Create data with extreme proportion (< 1% positive)
        df = pl.DataFrame(
            {
                "y_true": [0] * 999 + [1],  # 0.1% positive
                "y_pred": [0] * 1000,
            }
        )
        result = compute_metrics_for_group(df, "Extreme")
        assert result.ci_method == "clopper_pearson"

    def test_all_positive_outcomes(self) -> None:
        """Test with all positive outcomes."""
        df = pl.DataFrame(
            {
                "y_true": [1] * 50,
                "y_pred": [1] * 30 + [0] * 20,
            }
        )
        result = compute_metrics_for_group(df, "AllPos")
        assert result.n_positive == 50
        assert result.n_negative == 0
        assert result.tpr == 30 / 50

    def test_all_negative_outcomes(self) -> None:
        """Test with all negative outcomes."""
        df = pl.DataFrame(
            {
                "y_true": [0] * 50,
                "y_pred": [0] * 40 + [1] * 10,
            }
        )
        result = compute_metrics_for_group(df, "AllNeg")
        assert result.n_positive == 0
        assert result.n_negative == 50
        # TPR should be 0 (default for 0/0)
        assert result.tpr == 0.0


class TestComputeMetricCI:
    """Tests for compute_metric_ci function."""

    @pytest.fixture
    def group_metrics(self) -> GroupMetrics:
        """Create GroupMetrics for CI testing."""
        return GroupMetrics(
            group="Test",
            n=100,
            n_positive=40,
            n_negative=60,
            tp=20,
            fp=10,
            tn=50,
            fn=20,
            tpr=0.5,
            fpr=0.167,
            tnr=0.833,
            fnr=0.5,
            ppv=0.667,
            npv=0.714,
            accuracy=0.7,
            ci_method="wilson",
            sample_status="ADEQUATE",
            warning=None,
        )

    def test_tpr_ci(self, group_metrics: GroupMetrics) -> None:
        """Test TPR CI computation."""
        lower, upper = compute_metric_ci("tpr", group_metrics)
        assert lower < group_metrics.tpr < upper
        assert 0.0 <= lower <= 1.0
        assert 0.0 <= upper <= 1.0

    def test_fpr_ci(self, group_metrics: GroupMetrics) -> None:
        """Test FPR CI computation."""
        lower, upper = compute_metric_ci("fpr", group_metrics)
        assert lower < group_metrics.fpr < upper

    def test_ppv_ci(self, group_metrics: GroupMetrics) -> None:
        """Test PPV CI computation."""
        lower, upper = compute_metric_ci("ppv", group_metrics)
        assert lower < group_metrics.ppv < upper

    def test_npv_ci(self, group_metrics: GroupMetrics) -> None:
        """Test NPV CI computation."""
        lower, upper = compute_metric_ci("npv", group_metrics)
        assert lower < group_metrics.npv < upper

    def test_accuracy_ci(self, group_metrics: GroupMetrics) -> None:
        """Test accuracy CI computation."""
        lower, upper = compute_metric_ci("accuracy", group_metrics)
        assert lower < group_metrics.accuracy < upper

    def test_tnr_ci(self, group_metrics: GroupMetrics) -> None:
        """Test TNR CI computation."""
        lower, upper = compute_metric_ci("tnr", group_metrics)
        assert lower < group_metrics.tnr < upper

    def test_fnr_ci(self, group_metrics: GroupMetrics) -> None:
        """Test FNR CI computation."""
        lower, upper = compute_metric_ci("fnr", group_metrics)
        assert lower < group_metrics.fnr < upper

    def test_unknown_metric_raises(self, group_metrics: GroupMetrics) -> None:
        """Test that unknown metric raises ValueError."""
        with pytest.raises(ValueError, match="Unknown metric"):
            compute_metric_ci("unknown_metric", group_metrics)

    def test_clopper_pearson_used_when_specified(self) -> None:
        """Test that Clopper-Pearson is used when ci_method specifies it."""
        gm = GroupMetrics(
            group="Test",
            n=100,
            n_positive=1,
            n_negative=99,
            tp=1,
            fp=0,
            tn=99,
            fn=0,
            tpr=1.0,
            fpr=0.0,
            tnr=1.0,
            fnr=0.0,
            ppv=1.0,
            npv=1.0,
            accuracy=1.0,
            ci_method="clopper_pearson",
            sample_status="ADEQUATE",
            warning=None,
        )
        lower, upper = compute_metric_ci("tpr", gm)
        # Should not raise and return valid CI
        assert 0.0 <= lower <= 1.0
        assert 0.0 <= upper <= 1.0


class TestComputeGroupMetrics:
    """Tests for compute_group_metrics function."""

    @pytest.fixture
    def multi_group_df(self) -> pl.DataFrame:
        """Create DataFrame with multiple demographic groups."""
        return pl.DataFrame(
            {
                "group": ["A"] * 60 + ["B"] * 40,
                "y_true": [1] * 20 + [0] * 40 + [1] * 15 + [0] * 25,
                "y_pred": [1] * 15 + [0] * 45 + [1] * 12 + [0] * 28,
            }
        )

    def test_returns_dataframe(self, multi_group_df: pl.DataFrame) -> None:
        """Test that function returns a DataFrame."""
        result = compute_group_metrics(multi_group_df, "group")
        assert isinstance(result, pl.DataFrame)

    def test_includes_all_groups(self, multi_group_df: pl.DataFrame) -> None:
        """Test that all groups are included."""
        result = compute_group_metrics(multi_group_df, "group")
        groups = result["group"].to_list()
        assert "A" in groups
        assert "B" in groups

    def test_includes_overall_row(self, multi_group_df: pl.DataFrame) -> None:
        """Test that overall row is included."""
        result = compute_group_metrics(multi_group_df, "group")
        groups = result["group"].to_list()
        assert "_overall" in groups

    def test_default_metrics(self, multi_group_df: pl.DataFrame) -> None:
        """Test that default metrics are tpr, fpr, ppv."""
        result = compute_group_metrics(multi_group_df, "group")
        assert "tpr" in result.columns
        assert "fpr" in result.columns
        assert "ppv" in result.columns

    def test_custom_metrics(self, multi_group_df: pl.DataFrame) -> None:
        """Test custom metric selection."""
        result = compute_group_metrics(multi_group_df, "group", metrics=["tpr", "tnr", "accuracy"])
        assert "tpr" in result.columns
        assert "tnr" in result.columns
        assert "accuracy" in result.columns
        assert "fpr" not in result.columns

    def test_ci_columns_present(self, multi_group_df: pl.DataFrame) -> None:
        """Test that CI columns are present."""
        result = compute_group_metrics(multi_group_df, "group")
        assert "tpr_ci_lower" in result.columns
        assert "tpr_ci_upper" in result.columns
        assert "fpr_ci_lower" in result.columns
        assert "fpr_ci_upper" in result.columns

    def test_n_column_correct(self, multi_group_df: pl.DataFrame) -> None:
        """Test that n column has correct values."""
        result = compute_group_metrics(multi_group_df, "group")
        a_row = result.filter(pl.col("group") == "A")
        b_row = result.filter(pl.col("group") == "B")
        overall_row = result.filter(pl.col("group") == "_overall")

        assert a_row["n"][0] == 60
        assert b_row["n"][0] == 40
        assert overall_row["n"][0] == 100

    def test_sample_status_column(self, multi_group_df: pl.DataFrame) -> None:
        """Test that sample_status column is present."""
        result = compute_group_metrics(multi_group_df, "group")
        assert "sample_status" in result.columns
        statuses = result["sample_status"].to_list()
        assert all(s in ["ADEQUATE", "MODERATE", "LOW", "VERY_LOW"] for s in statuses)

    def test_warning_column(self, multi_group_df: pl.DataFrame) -> None:
        """Test that warning column is present."""
        result = compute_group_metrics(multi_group_df, "group")
        assert "warning" in result.columns

    def test_custom_confidence(self, multi_group_df: pl.DataFrame) -> None:
        """Test with custom confidence level."""
        result_95 = compute_group_metrics(multi_group_df, "group", metrics=["tpr"], confidence=0.95)
        result_99 = compute_group_metrics(multi_group_df, "group", metrics=["tpr"], confidence=0.99)

        # 99% CI should be wider than 95% CI
        ci_width_95 = result_95["tpr_ci_upper"][0] - result_95["tpr_ci_lower"][0]
        ci_width_99 = result_99["tpr_ci_upper"][0] - result_99["tpr_ci_lower"][0]
        assert ci_width_99 > ci_width_95

    def test_small_group(self) -> None:
        """Test with small group size."""
        df = pl.DataFrame(
            {
                "group": ["Small"] * 15,
                "y_true": [1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                "y_pred": [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            }
        )
        result = compute_group_metrics(df, "group")
        small_row = result.filter(pl.col("group") == "Small")
        # Should work without errors
        assert small_row["n"][0] == 15

    def test_single_group(self) -> None:
        """Test with single group."""
        df = pl.DataFrame(
            {
                "group": ["Only"] * 50,
                "y_true": [1] * 20 + [0] * 30,
                "y_pred": [1] * 15 + [0] * 35,
            }
        )
        result = compute_group_metrics(df, "group")
        # Should have "Only" and "_overall"
        assert len(result) == 2
        groups = result["group"].to_list()
        assert "Only" in groups
        assert "_overall" in groups

    def test_many_groups(self) -> None:
        """Test with many groups."""
        groups = [f"G{i}" for i in range(10)]
        df = pl.DataFrame(
            {
                "group": groups * 20,  # 200 total rows
                "y_true": [1, 0] * 100,
                "y_pred": [1, 0] * 100,
            }
        )
        result = compute_group_metrics(df, "group")
        # Should have 10 groups + 1 overall
        assert len(result) == 11
