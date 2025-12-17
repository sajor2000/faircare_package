"""Tests for input validation and binary ML model compatibility.

Ensures FairCareAI works with ANY binary ML healthcare model producing
probability scores in [0, 1].
"""

import numpy as np
import polars as pl
import pytest

from faircareai import FairCareAudit
from faircareai.core.exceptions import DataValidationError


class TestPandasInput:
    """Test pandas DataFrame auto-conversion to Polars."""

    def test_pandas_dataframe_accepted(self):
        """Pandas DataFrames should be auto-converted to Polars."""
        pd = pytest.importorskip("pandas")
        np.random.seed(42)
        pdf = pd.DataFrame(
            {
                "risk": np.random.uniform(0, 1, 100),
                "outcome": np.random.binomial(1, 0.3, 100),
                "group": ["A"] * 50 + ["B"] * 50,
            }
        )
        audit = FairCareAudit(pdf, "risk", "outcome")
        assert isinstance(audit.df, pl.DataFrame)

    def test_pandas_with_nan_rejected(self):
        """Pandas DataFrames with NaN in predictions should be rejected."""
        pd = pytest.importorskip("pandas")
        pdf = pd.DataFrame(
            {
                "risk": [0.3, np.nan, 0.5, 0.4],
                "outcome": [0, 1, 0, 1],
                "group": ["A", "A", "B", "B"],
            }
        )
        with pytest.raises(DataValidationError, match="null/NaN"):
            FairCareAudit(pdf, "risk", "outcome")

    def test_pandas_with_nan_in_target_rejected(self):
        """Pandas DataFrames with NaN in targets should be rejected."""
        pd = pytest.importorskip("pandas")
        pdf = pd.DataFrame(
            {
                "risk": [0.3, 0.4, 0.5, 0.6],
                "outcome": [0, np.nan, 0, 1],
                "group": ["A", "A", "B", "B"],
            }
        )
        with pytest.raises(DataValidationError, match="null/NaN"):
            FairCareAudit(pdf, "risk", "outcome")


class TestPolarsInput:
    """Test Polars DataFrame handling."""

    def test_polars_dataframe_direct(self):
        """Polars DataFrames should work directly without conversion."""
        np.random.seed(42)
        df = pl.DataFrame(
            {
                "risk": np.random.uniform(0, 1, 100).tolist(),
                "outcome": np.random.binomial(1, 0.3, 100).tolist(),
                "group": ["A"] * 50 + ["B"] * 50,
            }
        )
        audit = FairCareAudit(df, "risk", "outcome")
        assert audit.df is df  # Same object, no conversion

    def test_polars_with_null_rejected(self):
        """Polars DataFrames with null values should be rejected."""
        df = pl.DataFrame(
            {
                "risk": [0.3, None, 0.5, 0.4],
                "outcome": [0, 1, 0, 1],
                "group": ["A", "A", "B", "B"],
            }
        )
        with pytest.raises(DataValidationError, match="null/NaN"):
            FairCareAudit(df, "risk", "outcome")


class TestBinaryTargetValidation:
    """Test that only binary targets (0/1) are accepted."""

    def test_multiclass_rejected(self):
        """Multi-class targets should be rejected."""
        df = pl.DataFrame(
            {
                "risk": [0.3, 0.4, 0.5, 0.6],
                "outcome": [0, 1, 2, 0],  # Multi-class!
                "group": ["A"] * 4,
            }
        )
        with pytest.raises(DataValidationError, match="binary"):
            FairCareAudit(df, "risk", "outcome")

    def test_negative_target_rejected(self):
        """Negative target values should be rejected."""
        df = pl.DataFrame(
            {
                "risk": [0.3, 0.4, 0.5, 0.6],
                "outcome": [-1, 0, 1, 0],
                "group": ["A"] * 4,
            }
        )
        with pytest.raises(DataValidationError, match="binary"):
            FairCareAudit(df, "risk", "outcome")

    def test_integer_binary_accepted(self):
        """Integer binary targets (0, 1) should be accepted."""
        np.random.seed(42)
        df = pl.DataFrame(
            {
                "risk": np.random.uniform(0, 1, 100).tolist(),
                "outcome": [0, 1] * 50,
                "group": ["A"] * 50 + ["B"] * 50,
            }
        )
        audit = FairCareAudit(df, "risk", "outcome")
        assert audit.df is not None

    def test_float_binary_accepted(self):
        """Float binary targets (0.0, 1.0) should be accepted."""
        np.random.seed(42)
        df = pl.DataFrame(
            {
                "risk": np.random.uniform(0, 1, 100).tolist(),
                "outcome": [0.0, 1.0] * 50,
                "group": ["A"] * 50 + ["B"] * 50,
            }
        )
        audit = FairCareAudit(df, "risk", "outcome")
        assert audit.df is not None


class TestPredictionRangeValidation:
    """Test that predictions must be in [0, 1] range."""

    def test_predictions_above_one_rejected(self):
        """Predictions > 1 should be rejected."""
        df = pl.DataFrame(
            {
                "risk": [0.3, 1.5, 0.5, 0.4],  # 1.5 is out of range
                "outcome": [0, 1, 0, 1],
                "group": ["A"] * 4,
            }
        )
        with pytest.raises(DataValidationError, match=r"\[0.*1\]"):
            FairCareAudit(df, "risk", "outcome")

    def test_predictions_below_zero_rejected(self):
        """Predictions < 0 should be rejected."""
        df = pl.DataFrame(
            {
                "risk": [0.3, -0.2, 0.5, 0.4],  # -0.2 is out of range
                "outcome": [0, 1, 0, 1],
                "group": ["A"] * 4,
            }
        )
        with pytest.raises(DataValidationError, match=r"\[0.*1\]"):
            FairCareAudit(df, "risk", "outcome")

    def test_valid_probability_range_accepted(self):
        """Predictions in [0, 1] should be accepted."""
        np.random.seed(42)
        df = pl.DataFrame(
            {
                "risk": np.random.uniform(0, 1, 100).tolist(),
                "outcome": [0, 1] * 50,
                "group": ["A"] * 50 + ["B"] * 50,
            }
        )
        audit = FairCareAudit(df, "risk", "outcome")
        assert audit.df is not None

    def test_edge_values_zero_one_accepted(self):
        """Edge values 0.0 and 1.0 should be accepted."""
        df = pl.DataFrame(
            {
                "risk": [0.0, 1.0, 0.5, 0.5] * 25,
                "outcome": [0, 1, 0, 1] * 25,
                "group": ["A"] * 50 + ["B"] * 50,
            }
        )
        audit = FairCareAudit(df, "risk", "outcome")
        assert audit.df is not None


class TestTypeMismatch:
    """Test handling of incorrect data types."""

    def test_string_predictions_rejected(self):
        """String predictions should be rejected."""
        df = pl.DataFrame(
            {
                "risk": ["low", "high", "medium", "low"],
                "outcome": [0, 1, 0, 1],
                "group": ["A"] * 4,
            }
        )
        with pytest.raises(DataValidationError, match="numeric"):
            FairCareAudit(df, "risk", "outcome")

    def test_string_targets_rejected(self):
        """String targets should be rejected."""
        df = pl.DataFrame(
            {
                "risk": [0.3, 0.7, 0.5, 0.4],
                "outcome": ["no", "yes", "no", "yes"],
                "group": ["A"] * 4,
            }
        )
        with pytest.raises(DataValidationError, match="numeric"):
            FairCareAudit(df, "risk", "outcome")


class TestMissingColumns:
    """Test handling of missing required columns."""

    def test_missing_prediction_column_rejected(self):
        """Missing prediction column should raise error."""
        df = pl.DataFrame(
            {
                "outcome": [0, 1, 0, 1],
                "group": ["A"] * 4,
            }
        )
        with pytest.raises(DataValidationError, match="Missing.*columns"):
            FairCareAudit(df, "risk", "outcome")

    def test_missing_target_column_rejected(self):
        """Missing target column should raise error."""
        df = pl.DataFrame(
            {
                "risk": [0.3, 0.7, 0.5, 0.4],
                "group": ["A"] * 4,
            }
        )
        with pytest.raises(DataValidationError, match="Missing.*columns"):
            FairCareAudit(df, "risk", "outcome")


class TestPPVRatioNoneHandling:
    """Test that PPV ratio None values don't cause crashes."""

    def test_ppv_with_zero_reference_positives(self):
        """PPV ratio should handle reference group with no true positives."""
        from faircareai.metrics.fairness import compute_fairness_metrics

        # Reference group (White) has no true positives at threshold
        df = pl.DataFrame(
            {
                "y_prob": [0.3, 0.4] * 50 + [0.7, 0.8] * 50,
                "y_true": [0, 0] * 50 + [1, 1] * 50,
                "group": ["White"] * 100 + ["Black"] * 100,
            }
        )
        result = compute_fairness_metrics(
            df, "y_prob", "y_true", "group", threshold=0.5, reference="White"
        )
        # Should not crash - None values filtered
        assert isinstance(result, dict)
        assert "ppv_ratio" in result

    def test_fairness_summary_with_none_ppv(self):
        """Fairness summary should handle None PPV ratios gracefully."""
        from faircareai.metrics.fairness import _compute_fairness_summary

        # Metrics with None PPV ratio (occurs when reference has 0 TP)
        metrics = {
            "tpr_ratio": {"A": 1.0, "B": 0.9},
            "fpr_ratio": {"A": 1.0, "B": 1.1},
            "ppv_ratio": {"A": 1.0, "B": None},  # None value
        }
        result = _compute_fairness_summary(metrics)
        # Should not crash - None values filtered
        assert isinstance(result, dict)
        assert "predictive_parity" in result


class TestReferenceGroupTypeCoercion:
    """Test that reference group handles different types correctly."""

    def test_integer_reference_group(self):
        """Integer reference group should be converted to string."""
        from faircareai.metrics.fairness import compute_fairness_metrics

        np.random.seed(42)
        df = pl.DataFrame(
            {
                "y_prob": np.random.uniform(0, 1, 200).tolist(),
                "y_true": [0, 1] * 100,
                "group": [1] * 100 + [2] * 100,  # Integer groups
            }
        )
        result = compute_fairness_metrics(
            df, "y_prob", "y_true", "group", threshold=0.5, reference=1
        )
        # Reference should be coerced to string
        assert result["reference"] == "1"


class TestUnsupportedInputTypes:
    """Test handling of unsupported input types."""

    def test_list_input_rejected(self):
        """List input should be rejected with helpful error."""
        data = [[0.3, 0, "A"], [0.7, 1, "B"]]
        with pytest.raises(TypeError, match="Expected"):
            FairCareAudit(data, "risk", "outcome")

    def test_dict_input_rejected(self):
        """Dict input should be rejected with helpful error."""
        data = {"risk": [0.3, 0.7], "outcome": [0, 1]}
        with pytest.raises(TypeError, match="Expected"):
            FairCareAudit(data, "risk", "outcome")

    def test_numpy_array_rejected(self):
        """NumPy array should be rejected with helpful error."""
        data = np.array([[0.3, 0], [0.7, 1]])
        with pytest.raises(TypeError, match="Expected"):
            FairCareAudit(data, "risk", "outcome")
