"""
Tests for FairCareAI custom exceptions module.

Tests cover:
- FairCareAIError base exception
- InsufficientSampleError
- ConfigurationError
- DataValidationError
- MetricComputationError
- ColumnNotFoundError
- ReferenceGroupError
- BootstrapError
"""

import pytest

from faircareai.core.exceptions import (
    BootstrapError,
    ColumnNotFoundError,
    ConfigurationError,
    DataValidationError,
    FairCareAIError,
    InsufficientSampleError,
    MetricComputationError,
    ReferenceGroupError,
)


class TestFairCareAIError:
    """Tests for base FairCareAIError exception."""

    def test_basic_exception(self) -> None:
        """Test basic exception construction."""
        error = FairCareAIError("Test error message")
        assert str(error) == "Test error message"

    def test_catchable(self) -> None:
        """Test that exception can be caught."""
        with pytest.raises(FairCareAIError):
            raise FairCareAIError("Test")

    def test_inherits_from_exception(self) -> None:
        """Test that FairCareAIError inherits from Exception."""
        error = FairCareAIError("Test")
        assert isinstance(error, Exception)


class TestInsufficientSampleError:
    """Tests for InsufficientSampleError exception."""

    def test_basic_construction(self) -> None:
        """Test basic construction with n and min_required."""
        error = InsufficientSampleError(n=5, min_required=10)
        assert error.n == 5
        assert error.min_required == 10
        assert "5 < 10" in str(error)

    def test_with_context(self) -> None:
        """Test construction with context string."""
        error = InsufficientSampleError(n=5, min_required=10, context="computing AUROC")
        assert error.context == "computing AUROC"
        assert "computing AUROC" in str(error)

    def test_without_context(self) -> None:
        """Test construction without context."""
        error = InsufficientSampleError(n=3, min_required=5)
        assert error.context == ""
        assert "Insufficient sample size" in str(error)

    def test_attributes_set(self) -> None:
        """Test that all attributes are correctly set."""
        error = InsufficientSampleError(n=10, min_required=20, context="test")
        assert hasattr(error, "n")
        assert hasattr(error, "min_required")
        assert hasattr(error, "context")

    def test_inherits_from_faircareai_error(self) -> None:
        """Test inheritance from FairCareAIError."""
        error = InsufficientSampleError(n=1, min_required=2)
        assert isinstance(error, FairCareAIError)


class TestConfigurationError:
    """Tests for ConfigurationError exception."""

    def test_message_format(self) -> None:
        """Test error message formatting."""
        error = ConfigurationError(field="threshold", reason="must be between 0 and 1")
        assert "threshold" in str(error)
        assert "must be between 0 and 1" in str(error)

    def test_attributes_set(self) -> None:
        """Test that field and reason are stored."""
        error = ConfigurationError(field="n_bootstrap", reason="must be positive")
        assert error.field == "n_bootstrap"
        assert error.reason == "must be positive"

    def test_inherits_from_faircareai_error(self) -> None:
        """Test inheritance from FairCareAIError."""
        error = ConfigurationError(field="test", reason="test")
        assert isinstance(error, FairCareAIError)


class TestDataValidationError:
    """Tests for DataValidationError exception."""

    def test_with_column(self) -> None:
        """Test error message with column name."""
        error = DataValidationError(reason="contains null values", column="y_true")
        assert "y_true" in str(error)
        assert "contains null values" in str(error)

    def test_without_column(self) -> None:
        """Test error message without column name."""
        error = DataValidationError(reason="DataFrame is empty")
        assert "DataFrame is empty" in str(error)
        assert error.column is None

    def test_attributes_set(self) -> None:
        """Test that attributes are correctly set."""
        error = DataValidationError(reason="test reason", column="test_col")
        assert error.reason == "test reason"
        assert error.column == "test_col"

    def test_inherits_from_faircareai_error(self) -> None:
        """Test inheritance from FairCareAIError."""
        error = DataValidationError(reason="test")
        assert isinstance(error, FairCareAIError)


class TestMetricComputationError:
    """Tests for MetricComputationError exception."""

    def test_with_group(self) -> None:
        """Test error message with group name."""
        error = MetricComputationError(
            metric_name="AUROC", reason="single class in labels", group="Hispanic"
        )
        assert "AUROC" in str(error)
        assert "Hispanic" in str(error)
        assert "single class" in str(error)

    def test_without_group(self) -> None:
        """Test error message without group name."""
        error = MetricComputationError(metric_name="calibration", reason="no data")
        assert "calibration" in str(error)
        assert "no data" in str(error)
        assert error.group is None

    def test_attributes_set(self) -> None:
        """Test that attributes are correctly set."""
        error = MetricComputationError(metric_name="TPR", reason="division by zero", group="GroupA")
        assert error.metric_name == "TPR"
        assert error.reason == "division by zero"
        assert error.group == "GroupA"

    def test_inherits_from_faircareai_error(self) -> None:
        """Test inheritance from FairCareAIError."""
        error = MetricComputationError(metric_name="test", reason="test")
        assert isinstance(error, FairCareAIError)


class TestColumnNotFoundError:
    """Tests for ColumnNotFoundError exception."""

    def test_with_available(self) -> None:
        """Test error message with available columns."""
        error = ColumnNotFoundError(column="target", available=["y_true", "y_prob", "group"])
        assert "target" in str(error)
        assert "y_true" in str(error)

    def test_without_available(self) -> None:
        """Test error message without available columns."""
        error = ColumnNotFoundError(column="missing_col")
        assert "missing_col" in str(error)
        assert error.available is None

    def test_truncation_many_columns(self) -> None:
        """Test that many columns are truncated."""
        many_cols = [f"col_{i}" for i in range(20)]
        error = ColumnNotFoundError(column="target", available=many_cols)
        # Should show first 10 and mention how many more
        assert "10 more" in str(error)

    def test_attributes_set(self) -> None:
        """Test that attributes are correctly set."""
        error = ColumnNotFoundError(column="test", available=["a", "b"])
        assert error.column == "test"
        assert error.available == ["a", "b"]

    def test_inherits_from_faircareai_error(self) -> None:
        """Test inheritance from FairCareAIError."""
        error = ColumnNotFoundError(column="test")
        assert isinstance(error, FairCareAIError)


class TestReferenceGroupError:
    """Tests for ReferenceGroupError exception."""

    def test_message_format(self) -> None:
        """Test error message formatting."""
        error = ReferenceGroupError(reference="Asian", available=["White", "Black"])
        assert "Asian" in str(error)
        assert "White" in str(error)
        assert "Black" in str(error)

    def test_attributes_set(self) -> None:
        """Test that attributes are correctly set."""
        error = ReferenceGroupError(reference="Test", available=["A", "B", "C"])
        assert error.reference == "Test"
        assert error.available == ["A", "B", "C"]

    def test_inherits_from_faircareai_error(self) -> None:
        """Test inheritance from FairCareAIError."""
        error = ReferenceGroupError(reference="test", available=["a"])
        assert isinstance(error, FairCareAIError)


class TestBootstrapError:
    """Tests for BootstrapError exception."""

    def test_message_format(self) -> None:
        """Test basic error message formatting."""
        error = BootstrapError(reason="insufficient class variation")
        assert "insufficient class variation" in str(error)
        assert "Bootstrap CI" in str(error)

    def test_with_iteration_counts(self) -> None:
        """Test error message with iteration counts."""
        error = BootstrapError(reason="too many failures", n_successful=50, n_required=100)
        assert "50/100" in str(error)

    def test_without_iteration_counts(self) -> None:
        """Test error message without iteration counts."""
        error = BootstrapError(reason="test error")
        assert error.n_successful == 0
        assert error.n_required == 0

    def test_attributes_set(self) -> None:
        """Test that attributes are correctly set."""
        error = BootstrapError(reason="test", n_successful=25, n_required=50)
        assert error.reason == "test"
        assert error.n_successful == 25
        assert error.n_required == 50

    def test_inherits_from_faircareai_error(self) -> None:
        """Test inheritance from FairCareAIError."""
        error = BootstrapError(reason="test")
        assert isinstance(error, FairCareAIError)


class TestExceptionHierarchy:
    """Tests for exception hierarchy and catching."""

    def test_catch_all_with_base(self) -> None:
        """Test that all exceptions can be caught with FairCareAIError."""
        exceptions = [
            InsufficientSampleError(n=1, min_required=2),
            ConfigurationError(field="test", reason="test"),
            DataValidationError(reason="test"),
            MetricComputationError(metric_name="test", reason="test"),
            ColumnNotFoundError(column="test"),
            ReferenceGroupError(reference="test", available=["a"]),
            BootstrapError(reason="test"),
        ]

        for exc in exceptions:
            with pytest.raises(FairCareAIError):
                raise exc

    def test_specific_catch(self) -> None:
        """Test that specific exceptions can be caught individually."""
        with pytest.raises(InsufficientSampleError):
            raise InsufficientSampleError(n=1, min_required=10)

        with pytest.raises(ConfigurationError):
            raise ConfigurationError(field="x", reason="y")
