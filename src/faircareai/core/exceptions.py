"""FairCareAI Custom Exceptions.

Provides specific exception types for better error handling and debugging.
All exceptions inherit from FairCareAIError for easy catching.

Usage:
    from faircareai.core.exceptions import InsufficientSampleError

    if n < MIN_SAMPLE_SIZE:
        raise InsufficientSampleError(n, MIN_SAMPLE_SIZE, "computing AUROC")
"""


class FairCareAIError(Exception):
    """Base exception for all FairCareAI errors.

    All custom exceptions in FairCareAI inherit from this class,
    allowing users to catch all package-specific errors with a single except clause.

    Example:
        >>> try:
        ...     audit.run()
        ... except FairCareAIError as e:
        ...     print(f"FairCareAI error: {e}")
    """

    pass


class InsufficientSampleError(FairCareAIError):
    """Raised when sample size is too small for reliable computation.

    This error indicates that a statistical computation cannot be performed
    reliably due to insufficient data.

    Attributes:
        n: Actual sample size.
        min_required: Minimum required sample size.
        context: Description of the operation that failed.
    """

    def __init__(self, n: int, min_required: int, context: str = "") -> None:
        """Initialize InsufficientSampleError.

        Args:
            n: Actual sample size available.
            min_required: Minimum sample size required for the operation.
            context: Optional description of what operation was attempted.
        """
        self.n = n
        self.min_required = min_required
        self.context = context

        message = f"Insufficient sample size: {n} < {min_required}"
        if context:
            message = f"{message}. Context: {context}"

        super().__init__(message)


class ConfigurationError(FairCareAIError):
    """Raised when configuration is invalid or incomplete.

    This error indicates that the FairnessConfig or other configuration
    objects have invalid or missing required values.

    Attributes:
        field: The configuration field that is invalid.
        reason: Why the configuration is invalid.
    """

    def __init__(self, field: str, reason: str) -> None:
        """Initialize ConfigurationError.

        Args:
            field: Name of the invalid configuration field.
            reason: Description of why the configuration is invalid.
        """
        self.field = field
        self.reason = reason

        message = f"Invalid configuration for '{field}': {reason}"
        super().__init__(message)


class DataValidationError(FairCareAIError):
    """Raised when data validation fails.

    This error indicates that the input data does not meet the required
    format, schema, or quality standards.

    Attributes:
        column: The column that failed validation (if applicable).
        reason: Why the validation failed.
    """

    def __init__(self, reason: str, column: str | None = None) -> None:
        """Initialize DataValidationError.

        Args:
            reason: Description of the validation failure.
            column: Optional column name that failed validation.
        """
        self.column = column
        self.reason = reason

        if column:
            message = f"Data validation failed for column '{column}': {reason}"
        else:
            message = f"Data validation failed: {reason}"

        super().__init__(message)


class MetricComputationError(FairCareAIError):
    """Raised when a metric cannot be computed.

    This error indicates that a specific metric computation failed,
    typically due to data issues (e.g., all predictions are the same class).

    Attributes:
        metric_name: Name of the metric that failed.
        reason: Why the computation failed.
        group: Optional group name if this was a per-group computation.
    """

    def __init__(
        self,
        metric_name: str,
        reason: str,
        group: str | None = None,
    ) -> None:
        """Initialize MetricComputationError.

        Args:
            metric_name: Name of the metric that could not be computed.
            reason: Why the computation failed.
            group: Optional group name for subgroup-specific failures.
        """
        self.metric_name = metric_name
        self.reason = reason
        self.group = group

        if group:
            message = f"Cannot compute {metric_name} for group '{group}': {reason}"
        else:
            message = f"Cannot compute {metric_name}: {reason}"

        super().__init__(message)


class ColumnNotFoundError(FairCareAIError):
    """Raised when a required column is not found in the DataFrame.

    Attributes:
        column: The missing column name.
        available: List of available column names (if known).
    """

    def __init__(self, column: str, available: list[str] | None = None) -> None:
        """Initialize ColumnNotFoundError.

        Args:
            column: Name of the missing column.
            available: Optional list of available columns for help message.
        """
        self.column = column
        self.available = available

        message = f"Column '{column}' not found in DataFrame"
        if available:
            message = f"{message}. Available columns: {', '.join(available[:10])}"
            if len(available) > 10:
                message = f"{message}, ... ({len(available) - 10} more)"

        super().__init__(message)


class ReferenceGroupError(FairCareAIError):
    """Raised when the specified reference group is invalid.

    Attributes:
        reference: The specified reference group.
        available: List of available group values.
    """

    def __init__(self, reference: str, available: list[str]) -> None:
        """Initialize ReferenceGroupError.

        Args:
            reference: The invalid reference group name.
            available: List of valid group values.
        """
        self.reference = reference
        self.available = available

        message = (
            f"Reference group '{reference}' not found. "
            f"Available groups: {', '.join(str(g) for g in available)}"
        )
        super().__init__(message)


class BootstrapError(FairCareAIError):
    """Raised when bootstrap confidence interval computation fails.

    Attributes:
        reason: Why the bootstrap failed.
        n_successful: Number of successful bootstrap iterations.
        n_required: Minimum required successful iterations.
    """

    def __init__(
        self,
        reason: str,
        n_successful: int = 0,
        n_required: int = 0,
    ) -> None:
        """Initialize BootstrapError.

        Args:
            reason: Why the bootstrap computation failed.
            n_successful: Number of successful bootstrap iterations completed.
            n_required: Minimum required successful iterations.
        """
        self.reason = reason
        self.n_successful = n_successful
        self.n_required = n_required

        message = f"Bootstrap CI computation failed: {reason}"
        if n_successful > 0 or n_required > 0:
            message = f"{message} ({n_successful}/{n_required} successful iterations)"

        super().__init__(message)
