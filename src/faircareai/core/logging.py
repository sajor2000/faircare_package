"""FairCareAI Logging Configuration.

Provides centralized logging infrastructure for the package.
All modules should use get_logger(__name__) to obtain a logger.

Usage:
    from faircareai.core.logging import get_logger
    logger = get_logger(__name__)

    logger.info("Processing data")
    logger.warning("Small sample size: %d", n)
    logger.debug("Bootstrap iteration %d failed", i)
"""

import logging
import sys
from typing import Final

# Package-wide logger name
LOGGER_NAME: Final[str] = "faircareai"

# Default format
DEFAULT_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
SIMPLE_FORMAT: Final[str] = "%(levelname)s - %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger for the given module name.

    Creates a child logger under the faircareai namespace to ensure
    consistent configuration across all modules.

    Args:
        name: Module name (typically __name__).

    Returns:
        Configured Logger instance.

    Example:
        >>> from faircareai.core.logging import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing %d records", 1000)
    """
    # If name starts with faircareai, use it directly
    # Otherwise, prepend faircareai to maintain hierarchy
    logger_name = name if name.startswith(LOGGER_NAME) else f"{LOGGER_NAME}.{name}"
    return logging.getLogger(logger_name)


def configure_logging(
    level: int = logging.INFO,
    format_string: str | None = None,
    stream: bool = True,
    filename: str | None = None,
) -> None:
    """Configure FairCareAI logging globally.

    Sets up the root faircareai logger with the specified configuration.
    Should be called once at application startup if custom logging is needed.

    Args:
        level: Logging level (default: INFO). Use logging.DEBUG for verbose output.
        format_string: Custom format string (optional). Defaults to timestamp format.
        stream: Whether to log to stdout (default: True).
        filename: Optional filename for file logging.

    Example:
        >>> from faircareai.core.logging import configure_logging
        >>> import logging
        >>> configure_logging(level=logging.DEBUG)
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    formatter = logging.Formatter(format_string or DEFAULT_FORMAT)

    if stream:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if filename:
        file_handler = logging.FileHandler(filename, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False


def set_log_level(level: int) -> None:
    """Set the logging level for all FairCareAI loggers.

    Convenience function to adjust verbosity without full reconfiguration.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.WARNING).

    Example:
        >>> import logging
        >>> from faircareai.core.logging import set_log_level
        >>> set_log_level(logging.DEBUG)  # Enable verbose output
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)


def disable_logging() -> None:
    """Disable all FairCareAI logging.

    Useful for testing or when running in production where logs are not needed.

    Example:
        >>> from faircareai.core.logging import disable_logging
        >>> disable_logging()
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.disabled = True


def enable_logging() -> None:
    """Re-enable FairCareAI logging after it was disabled.

    Example:
        >>> from faircareai.core.logging import enable_logging
        >>> enable_logging()
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.disabled = False


# Initialize default configuration on import
# Users can call configure_logging() to customize
_root_logger = logging.getLogger(LOGGER_NAME)
if not _root_logger.handlers:
    # Set up minimal default handler (WARNING level to avoid noise)
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter(SIMPLE_FORMAT))
    _root_logger.addHandler(_handler)
    _root_logger.setLevel(logging.WARNING)
    _root_logger.propagate = False
