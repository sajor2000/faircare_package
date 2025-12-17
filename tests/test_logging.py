"""
Tests for FairCareAI logging module.

Tests cover:
- get_logger function
- configure_logging function
- set_log_level function
- disable_logging and enable_logging functions
- Logger constants
"""

import logging
import tempfile
from pathlib import Path

import pytest

from faircareai.core.logging import (
    DEFAULT_FORMAT,
    LOGGER_NAME,
    SIMPLE_FORMAT,
    configure_logging,
    disable_logging,
    enable_logging,
    get_logger,
    set_log_level,
)


class TestGetLogger:
    """Tests for get_logger function."""

    def test_returns_logger(self) -> None:
        """Test that function returns a Logger instance."""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)

    def test_faircareai_prefix_added(self) -> None:
        """Test that faircareai prefix is added to name."""
        logger = get_logger("my_module")
        assert logger.name == f"{LOGGER_NAME}.my_module"

    def test_faircareai_name_not_doubled(self) -> None:
        """Test that faircareai prefix is not doubled."""
        logger = get_logger("faircareai.core.metrics")
        assert logger.name == "faircareai.core.metrics"
        assert not logger.name.startswith("faircareai.faircareai")

    def test_root_logger_name(self) -> None:
        """Test that LOGGER_NAME itself works."""
        logger = get_logger(LOGGER_NAME)
        assert logger.name == LOGGER_NAME

    def test_empty_string_name(self) -> None:
        """Test with empty string name."""
        logger = get_logger("")
        assert logger.name == f"{LOGGER_NAME}."

    def test_nested_module_name(self) -> None:
        """Test with deeply nested module name."""
        logger = get_logger("a.b.c.d")
        assert logger.name == f"{LOGGER_NAME}.a.b.c.d"


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def teardown_method(self) -> None:
        """Reset logger state after each test."""
        logger = logging.getLogger(LOGGER_NAME)
        logger.handlers.clear()
        logger.setLevel(logging.WARNING)
        logger.disabled = False

    def test_sets_level_info(self) -> None:
        """Test that INFO level is set correctly."""
        configure_logging(level=logging.INFO)
        logger = logging.getLogger(LOGGER_NAME)
        assert logger.level == logging.INFO

    def test_sets_level_debug(self) -> None:
        """Test that DEBUG level is set correctly."""
        configure_logging(level=logging.DEBUG)
        logger = logging.getLogger(LOGGER_NAME)
        assert logger.level == logging.DEBUG

    def test_sets_level_warning(self) -> None:
        """Test that WARNING level is set correctly."""
        configure_logging(level=logging.WARNING)
        logger = logging.getLogger(LOGGER_NAME)
        assert logger.level == logging.WARNING

    def test_default_level_is_info(self) -> None:
        """Test that default level is INFO."""
        configure_logging()
        logger = logging.getLogger(LOGGER_NAME)
        assert logger.level == logging.INFO

    def test_stream_handler_added(self) -> None:
        """Test that stream handler is added by default."""
        configure_logging(stream=True)
        logger = logging.getLogger(LOGGER_NAME)
        stream_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(stream_handlers) == 1

    def test_no_stream_handler_when_disabled(self) -> None:
        """Test that stream handler is not added when stream=False."""
        configure_logging(stream=False)
        logger = logging.getLogger(LOGGER_NAME)
        stream_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) == 0

    def test_file_handler_added(self) -> None:
        """Test that file handler is added when filename provided."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            filename = f.name

        try:
            configure_logging(filename=filename)
            logger = logging.getLogger(LOGGER_NAME)
            file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
            assert len(file_handlers) == 1
        finally:
            Path(filename).unlink(missing_ok=True)

    def test_custom_format_string(self) -> None:
        """Test that custom format string is used."""
        custom_format = "%(message)s"
        configure_logging(format_string=custom_format)
        logger = logging.getLogger(LOGGER_NAME)
        if logger.handlers:
            formatter = logger.handlers[0].formatter
            assert formatter is not None
            assert formatter._fmt == custom_format

    def test_default_format_used(self) -> None:
        """Test that DEFAULT_FORMAT is used when no format provided."""
        configure_logging()
        logger = logging.getLogger(LOGGER_NAME)
        if logger.handlers:
            formatter = logger.handlers[0].formatter
            assert formatter is not None
            assert formatter._fmt == DEFAULT_FORMAT

    def test_clears_existing_handlers(self) -> None:
        """Test that existing handlers are cleared on reconfiguration."""
        configure_logging()
        logger = logging.getLogger(LOGGER_NAME)
        initial_count = len(logger.handlers)

        configure_logging()
        assert len(logger.handlers) == initial_count  # Should not accumulate

    def test_propagate_disabled(self) -> None:
        """Test that propagation to root logger is disabled."""
        configure_logging()
        logger = logging.getLogger(LOGGER_NAME)
        assert logger.propagate is False


class TestSetLogLevel:
    """Tests for set_log_level function."""

    def teardown_method(self) -> None:
        """Reset logger state after each test."""
        logger = logging.getLogger(LOGGER_NAME)
        logger.setLevel(logging.WARNING)

    def test_sets_debug_level(self) -> None:
        """Test setting DEBUG level."""
        set_log_level(logging.DEBUG)
        logger = logging.getLogger(LOGGER_NAME)
        assert logger.level == logging.DEBUG

    def test_sets_info_level(self) -> None:
        """Test setting INFO level."""
        set_log_level(logging.INFO)
        logger = logging.getLogger(LOGGER_NAME)
        assert logger.level == logging.INFO

    def test_sets_warning_level(self) -> None:
        """Test setting WARNING level."""
        set_log_level(logging.WARNING)
        logger = logging.getLogger(LOGGER_NAME)
        assert logger.level == logging.WARNING

    def test_sets_error_level(self) -> None:
        """Test setting ERROR level."""
        set_log_level(logging.ERROR)
        logger = logging.getLogger(LOGGER_NAME)
        assert logger.level == logging.ERROR

    def test_sets_critical_level(self) -> None:
        """Test setting CRITICAL level."""
        set_log_level(logging.CRITICAL)
        logger = logging.getLogger(LOGGER_NAME)
        assert logger.level == logging.CRITICAL


class TestDisableEnableLogging:
    """Tests for disable_logging and enable_logging functions."""

    def teardown_method(self) -> None:
        """Reset logger state after each test."""
        logger = logging.getLogger(LOGGER_NAME)
        logger.disabled = False

    def test_disable_logging_sets_disabled(self) -> None:
        """Test that disable_logging sets disabled flag."""
        disable_logging()
        logger = logging.getLogger(LOGGER_NAME)
        assert logger.disabled is True

    def test_enable_logging_clears_disabled(self) -> None:
        """Test that enable_logging clears disabled flag."""
        disable_logging()
        enable_logging()
        logger = logging.getLogger(LOGGER_NAME)
        assert logger.disabled is False

    def test_enable_on_already_enabled(self) -> None:
        """Test that enable_logging is safe when already enabled."""
        logger = logging.getLogger(LOGGER_NAME)
        logger.disabled = False
        enable_logging()
        assert logger.disabled is False

    def test_disable_on_already_disabled(self) -> None:
        """Test that disable_logging is safe when already disabled."""
        disable_logging()
        disable_logging()  # Should not raise
        logger = logging.getLogger(LOGGER_NAME)
        assert logger.disabled is True


class TestLoggerConstants:
    """Tests for logger constants."""

    def test_logger_name_value(self) -> None:
        """Test that LOGGER_NAME has expected value."""
        assert LOGGER_NAME == "faircareai"

    def test_default_format_contains_required_fields(self) -> None:
        """Test that DEFAULT_FORMAT contains expected fields."""
        assert "%(asctime)s" in DEFAULT_FORMAT
        assert "%(name)s" in DEFAULT_FORMAT
        assert "%(levelname)s" in DEFAULT_FORMAT
        assert "%(message)s" in DEFAULT_FORMAT

    def test_simple_format_contains_required_fields(self) -> None:
        """Test that SIMPLE_FORMAT contains expected fields."""
        assert "%(levelname)s" in SIMPLE_FORMAT
        assert "%(message)s" in SIMPLE_FORMAT


class TestLoggerIntegration:
    """Integration tests for logger functionality."""

    def teardown_method(self) -> None:
        """Reset logger state after each test."""
        logger = logging.getLogger(LOGGER_NAME)
        logger.handlers.clear()
        logger.setLevel(logging.WARNING)
        logger.disabled = False

    def test_child_logger_inherits_level(self) -> None:
        """Test that child loggers inherit parent level."""
        set_log_level(logging.DEBUG)
        parent = logging.getLogger(LOGGER_NAME)
        child = get_logger("child_module")

        # Child should get effective level from parent
        assert child.getEffectiveLevel() == logging.DEBUG

    def test_log_message_written(self) -> None:
        """Test that log messages are written to handlers."""
        import io

        stream = io.StringIO()
        configure_logging(level=logging.INFO, stream=False)

        logger = logging.getLogger(LOGGER_NAME)
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

        test_message = "Test log message"
        logger.info(test_message)

        output = stream.getvalue()
        assert test_message in output

    def test_file_logging_writes_to_file(self) -> None:
        """Test that file logging writes messages to file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            filename = f.name

        try:
            configure_logging(level=logging.INFO, stream=False, filename=filename)
            logger = logging.getLogger(LOGGER_NAME)

            test_message = "File log test message"
            logger.info(test_message)

            # Close handlers to flush
            for handler in logger.handlers:
                handler.close()

            content = Path(filename).read_text()
            assert test_message in content
        finally:
            Path(filename).unlink(missing_ok=True)
