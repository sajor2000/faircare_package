"""
Security tests for report generation.

Tests XSS prevention, HTML escaping, and safe error handling in reports.
"""

import html
from unittest.mock import Mock

import pytest

from faircareai.reports.generator import (
    _generate_flags_section,
)


@pytest.fixture
def sample_audit_results():
    """Create a mock AuditResults object for testing."""
    results = Mock()
    results.flags = []
    return results


class TestXSSPrevention:
    """Test XSS vulnerability prevention in report generation."""

    def test_flag_message_xss_prevention(self, sample_audit_results):
        """Test that flag messages with HTML/JS are properly escaped."""
        # Create malicious flag with XSS payload
        malicious_message = '<script>alert("XSS")</script>'
        malicious_details = '<img src=x onerror="alert(1)">'

        sample_audit_results.flags = [
            {
                "severity": "error",
                "message": malicious_message,
                "details": malicious_details,
            }
        ]

        # Generate flags section
        html_output = _generate_flags_section(sample_audit_results)

        # Verify XSS payloads are escaped, not executed
        assert html.escape(malicious_message) in html_output
        assert html.escape(malicious_details) in html_output
        # The dangerous tags should be escaped
        assert '<script>' not in html_output
        assert '</script>' not in html_output
        assert '<img src=x' not in html_output
        # The escaped versions should be present
        assert '&lt;script&gt;' in html_output
        assert '&lt;img src=x' in html_output

    def test_flag_message_special_characters(self, sample_audit_results):
        """Test that special HTML characters in flags are properly escaped."""
        special_chars_message = 'Error: x < y && z > 0 & "quoted"'
        special_chars_details = "Detail with <brackets> and &ampersand"

        sample_audit_results.flags = [
            {
                "severity": "warning",
                "message": special_chars_message,
                "details": special_chars_details,
            }
        ]

        # Generate flags section
        html_output = _generate_flags_section(sample_audit_results)

        # Verify special characters are escaped
        assert html.escape(special_chars_message) in html_output
        assert html.escape(special_chars_details) in html_output
        assert "&lt;" in html_output  # < should be escaped
        assert "&gt;" in html_output  # > should be escaped
        assert "&amp;" in html_output  # & should be escaped

    def test_flag_empty_message(self, sample_audit_results):
        """Test that empty flag messages don't cause issues."""
        sample_audit_results.flags = [
            {
                "severity": "warning",
                "message": "",
                "details": "",
            }
        ]

        # Generate flags section
        html_output = _generate_flags_section(sample_audit_results)

        # Should not crash and should contain the severity
        assert "WARNING:" in html_output
        assert html_output is not None

    def test_flag_unicode_characters(self, sample_audit_results):
        """Test that unicode characters in flags are handled correctly."""
        unicode_message = "Error: température non valide → ≥ 100°C ⚠️"
        unicode_details = "Détails: vérifié ✓ échec ✗"

        sample_audit_results.flags = [
            {
                "severity": "error",
                "message": unicode_message,
                "details": unicode_details,
            }
        ]

        # Generate flags section
        html_output = _generate_flags_section(sample_audit_results)

        # Verify unicode is preserved (html.escape preserves unicode)
        # The escaped version should contain the unicode characters
        escaped_msg = html.escape(unicode_message)
        assert escaped_msg in html_output or unicode_message in html_output

    def test_multiple_flags_with_mixed_content(self, sample_audit_results):
        """Test multiple flags with different content types."""
        sample_audit_results.flags = [
            {
                "severity": "error",
                "message": "Normal error message",
                "details": "Normal details",
            },
            {
                "severity": "warning",
                "message": '<script>alert("test")</script>',
                "details": "Safe details",
            },
            {
                "severity": "error",
                "message": "Error with <tags>",
                "details": '<a href="javascript:void(0)">Click</a>',
            },
        ]

        # Generate flags section
        html_output = _generate_flags_section(sample_audit_results)

        # Verify all malicious content is escaped (tags are escaped, not removed)
        assert '<script>' not in html_output
        assert '</script>' not in html_output
        assert '<a href=' not in html_output
        # The escaped versions should be present
        assert '&lt;script&gt;' in html_output
        assert '&lt;a href=' in html_output
        assert html.escape('<script>alert("test")</script>') in html_output
        assert html.escape('<a href="javascript:void(0)">Click</a>') in html_output

    def test_no_flags(self, sample_audit_results):
        """Test report generation with no flags."""
        sample_audit_results.flags = []

        # Generate flags section
        html_output = _generate_flags_section(sample_audit_results)

        # Should show "no flags" message
        assert "No flags or warnings raised" in html_output


class TestExceptionMessageSafety:
    """Test that exception messages in reports are safely handled."""

    def test_exception_messages_are_escaped(self):
        """Test that exception messages with HTML are escaped in error placeholders."""
        # This test verifies the pattern but can't easily test runtime exceptions
        # The key is that we use html.escape(str(e)) in all exception handlers

        # Test the escape function works correctly
        malicious_error = '<script>alert("error")</script>'
        escaped = html.escape(malicious_error)

        assert escaped == '&lt;script&gt;alert(&quot;error&quot;)&lt;/script&gt;'
        # The key test: script tags are escaped and won't execute
        assert '<script>' not in escaped
        assert '</script>' not in escaped
        # alert() is still in the text but can't execute without script tags
        assert 'alert(&quot;error&quot;)' in escaped


class TestHTMLEscapeFunction:
    """Test the html.escape function behavior used throughout reports."""

    def test_escape_basic_html(self):
        """Test basic HTML escaping."""
        input_str = '<div>Test</div>'
        expected = '&lt;div&gt;Test&lt;/div&gt;'
        assert html.escape(input_str) == expected

    def test_escape_script_tags(self):
        """Test script tag escaping."""
        input_str = '<script>alert(1)</script>'
        expected = '&lt;script&gt;alert(1)&lt;/script&gt;'
        assert html.escape(input_str) == expected

    def test_escape_quotes(self):
        """Test quote escaping."""
        input_str = 'He said "Hello"'
        expected = 'He said &quot;Hello&quot;'
        assert html.escape(input_str) == expected

    def test_escape_ampersand(self):
        """Test ampersand escaping."""
        input_str = 'X & Y'
        expected = 'X &amp; Y'
        assert html.escape(input_str) == expected

    def test_escape_less_than_greater_than(self):
        """Test < and > escaping."""
        input_str = 'if x < 10 and y > 5'
        expected = 'if x &lt; 10 and y &gt; 5'
        assert html.escape(input_str) == expected

    def test_escape_preserves_unicode(self):
        """Test that unicode is preserved during escaping."""
        input_str = 'Température: 25°C ☀️'
        # Unicode should be preserved, only HTML chars escaped
        result = html.escape(input_str)
        assert '°C' in result
        assert '☀️' in result

    def test_escape_empty_string(self):
        """Test escaping empty string."""
        assert html.escape('') == ''

    def test_escape_none_converted_to_string(self):
        """Test that None is converted before escaping."""
        # In our code we use str(e) before escaping
        result = html.escape(str(None))
        assert result == 'None'
