"""
Tests for FairCareAI visualization tables module.

Tests cover:
- create_executive_scorecard function
- create_plain_language_summary function
"""

import pytest

from faircareai.visualization.tables import (
    create_executive_scorecard,
    create_plain_language_summary,
)
from faircareai.visualization.themes import SEMANTIC_COLORS, TYPOGRAPHY


class TestCreateExecutiveScorecard:
    """Tests for create_executive_scorecard function."""

    def test_returns_gt_object(self) -> None:
        """Test that function returns a GT object."""
        pytest.importorskip("great_tables")
        result = create_executive_scorecard(5, 3, 2, 1000, 4)
        # Should have GT type
        assert result is not None
        assert "GT" in type(result).__name__

    def test_all_zeros(self) -> None:
        """Test with all zero counts."""
        pytest.importorskip("great_tables")
        result = create_executive_scorecard(0, 0, 0, 100, 2)
        assert result is not None

    def test_all_pass(self) -> None:
        """Test with all pass counts."""
        pytest.importorskip("great_tables")
        result = create_executive_scorecard(10, 0, 0, 500, 3)
        assert result is not None

    def test_all_flag(self) -> None:
        """Test with all flag counts."""
        pytest.importorskip("great_tables")
        result = create_executive_scorecard(0, 0, 10, 500, 3)
        assert result is not None

    def test_all_warn(self) -> None:
        """Test with all warn counts."""
        pytest.importorskip("great_tables")
        result = create_executive_scorecard(0, 10, 0, 500, 3)
        assert result is not None

    def test_custom_model_name(self) -> None:
        """Test with custom model name."""
        pytest.importorskip("great_tables")
        result = create_executive_scorecard(5, 3, 2, 1000, 4, model_name="ICU Mortality")
        assert result is not None

    def test_large_sample_size(self) -> None:
        """Test with large sample size (comma formatting)."""
        pytest.importorskip("great_tables")
        result = create_executive_scorecard(5, 3, 2, 1000000, 4)
        assert result is not None

    def test_single_group(self) -> None:
        """Test with single group."""
        pytest.importorskip("great_tables")
        result = create_executive_scorecard(5, 3, 2, 100, 1)
        assert result is not None

    def test_many_groups(self) -> None:
        """Test with many groups."""
        pytest.importorskip("great_tables")
        result = create_executive_scorecard(5, 3, 2, 100, 20)
        assert result is not None


class TestCreatePlainLanguageSummary:
    """Tests for create_plain_language_summary function."""

    def test_returns_string(self) -> None:
        """Test that function returns a string."""
        result = create_plain_language_summary(5, 3, 2, "Group A", "TPR", 0.15)
        assert isinstance(result, str)

    def test_flag_status(self) -> None:
        """Test status when flags present."""
        result = create_plain_language_summary(5, 3, 2, "Group A", "TPR", 0.15)
        assert "REVIEW SUGGESTED" in result
        assert SEMANTIC_COLORS["fail"] in result

    def test_warn_status_no_flags(self) -> None:
        """Test status when warnings but no flags."""
        result = create_plain_language_summary(5, 3, 0, "Group A", "TPR", 0.08)
        assert "CONSIDERATIONS NOTED" in result
        assert SEMANTIC_COLORS["warn"] in result

    def test_pass_status_no_flags_no_warns(self) -> None:
        """Test status when no flags and no warnings."""
        result = create_plain_language_summary(10, 0, 0, "Group A", "TPR", 0.02)
        assert "NO FLAGS" in result
        assert SEMANTIC_COLORS["pass"] in result

    def test_contains_worst_group(self) -> None:
        """Test that result contains worst group name."""
        result = create_plain_language_summary(5, 3, 2, "Black/African American", "TPR", 0.15)
        assert "Black/African American" in result

    def test_contains_worst_metric(self) -> None:
        """Test that result contains worst metric name."""
        result = create_plain_language_summary(5, 3, 2, "Group A", "Sensitivity", 0.15)
        assert "Sensitivity" in result

    def test_contains_percentage(self) -> None:
        """Test that result contains percentage value."""
        result = create_plain_language_summary(5, 3, 2, "Group A", "TPR", 0.15)
        # 0.15 * 100 = 15.0%
        assert "15.0%" in result

    def test_negative_disparity_uses_absolute(self) -> None:
        """Test that negative disparity shows as positive percentage."""
        result = create_plain_language_summary(5, 3, 2, "Group A", "TPR", -0.15)
        # Should show 15.0% not -15.0%
        assert "15.0%" in result

    def test_contains_advisory_disclaimer(self) -> None:
        """Test that result contains advisory disclaimer."""
        result = create_plain_language_summary(5, 3, 2, "Group A", "TPR", 0.15)
        assert "Advisory guidance" in result
        assert "clinical stakeholders" in result

    def test_contains_font_family(self) -> None:
        """Test that result contains correct font family."""
        result = create_plain_language_summary(5, 3, 2, "Group A", "TPR", 0.15)
        assert TYPOGRAPHY["data_font"] in result

    def test_zero_disparity(self) -> None:
        """Test with zero disparity value."""
        result = create_plain_language_summary(10, 0, 0, "Group A", "TPR", 0.0)
        assert "0.0%" in result

    def test_large_disparity(self) -> None:
        """Test with large disparity value."""
        result = create_plain_language_summary(0, 0, 5, "Group A", "TPR", 0.50)
        assert "50.0%" in result

    def test_small_disparity(self) -> None:
        """Test with small disparity value."""
        result = create_plain_language_summary(10, 0, 0, "Group A", "TPR", 0.001)
        assert "0.1%" in result

    def test_html_structure(self) -> None:
        """Test that result has proper HTML structure."""
        result = create_plain_language_summary(5, 3, 2, "Group A", "TPR", 0.15)
        assert "<div" in result
        assert "</div>" in result
        assert "<p" in result
        assert "</p>" in result

    def test_flag_count_one(self) -> None:
        """Test with exactly one flag."""
        result = create_plain_language_summary(5, 3, 1, "Group A", "TPR", 0.15)
        assert "REVIEW SUGGESTED" in result

    def test_warn_count_one_no_flags(self) -> None:
        """Test with exactly one warning and no flags."""
        result = create_plain_language_summary(5, 1, 0, "Group A", "TPR", 0.08)
        assert "CONSIDERATIONS NOTED" in result

    def test_all_zeros(self) -> None:
        """Test with all zero counts."""
        result = create_plain_language_summary(0, 0, 0, "Group A", "TPR", 0.02)
        assert "NO FLAGS" in result

    def test_disparities_flagged_message(self) -> None:
        """Test that flag status includes correct advisory message."""
        result = create_plain_language_summary(0, 0, 5, "Group A", "TPR", 0.20)
        assert "Disparities flagged for clinical review" in result

    def test_considerations_message(self) -> None:
        """Test that warn status includes correct advisory message."""
        result = create_plain_language_summary(5, 3, 0, "Group A", "TPR", 0.08)
        assert "Some metrics may warrant discussion" in result

    def test_no_disparities_message(self) -> None:
        """Test that pass status includes correct advisory message."""
        result = create_plain_language_summary(10, 0, 0, "Group A", "TPR", 0.02)
        assert "No significant disparities detected at current thresholds" in result

    def test_inline_style_present(self) -> None:
        """Test that inline styles are present."""
        result = create_plain_language_summary(5, 3, 2, "Group A", "TPR", 0.15)
        assert "style=" in result
        assert "background:" in result
        assert "color:" in result

    def test_border_radius_styling(self) -> None:
        """Test that border radius is applied."""
        result = create_plain_language_summary(5, 3, 2, "Group A", "TPR", 0.15)
        assert "border-radius" in result

    def test_largest_disparity_label(self) -> None:
        """Test that 'Largest disparity' label is present."""
        result = create_plain_language_summary(5, 3, 2, "Group A", "TPR", 0.15)
        assert "Largest disparity" in result

    def test_difference_from_reference_label(self) -> None:
        """Test that 'difference from reference' label is present."""
        result = create_plain_language_summary(5, 3, 2, "Group A", "TPR", 0.15)
        assert "difference from reference" in result


class TestCreateExecutiveScorecardImportError:
    """Tests for create_executive_scorecard import error handling."""

    def test_import_error_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that meaningful import error is raised without great-tables."""

        # Skip if great_tables is actually installed (we can't easily test this)
        try:
            import great_tables  # noqa: F401

            pytest.skip("great_tables is installed, cannot test import error")
        except ImportError:
            # great_tables not installed, we can test the error
            with pytest.raises(ImportError) as exc_info:
                create_executive_scorecard(5, 3, 2, 1000, 4)
            assert "great-tables" in str(exc_info.value)
