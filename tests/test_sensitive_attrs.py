"""
Tests for FairCareAI sensitive attribute auto-detection module.

Tests cover:
- suggest_sensitive_attributes function
- display_suggestions function
- validate_attribute function
- get_reference_group function
- SUGGESTED_PATTERNS constant
"""

import polars as pl
import pytest

from faircareai.data.sensitive_attrs import (
    SUGGESTED_PATTERNS,
    display_suggestions,
    get_reference_group,
    suggest_sensitive_attributes,
    validate_attribute,
)


class TestSuggestedPatterns:
    """Tests for SUGGESTED_PATTERNS constant."""

    def test_contains_race(self) -> None:
        """Test that race patterns are defined."""
        assert "race" in SUGGESTED_PATTERNS
        assert "patterns" in SUGGESTED_PATTERNS["race"]
        assert "race" in SUGGESTED_PATTERNS["race"]["patterns"]

    def test_contains_sex(self) -> None:
        """Test that sex patterns are defined."""
        assert "sex" in SUGGESTED_PATTERNS
        assert "patterns" in SUGGESTED_PATTERNS["sex"]
        assert "sex" in SUGGESTED_PATTERNS["sex"]["patterns"]

    def test_contains_age_group(self) -> None:
        """Test that age_group patterns are defined."""
        assert "age_group" in SUGGESTED_PATTERNS

    def test_contains_insurance(self) -> None:
        """Test that insurance patterns are defined."""
        assert "insurance" in SUGGESTED_PATTERNS

    def test_contains_language(self) -> None:
        """Test that language patterns are defined."""
        assert "language" in SUGGESTED_PATTERNS

    def test_contains_disability(self) -> None:
        """Test that disability patterns are defined."""
        assert "disability" in SUGGESTED_PATTERNS

    def test_all_have_required_keys(self) -> None:
        """Test that all patterns have required keys."""
        for name, config in SUGGESTED_PATTERNS.items():
            assert "patterns" in config, f"{name} missing patterns"
            assert "suggested_reference" in config, f"{name} missing suggested_reference"
            assert "clinical_justification" in config, f"{name} missing clinical_justification"

    def test_race_suggested_reference(self) -> None:
        """Test that race has White as suggested reference."""
        assert SUGGESTED_PATTERNS["race"]["suggested_reference"] == "White"

    def test_sex_suggested_reference(self) -> None:
        """Test that sex has Male as suggested reference."""
        assert SUGGESTED_PATTERNS["sex"]["suggested_reference"] == "Male"


class TestSuggestSensitiveAttributes:
    """Tests for suggest_sensitive_attributes function."""

    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        """Create sample DataFrame with common columns."""
        return pl.DataFrame(
            {
                "patient_id": [1, 2, 3, 4, 5],
                "race": ["White", "Black", "Hispanic", "White", "Asian"],
                "sex": ["Male", "Female", "Male", "Female", "Male"],
                "age_group": ["18-30", "31-45", "46-60", "61-75", "75+"],
                "y_true": [0, 1, 0, 1, 0],
                "y_prob": [0.2, 0.8, 0.3, 0.7, 0.4],
            }
        )

    @pytest.fixture
    def df_with_insurance(self) -> pl.DataFrame:
        """Create DataFrame with insurance column."""
        return pl.DataFrame(
            {
                "patient_id": [1, 2, 3],
                "insurance": ["Commercial", "Medicaid", "Medicare"],
                "y_true": [0, 1, 0],
            }
        )

    def test_returns_list(self, sample_df: pl.DataFrame) -> None:
        """Test that function returns a list."""
        result = suggest_sensitive_attributes(sample_df)
        assert isinstance(result, list)

    def test_detects_race(self, sample_df: pl.DataFrame) -> None:
        """Test that race column is detected."""
        result = suggest_sensitive_attributes(sample_df)
        names = [s["suggested_name"] for s in result]
        assert "race" in names

    def test_detects_sex(self, sample_df: pl.DataFrame) -> None:
        """Test that sex column is detected."""
        result = suggest_sensitive_attributes(sample_df)
        names = [s["suggested_name"] for s in result]
        assert "sex" in names

    def test_detects_age_group(self, sample_df: pl.DataFrame) -> None:
        """Test that age_group column is detected."""
        result = suggest_sensitive_attributes(sample_df)
        names = [s["suggested_name"] for s in result]
        assert "age_group" in names

    def test_detects_insurance(self, df_with_insurance: pl.DataFrame) -> None:
        """Test that insurance column is detected."""
        result = suggest_sensitive_attributes(df_with_insurance)
        names = [s["suggested_name"] for s in result]
        assert "insurance" in names

    def test_suggestion_has_required_keys(self, sample_df: pl.DataFrame) -> None:
        """Test that suggestions have all required keys."""
        result = suggest_sensitive_attributes(sample_df)
        required_keys = [
            "suggested_name",
            "detected_column",
            "unique_values",
            "n_unique",
            "missing_rate",
            "suggested_reference",
            "clinical_justification",
            "accepted",
        ]
        for suggestion in result:
            for key in required_keys:
                assert key in suggestion, f"Missing key: {key}"

    def test_accepted_always_false(self, sample_df: pl.DataFrame) -> None:
        """Test that accepted is always False by default."""
        result = suggest_sensitive_attributes(sample_df)
        for suggestion in result:
            assert suggestion["accepted"] is False

    def test_unique_values_limited(self, sample_df: pl.DataFrame) -> None:
        """Test that unique values are limited to first 10."""
        result = suggest_sensitive_attributes(sample_df)
        for suggestion in result:
            assert len(suggestion["unique_values"]) <= 10

    def test_missing_rate_computed(self, sample_df: pl.DataFrame) -> None:
        """Test that missing rate is computed."""
        result = suggest_sensitive_attributes(sample_df)
        for suggestion in result:
            assert 0.0 <= suggestion["missing_rate"] <= 1.0

    def test_no_suggestions_for_unrelated_columns(self) -> None:
        """Test that unrelated columns are not suggested."""
        df = pl.DataFrame(
            {
                "patient_id": [1, 2, 3],
                "lab_value": [1.5, 2.3, 3.1],
                "diagnosis": ["A", "B", "C"],
            }
        )
        result = suggest_sensitive_attributes(df)
        assert len(result) == 0

    def test_case_insensitive_matching(self) -> None:
        """Test that column matching is case-insensitive."""
        df = pl.DataFrame(
            {
                "RACE": ["White", "Black"],
                "SEX": ["Male", "Female"],
            }
        )
        result = suggest_sensitive_attributes(df)
        names = [s["suggested_name"] for s in result]
        assert "race" in names
        assert "sex" in names

    def test_preserves_actual_column_name(self) -> None:
        """Test that actual column name is preserved."""
        df = pl.DataFrame(
            {
                "RACE": ["White", "Black"],
            }
        )
        result = suggest_sensitive_attributes(df)
        assert result[0]["detected_column"] == "RACE"

    def test_handles_missing_values(self) -> None:
        """Test handling of columns with missing values."""
        df = pl.DataFrame(
            {
                "race": ["White", None, "Black", None, "Hispanic"],
            }
        )
        result = suggest_sensitive_attributes(df)
        assert result[0]["missing_rate"] == pytest.approx(0.4, rel=0.01)

    def test_alternative_patterns_matched(self) -> None:
        """Test that alternative column patterns are matched."""
        df = pl.DataFrame(
            {
                "patient_race": ["White", "Black"],
                "patient_sex": ["Male", "Female"],
            }
        )
        result = suggest_sensitive_attributes(df)
        names = [s["suggested_name"] for s in result]
        assert "race" in names
        assert "sex" in names


class TestDisplaySuggestions:
    """Tests for display_suggestions function."""

    @pytest.fixture
    def sample_suggestions(self) -> list[dict]:
        """Create sample suggestions for display."""
        return [
            {
                "suggested_name": "race",
                "detected_column": "race",
                "unique_values": ["White", "Black", "Hispanic", "Asian"],
                "n_unique": 4,
                "missing_rate": 0.05,
                "suggested_reference": "White",
                "clinical_justification": "Required for CMS health equity monitoring.",
                "accepted": False,
            },
            {
                "suggested_name": "sex",
                "detected_column": "sex",
                "unique_values": ["Male", "Female"],
                "n_unique": 2,
                "missing_rate": 0.0,
                "suggested_reference": "Male",
                "clinical_justification": "Important for sex-based disparities.",
                "accepted": False,
            },
        ]

    def test_returns_string(self, sample_suggestions: list[dict]) -> None:
        """Test that function returns a string."""
        result = display_suggestions(sample_suggestions)
        assert isinstance(result, str)

    def test_contains_header(self, sample_suggestions: list[dict]) -> None:
        """Test that output contains header."""
        result = display_suggestions(sample_suggestions)
        assert "SUGGESTED SENSITIVE ATTRIBUTES" in result

    def test_contains_attribute_names(self, sample_suggestions: list[dict]) -> None:
        """Test that output contains attribute names."""
        result = display_suggestions(sample_suggestions)
        assert "RACE" in result
        assert "SEX" in result

    def test_contains_column_names(self, sample_suggestions: list[dict]) -> None:
        """Test that output contains column names."""
        result = display_suggestions(sample_suggestions)
        assert "Column: race" in result
        assert "Column: sex" in result

    def test_contains_unique_count(self, sample_suggestions: list[dict]) -> None:
        """Test that output contains unique value counts."""
        result = display_suggestions(sample_suggestions)
        assert "N unique:" in result

    def test_contains_missing_rate(self, sample_suggestions: list[dict]) -> None:
        """Test that output contains missing rate."""
        result = display_suggestions(sample_suggestions)
        assert "Missing:" in result

    def test_contains_suggested_reference(self, sample_suggestions: list[dict]) -> None:
        """Test that output contains suggested reference."""
        result = display_suggestions(sample_suggestions)
        assert "Suggested reference:" in result
        assert "White" in result

    def test_contains_usage_instructions(self, sample_suggestions: list[dict]) -> None:
        """Test that output contains usage instructions."""
        result = display_suggestions(sample_suggestions)
        assert "accept_suggested_attributes" in result

    def test_empty_suggestions_message(self) -> None:
        """Test message for empty suggestions."""
        result = display_suggestions([])
        assert "NO SENSITIVE ATTRIBUTES DETECTED" in result
        assert "manually add attributes" in result

    def test_values_preview_truncated(self) -> None:
        """Test that values preview is truncated for long lists."""
        suggestions = [
            {
                "suggested_name": "race",
                "detected_column": "race",
                "unique_values": ["A", "B", "C", "D", "E", "F", "G", "H"],
                "n_unique": 8,
                "missing_rate": 0.0,
                "suggested_reference": "A",
                "clinical_justification": "Test justification.",
                "accepted": False,
            }
        ]
        result = display_suggestions(suggestions)
        assert "..." in result


class TestValidateAttribute:
    """Tests for validate_attribute function."""

    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        """Create sample DataFrame for validation."""
        return pl.DataFrame(
            {
                "race": ["White"] * 200 + ["Black"] * 150 + ["Hispanic"] * 50 + [None] * 10,
                "sex": ["Male", "Female"] * 205,
                "small_group": ["A"] * 400 + ["B"] * 5 + ["C"] * 5,
            }
        )

    def test_returns_list(self, sample_df: pl.DataFrame) -> None:
        """Test that function returns a list."""
        result = validate_attribute(sample_df, "race", "race")
        assert isinstance(result, list)

    def test_valid_attribute_no_issues(self, sample_df: pl.DataFrame) -> None:
        """Test that valid attribute has no issues."""
        result = validate_attribute(sample_df, "race", "race", reference="White")
        # Should only have the warning about null values (2.4%)
        issues = [i for i in result if "not found" in i or "Reference group" in i]
        assert len(issues) == 0

    def test_column_not_found(self, sample_df: pl.DataFrame) -> None:
        """Test that missing column is detected."""
        result = validate_attribute(sample_df, "ethnicity", "ethnicity_col")
        assert len(result) == 1
        assert "not found" in result[0]

    def test_invalid_reference_group(self, sample_df: pl.DataFrame) -> None:
        """Test that invalid reference group is detected."""
        result = validate_attribute(sample_df, "race", "race", reference="Asian")
        issues = [i for i in result if "Reference group" in i]
        assert len(issues) == 1

    def test_missing_categories(self, sample_df: pl.DataFrame) -> None:
        """Test that missing categories are detected."""
        result = validate_attribute(
            sample_df, "race", "race", categories=["White", "Black", "Asian"]
        )
        issues = [i for i in result if "Expected categories" in i]
        assert len(issues) == 1
        assert "Asian" in issues[0]

    def test_high_missing_rate_warning(self) -> None:
        """Test warning for high missing rate."""
        df = pl.DataFrame(
            {
                "race": ["White"] * 50 + [None] * 50,
            }
        )
        result = validate_attribute(df, "race", "race")
        issues = [i for i in result if "missing rate" in i.lower()]
        assert len(issues) == 1

    def test_small_group_warning(self, sample_df: pl.DataFrame) -> None:
        """Test warning for small groups."""
        result = validate_attribute(sample_df, "small", "small_group")
        issues = [i for i in result if "Small subgroups" in i]
        assert len(issues) == 1
        assert "B" in issues[0] or "C" in issues[0]

    def test_no_warning_for_adequate_groups(self, sample_df: pl.DataFrame) -> None:
        """Test no warning for adequately sized groups."""
        result = validate_attribute(sample_df, "sex", "sex")
        issues = [i for i in result if "Small subgroups" in i]
        assert len(issues) == 0


class TestGetReferenceGroup:
    """Tests for get_reference_group function."""

    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        """Create sample DataFrame for reference group tests."""
        return pl.DataFrame(
            {
                "race": ["White"] * 200 + ["Black"] * 150 + ["Hispanic"] * 50,
                "sex": ["Male"] * 200 + ["Female"] * 200,
            }
        )

    def test_returns_string(self, sample_df: pl.DataFrame) -> None:
        """Test that function returns a string."""
        result = get_reference_group(sample_df, "race")
        assert isinstance(result, str)

    def test_uses_suggested_reference(self, sample_df: pl.DataFrame) -> None:
        """Test that valid suggested reference is used."""
        result = get_reference_group(sample_df, "race", suggested_reference="Black")
        assert result == "Black"

    def test_uses_largest_when_no_suggestion(self, sample_df: pl.DataFrame) -> None:
        """Test that largest group is used when no suggestion."""
        result = get_reference_group(sample_df, "race")
        assert result == "White"

    def test_uses_largest_when_invalid_suggestion(self, sample_df: pl.DataFrame) -> None:
        """Test that largest group is used when suggestion is invalid."""
        result = get_reference_group(sample_df, "race", suggested_reference="Asian")
        assert result == "White"

    def test_equal_groups_returns_one(self, sample_df: pl.DataFrame) -> None:
        """Test that one group is returned when groups are equal."""
        result = get_reference_group(sample_df, "sex")
        assert result in ["Male", "Female"]

    def test_empty_column_raises(self) -> None:
        """Test that empty column raises ValueError."""
        df = pl.DataFrame(
            {
                "race": pl.Series([], dtype=pl.Utf8),
            }
        )
        with pytest.raises(ValueError, match="has no data"):
            get_reference_group(df, "race")

    def test_handles_null_values(self) -> None:
        """Test that null values are handled correctly."""
        df = pl.DataFrame(
            {
                "race": ["White"] * 100 + ["Black"] * 50 + [None] * 50,
            }
        )
        result = get_reference_group(df, "race")
        assert result == "White"
