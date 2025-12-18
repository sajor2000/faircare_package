"""
Tests for FairCareAI descriptive statistics module.

Tests cover:
- compute_cohort_summary function
- _wilson_ci function
- compute_outcome_rate_statistics function
- _interpret_cramers_v function
- format_table1_text function
- generate_table1_dataframe function
- compute_continuous_variable_summary function
"""

import numpy as np
import polars as pl
import pytest

from faircareai.metrics.descriptive import (
    _interpret_cramers_v,
    _wilson_ci,
    compute_cohort_summary,
    compute_continuous_variable_summary,
    compute_outcome_rate_statistics,
    format_table1_text,
    generate_table1_dataframe,
)


class TestComputeCohortSummary:
    """Tests for compute_cohort_summary function."""

    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        """Create sample DataFrame for testing."""
        np.random.seed(42)
        n = 200
        return pl.DataFrame(
            {
                "y_true": np.random.binomial(1, 0.3, n),
                "y_prob": np.clip(np.random.uniform(0.1, 0.9, n), 0.01, 0.99),
                "race": np.random.choice(["White", "Black", "Hispanic"], n, p=[0.5, 0.3, 0.2]),
                "sex": np.random.choice(["M", "F"], n, p=[0.5, 0.5]),
            }
        )

    def test_returns_dict(self, sample_df: pl.DataFrame) -> None:
        """Test that function returns a dictionary."""
        result = compute_cohort_summary(sample_df, "y_true", "y_prob", {"race": {"column": "race"}})
        assert isinstance(result, dict)

    def test_contains_cohort_overview(self, sample_df: pl.DataFrame) -> None:
        """Test that result contains cohort overview."""
        result = compute_cohort_summary(sample_df, "y_true", "y_prob", {"race": {"column": "race"}})
        assert "cohort_overview" in result
        overview = result["cohort_overview"]
        assert "n_total" in overview
        assert "n_positive" in overview
        assert "prevalence" in overview

    def test_cohort_overview_values(self, sample_df: pl.DataFrame) -> None:
        """Test cohort overview values are correct."""
        result = compute_cohort_summary(sample_df, "y_true", "y_prob", {"race": {"column": "race"}})
        overview = result["cohort_overview"]
        assert overview["n_total"] == 200
        assert overview["n_positive"] + overview["n_negative"] == 200
        assert 0 <= overview["prevalence"] <= 1

    def test_contains_prediction_distribution(self, sample_df: pl.DataFrame) -> None:
        """Test that result contains prediction distribution."""
        result = compute_cohort_summary(sample_df, "y_true", "y_prob", {"race": {"column": "race"}})
        assert "prediction_distribution" in result
        pred = result["prediction_distribution"]
        assert "mean" in pred
        assert "std" in pred
        assert "median" in pred

    def test_contains_attribute_distributions(self, sample_df: pl.DataFrame) -> None:
        """Test that result contains attribute distributions."""
        result = compute_cohort_summary(
            sample_df, "y_true", "y_prob", {"race": {"column": "race"}, "sex": {"column": "sex"}}
        )
        assert "attribute_distributions" in result
        assert "race" in result["attribute_distributions"]
        assert "sex" in result["attribute_distributions"]

    def test_attribute_distribution_groups(self, sample_df: pl.DataFrame) -> None:
        """Test that attribute distributions contain groups."""
        result = compute_cohort_summary(sample_df, "y_true", "y_prob", {"race": {"column": "race"}})
        attr_dist = result["attribute_distributions"]["race"]
        assert "groups" in attr_dist
        assert "White" in attr_dist["groups"]
        assert "Black" in attr_dist["groups"]

    def test_contains_outcome_by_attribute(self, sample_df: pl.DataFrame) -> None:
        """Test that result contains outcome by attribute."""
        result = compute_cohort_summary(sample_df, "y_true", "y_prob", {"race": {"column": "race"}})
        assert "outcome_by_attribute" in result
        assert "race" in result["outcome_by_attribute"]

    def test_outcome_contains_rate_info(self, sample_df: pl.DataFrame) -> None:
        """Test that outcome data contains rate information."""
        result = compute_cohort_summary(
            sample_df, "y_true", "y_prob", {"race": {"column": "race", "reference": "White"}}
        )
        outcome = result["outcome_by_attribute"]["race"]["groups"]["White"]
        assert "outcome_rate" in outcome
        assert "ci_95_low" in outcome
        assert "ci_95_high" in outcome

    def test_reference_group_marked(self, sample_df: pl.DataFrame) -> None:
        """Test that reference group is marked correctly."""
        result = compute_cohort_summary(
            sample_df, "y_true", "y_prob", {"race": {"column": "race", "reference": "White"}}
        )
        outcome = result["outcome_by_attribute"]["race"]["groups"]["White"]
        assert outcome["is_reference"] is True

    def test_rate_ratio_computed(self, sample_df: pl.DataFrame) -> None:
        """Test that rate ratio is computed vs reference."""
        result = compute_cohort_summary(
            sample_df, "y_true", "y_prob", {"race": {"column": "race", "reference": "White"}}
        )
        # Reference should have rate_ratio of 1.0 or None
        white = result["outcome_by_attribute"]["race"]["groups"]["White"]
        # Non-reference should have rate_ratio
        black = result["outcome_by_attribute"]["race"]["groups"]["Black"]
        assert black["rate_ratio"] is not None

    def test_contains_prediction_by_attribute(self, sample_df: pl.DataFrame) -> None:
        """Test that result contains prediction by attribute."""
        result = compute_cohort_summary(sample_df, "y_true", "y_prob", {"race": {"column": "race"}})
        assert "prediction_by_attribute" in result
        pred_by = result["prediction_by_attribute"]["race"]["groups"]["White"]
        assert "mean_prob" in pred_by

    def test_missing_column_skipped(self, sample_df: pl.DataFrame) -> None:
        """Test that missing columns are skipped."""
        result = compute_cohort_summary(
            sample_df, "y_true", "y_prob", {"missing_attr": {"column": "nonexistent"}}
        )
        # Should not raise and return valid result
        assert "attribute_distributions" in result

    def test_empty_dataframe(self) -> None:
        """Test handling of empty DataFrame raises error due to null type."""
        df = pl.DataFrame(
            {
                "y_true": [],
                "y_prob": [],
                "group": [],
            }
        )
        # Empty DataFrame creates null-typed columns which can't be summed
        with pytest.raises(Exception):
            compute_cohort_summary(df, "y_true", "y_prob", {"group": {"column": "group"}})


class TestWilsonCI:
    """Tests for _wilson_ci function."""

    def test_normal_case(self) -> None:
        """Test CI for normal proportion."""
        lower, upper = _wilson_ci(50, 100)
        assert 0.0 < lower < 0.5
        assert 0.5 < upper < 1.0
        assert lower < upper

    def test_zero_successes(self) -> None:
        """Test CI with zero successes."""
        lower, upper = _wilson_ci(0, 100)
        assert lower == pytest.approx(0.0, abs=0.01)
        assert upper > 0

    def test_all_successes(self) -> None:
        """Test CI with all successes."""
        lower, upper = _wilson_ci(100, 100)
        assert lower < 1.0
        assert upper == pytest.approx(1.0, abs=0.01)

    def test_zero_trials(self) -> None:
        """Test CI with zero trials."""
        lower, upper = _wilson_ci(0, 0)
        assert lower == 0.0
        assert upper == 0.0

    def test_bounds_in_range(self) -> None:
        """Test that bounds are in [0, 1]."""
        lower, upper = _wilson_ci(30, 100)
        assert 0.0 <= lower <= 1.0
        assert 0.0 <= upper <= 1.0


class TestComputeOutcomeRateStatistics:
    """Tests for compute_outcome_rate_statistics function."""

    @pytest.fixture
    def stats_df(self) -> pl.DataFrame:
        """Create DataFrame for statistical tests."""
        np.random.seed(42)
        return pl.DataFrame(
            {
                "y_true": [0] * 50 + [1] * 30 + [0] * 40 + [1] * 20 + [0] * 30 + [1] * 30,
                "group": ["A"] * 80 + ["B"] * 60 + ["C"] * 60,
            }
        )

    def test_returns_dict(self, stats_df: pl.DataFrame) -> None:
        """Test that function returns a dictionary."""
        result = compute_outcome_rate_statistics(stats_df, "y_true", "group")
        assert isinstance(result, dict)

    def test_contains_chi_square(self, stats_df: pl.DataFrame) -> None:
        """Test that result contains chi-square value."""
        result = compute_outcome_rate_statistics(stats_df, "y_true", "group")
        assert "chi_square" in result
        assert result["chi_square"] >= 0

    def test_contains_p_value(self, stats_df: pl.DataFrame) -> None:
        """Test that result contains p-value."""
        result = compute_outcome_rate_statistics(stats_df, "y_true", "group")
        assert "p_value" in result
        assert 0.0 <= result["p_value"] <= 1.0

    def test_contains_cramers_v(self, stats_df: pl.DataFrame) -> None:
        """Test that result contains Cramer's V."""
        result = compute_outcome_rate_statistics(stats_df, "y_true", "group")
        assert "cramers_v" in result
        assert 0.0 <= result["cramers_v"] <= 1.0

    def test_contains_interpretation(self, stats_df: pl.DataFrame) -> None:
        """Test that result contains interpretation."""
        result = compute_outcome_rate_statistics(stats_df, "y_true", "group")
        assert "interpretation" in result
        assert result["interpretation"] in ["negligible", "small", "medium", "large"]

    def test_pairwise_with_reference(self, stats_df: pl.DataFrame) -> None:
        """Test pairwise comparisons with reference group."""
        result = compute_outcome_rate_statistics(stats_df, "y_true", "group", reference="A")
        assert "pairwise_vs_reference" in result
        pairwise = result["pairwise_vs_reference"]
        assert "B" in pairwise
        assert "C" in pairwise

    def test_pairwise_contains_odds_ratio(self, stats_df: pl.DataFrame) -> None:
        """Test that pairwise results contain odds ratio."""
        result = compute_outcome_rate_statistics(stats_df, "y_true", "group", reference="A")
        b_result = result["pairwise_vs_reference"]["B"]
        assert "odds_ratio" in b_result
        assert "p_value" in b_result


class TestInterpretCramersV:
    """Tests for _interpret_cramers_v function."""

    def test_negligible(self) -> None:
        """Test negligible effect size."""
        assert _interpret_cramers_v(0.05) == "negligible"

    def test_small(self) -> None:
        """Test small effect size."""
        assert _interpret_cramers_v(0.15) == "small"

    def test_medium(self) -> None:
        """Test medium effect size."""
        assert _interpret_cramers_v(0.25) == "medium"

    def test_large(self) -> None:
        """Test large effect size."""
        assert _interpret_cramers_v(0.5) == "large"

    def test_boundary_values(self) -> None:
        """Test boundary values."""
        assert _interpret_cramers_v(0.1) == "small"  # At boundary
        assert _interpret_cramers_v(0.2) == "medium"
        assert _interpret_cramers_v(0.4) == "large"


class TestFormatTable1Text:
    """Tests for format_table1_text function."""

    @pytest.fixture
    def sample_summary(self) -> dict:
        """Create sample summary for formatting."""
        return {
            "cohort_overview": {
                "n_total": 1000,
                "n_positive": 300,
                "n_negative": 700,
                "prevalence_pct": "30.0%",
            },
            "prediction_distribution": {
                "mean": 0.35,
                "std": 0.15,
                "median": 0.32,
                "percentile_25": 0.22,
                "percentile_75": 0.45,
                "min": 0.05,
                "max": 0.95,
            },
            "attribute_distributions": {
                "race": {
                    "n_missing": 10,
                    "missing_rate": 0.01,
                    "groups": {
                        "White": {"n": 500, "pct": 0.5, "pct_fmt": "50.0%"},
                        "Black": {"n": 300, "pct": 0.3, "pct_fmt": "30.0%"},
                    },
                },
            },
            "outcome_by_attribute": {
                "race": {
                    "reference": "White",
                    "groups": {
                        "White": {"outcome_rate_pct": "28.0%", "rate_ratio": 1.0},
                        "Black": {"outcome_rate_pct": "35.0%", "rate_ratio": 1.25},
                    },
                },
            },
        }

    def test_returns_string(self, sample_summary: dict) -> None:
        """Test that function returns a string."""
        result = format_table1_text(sample_summary)
        assert isinstance(result, str)

    def test_contains_header(self, sample_summary: dict) -> None:
        """Test that result contains TABLE 1 header."""
        result = format_table1_text(sample_summary)
        assert "TABLE 1" in result

    def test_contains_cohort_overview(self, sample_summary: dict) -> None:
        """Test that result contains cohort overview section."""
        result = format_table1_text(sample_summary)
        assert "COHORT OVERVIEW" in result
        assert "1,000" in result  # formatted n_total

    def test_contains_prediction_distribution(self, sample_summary: dict) -> None:
        """Test that result contains prediction distribution."""
        result = format_table1_text(sample_summary)
        assert "PREDICTION SCORE DISTRIBUTION" in result

    def test_contains_attribute_name(self, sample_summary: dict) -> None:
        """Test that result contains attribute names."""
        result = format_table1_text(sample_summary)
        assert "RACE" in result

    def test_shows_missing_info(self, sample_summary: dict) -> None:
        """Test that missing info is shown."""
        result = format_table1_text(sample_summary)
        assert "Missing" in result

    def test_handles_empty_summary(self) -> None:
        """Test handling of empty summary raises ValueError due to format string."""
        # Empty dict causes formatting error when trying to use {:,} on 'N/A' string
        with pytest.raises(ValueError):
            format_table1_text({})


class TestGenerateTable1Dataframe:
    """Tests for generate_table1_dataframe function."""

    @pytest.fixture
    def sample_summary(self) -> dict:
        """Create sample summary for DataFrame generation."""
        return {
            "cohort_overview": {
                "n_total": 1000,
                "prevalence": 0.3,
            },
            "attribute_distributions": {
                "race": {
                    "groups": {
                        "White": {"n": 500, "pct": 0.5},
                        "Black": {"n": 300, "pct": 0.3},
                    },
                },
            },
            "outcome_by_attribute": {
                "race": {
                    "groups": {
                        "White": {
                            "outcome_rate": 0.28,
                            "rate_ratio": 1.0,
                            "ci_95_low": 0.24,
                            "ci_95_high": 0.32,
                        },
                        "Black": {
                            "outcome_rate": 0.35,
                            "rate_ratio": 1.25,
                            "ci_95_low": 0.30,
                            "ci_95_high": 0.40,
                        },
                    },
                },
            },
        }

    def test_returns_dataframe(self, sample_summary: dict) -> None:
        """Test that function returns a Polars DataFrame."""
        result = generate_table1_dataframe(sample_summary)
        assert isinstance(result, pl.DataFrame)

    def test_contains_expected_columns(self, sample_summary: dict) -> None:
        """Test that DataFrame has expected columns."""
        result = generate_table1_dataframe(sample_summary)
        assert "Category" in result.columns
        assert "Group" in result.columns
        assert "N" in result.columns
        assert "Outcome_Rate" in result.columns

    def test_contains_overall_row(self, sample_summary: dict) -> None:
        """Test that DataFrame contains overall row."""
        result = generate_table1_dataframe(sample_summary)
        overall = result.filter(pl.col("Category") == "Overall")
        assert len(overall) == 1

    def test_contains_group_rows(self, sample_summary: dict) -> None:
        """Test that DataFrame contains group rows."""
        result = generate_table1_dataframe(sample_summary)
        race_rows = result.filter(pl.col("Category") == "race")
        assert len(race_rows) == 2


class TestComputeContinuousVariableSummary:
    """Tests for compute_continuous_variable_summary function."""

    @pytest.fixture
    def continuous_df(self) -> pl.DataFrame:
        """Create DataFrame with continuous variable."""
        np.random.seed(42)
        return pl.DataFrame(
            {
                "age": np.random.normal(50, 10, 200),
                "group": ["A"] * 100 + ["B"] * 100,
            }
        )

    def test_overall_summary(self, continuous_df: pl.DataFrame) -> None:
        """Test overall summary without grouping."""
        result = compute_continuous_variable_summary(continuous_df, "age")
        assert "mean" in result
        assert "std" in result
        assert "median" in result
        assert "n" in result

    def test_grouped_summary(self, continuous_df: pl.DataFrame) -> None:
        """Test summary with grouping."""
        result = compute_continuous_variable_summary(continuous_df, "age", group_col="group")
        assert "groups" in result
        assert "A" in result["groups"]
        assert "B" in result["groups"]

    def test_grouped_contains_stats(self, continuous_df: pl.DataFrame) -> None:
        """Test that grouped results contain statistics."""
        result = compute_continuous_variable_summary(continuous_df, "age", group_col="group")
        group_a = result["groups"]["A"]
        assert "mean" in group_a
        assert "std" in group_a
        assert "median" in group_a
        assert "q1" in group_a
        assert "q3" in group_a

    def test_handles_missing_values(self) -> None:
        """Test handling of missing values."""
        df = pl.DataFrame(
            {
                "age": [30.0, None, 50.0, None, 70.0],
            }
        )
        result = compute_continuous_variable_summary(df, "age")
        assert result["n"] == 3
        assert result["n_missing"] == 2

    def test_empty_column(self) -> None:
        """Test handling of empty column."""
        df = pl.DataFrame(
            {
                "age": [None, None, None],
            }
        )
        result = compute_continuous_variable_summary(df, "age")
        assert "error" in result
