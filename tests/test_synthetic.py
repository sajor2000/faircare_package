"""
Tests for FairCareAI synthetic data generation.

Tests cover:
- ICU mortality data generation
- Data shape and columns
- Value ranges and distributions
- Reproducibility with seed
- Summary statistics
"""

import numpy as np
import polars as pl
import pytest

from faircareai.data.synthetic import generate_icu_mortality_data, get_data_summary


class TestGenerateICUMortalityData:
    """Tests for ICU mortality data generator."""

    def test_returns_polars_dataframe(self) -> None:
        """Test that function returns a Polars DataFrame."""
        df = generate_icu_mortality_data(n_samples=100, seed=42)
        assert isinstance(df, pl.DataFrame)

    def test_returns_correct_shape(self) -> None:
        """Test that output has correct number of rows and columns."""
        df = generate_icu_mortality_data(n_samples=500, seed=42)
        assert df.shape == (500, 9)

    def test_column_names(self) -> None:
        """Test that all expected columns are present."""
        df = generate_icu_mortality_data(n_samples=100, seed=42)
        expected_columns = [
            "patient_id",
            "age_group",
            "sex",
            "race_ethnicity",
            "insurance",
            "language",
            "y_true",
            "y_prob",
            "y_pred",
        ]
        assert list(df.columns) == expected_columns

    def test_patient_ids_unique(self) -> None:
        """Test that patient IDs are unique."""
        df = generate_icu_mortality_data(n_samples=1000, seed=42)
        assert df["patient_id"].n_unique() == 1000

    def test_patient_ids_sequential(self) -> None:
        """Test that patient IDs start at 1 and are sequential."""
        df = generate_icu_mortality_data(n_samples=100, seed=42)
        assert df["patient_id"].min() == 1
        assert df["patient_id"].max() == 100

    def test_y_prob_in_range(self) -> None:
        """Test that y_prob values are in [0.01, 0.99]."""
        df = generate_icu_mortality_data(n_samples=2000, seed=42)
        assert df["y_prob"].min() >= 0.01
        assert df["y_prob"].max() <= 0.99

    def test_y_pred_binary(self) -> None:
        """Test that y_pred contains only 0 or 1."""
        df = generate_icu_mortality_data(n_samples=500, seed=42)
        unique_vals = set(df["y_pred"].unique().to_list())
        assert unique_vals.issubset({0, 1})

    def test_y_true_binary(self) -> None:
        """Test that y_true contains only 0 or 1."""
        df = generate_icu_mortality_data(n_samples=500, seed=42)
        unique_vals = set(df["y_true"].unique().to_list())
        assert unique_vals.issubset({0, 1})

    def test_age_group_values(self) -> None:
        """Test that age_group contains expected values."""
        df = generate_icu_mortality_data(n_samples=1000, seed=42)
        expected_values = {"18-44", "45-64", "65-79", "80+"}
        actual_values = set(df["age_group"].unique().to_list())
        assert actual_values.issubset(expected_values)

    def test_sex_values(self) -> None:
        """Test that sex contains expected values."""
        df = generate_icu_mortality_data(n_samples=1000, seed=42)
        expected_values = {"Male", "Female"}
        actual_values = set(df["sex"].unique().to_list())
        assert actual_values == expected_values

    def test_race_ethnicity_values(self) -> None:
        """Test that race_ethnicity contains expected values."""
        df = generate_icu_mortality_data(n_samples=2000, seed=42)
        expected_values = {"White", "Black", "Hispanic", "Asian", "Other"}
        actual_values = set(df["race_ethnicity"].unique().to_list())
        assert actual_values.issubset(expected_values)

    def test_insurance_values(self) -> None:
        """Test that insurance contains expected values."""
        df = generate_icu_mortality_data(n_samples=1000, seed=42)
        expected_values = {"Private", "Medicare", "Medicaid", "Uninsured"}
        actual_values = set(df["insurance"].unique().to_list())
        assert actual_values.issubset(expected_values)

    def test_language_values(self) -> None:
        """Test that language contains expected values."""
        df = generate_icu_mortality_data(n_samples=2000, seed=42)
        expected_values = {"English", "Spanish", "Other"}
        actual_values = set(df["language"].unique().to_list())
        assert actual_values.issubset(expected_values)

    def test_reproducibility_with_seed(self) -> None:
        """Test that same seed produces identical results."""
        df1 = generate_icu_mortality_data(n_samples=500, seed=123)
        df2 = generate_icu_mortality_data(n_samples=500, seed=123)
        assert df1.equals(df2)

    def test_different_seeds_produce_different_data(self) -> None:
        """Test that different seeds produce different results."""
        df1 = generate_icu_mortality_data(n_samples=500, seed=42)
        df2 = generate_icu_mortality_data(n_samples=500, seed=99)
        assert not df1.equals(df2)

    def test_disparity_strength_affects_data(self) -> None:
        """Test that disparity_strength parameter affects TPR."""
        df_low = generate_icu_mortality_data(n_samples=5000, seed=42, disparity_strength=0.01)
        df_high = generate_icu_mortality_data(n_samples=5000, seed=42, disparity_strength=0.20)
        # Data should be different
        assert not df_low.equals(df_high)

    def test_threshold_affects_predictions(self) -> None:
        """Test that threshold parameter affects y_pred."""
        df_low = generate_icu_mortality_data(n_samples=1000, seed=42, threshold=0.3)
        df_high = generate_icu_mortality_data(n_samples=1000, seed=42, threshold=0.7)
        # More predictions at lower threshold
        assert df_low["y_pred"].sum() > df_high["y_pred"].sum()

    def test_small_n_samples(self) -> None:
        """Test generation with small sample size."""
        df = generate_icu_mortality_data(n_samples=10, seed=42)
        assert len(df) == 10
        assert df.shape[1] == 9

    def test_large_n_samples(self) -> None:
        """Test generation with large sample size."""
        df = generate_icu_mortality_data(n_samples=10000, seed=42)
        assert len(df) == 10000

    def test_mortality_rate_reasonable(self) -> None:
        """Test that mortality rate is in reasonable range for ICU (~15%)."""
        df = generate_icu_mortality_data(n_samples=5000, seed=42)
        mortality_rate = df["y_true"].sum() / len(df)
        # Should be roughly around 15% (base rate), allow some variance
        assert 0.08 < mortality_rate < 0.30

    def test_y_prob_correlates_with_y_true(self) -> None:
        """Test that y_prob is higher for actual positive cases."""
        df = generate_icu_mortality_data(n_samples=2000, seed=42)
        avg_prob_positive = df.filter(pl.col("y_true") == 1)["y_prob"].mean()
        avg_prob_negative = df.filter(pl.col("y_true") == 0)["y_prob"].mean()
        # Positive cases should have higher average probability
        assert avg_prob_positive > avg_prob_negative

    def test_default_parameters(self) -> None:
        """Test generation with default parameters."""
        df = generate_icu_mortality_data()
        assert len(df) == 2000
        assert df.shape[1] == 9


class TestGetDataSummary:
    """Tests for data summary function."""

    def test_returns_dict(self) -> None:
        """Test that function returns a dictionary."""
        df = generate_icu_mortality_data(n_samples=100, seed=42)
        summary = get_data_summary(df)
        assert isinstance(summary, dict)

    def test_returns_all_fields(self) -> None:
        """Test that summary contains all expected fields."""
        df = generate_icu_mortality_data(n_samples=100, seed=42)
        summary = get_data_summary(df)
        expected_keys = {
            "n_samples",
            "n_deaths",
            "mortality_rate",
            "n_predicted_deaths",
            "prediction_rate",
            "demographics",
        }
        assert set(summary.keys()) == expected_keys

    def test_n_samples_correct(self) -> None:
        """Test that n_samples matches input."""
        df = generate_icu_mortality_data(n_samples=500, seed=42)
        summary = get_data_summary(df)
        assert summary["n_samples"] == 500

    def test_mortality_rate_calculation(self) -> None:
        """Test that mortality_rate is correctly calculated."""
        df = generate_icu_mortality_data(n_samples=1000, seed=42)
        summary = get_data_summary(df)
        expected_rate = df["y_true"].sum() / len(df)
        assert summary["mortality_rate"] == expected_rate

    def test_prediction_rate_calculation(self) -> None:
        """Test that prediction_rate is correctly calculated."""
        df = generate_icu_mortality_data(n_samples=1000, seed=42)
        summary = get_data_summary(df)
        expected_rate = df["y_pred"].sum() / len(df)
        assert summary["prediction_rate"] == expected_rate

    def test_demographics_contains_all_attributes(self) -> None:
        """Test that demographics contains all demographic columns."""
        df = generate_icu_mortality_data(n_samples=500, seed=42)
        summary = get_data_summary(df)
        expected_attrs = {"age_group", "sex", "race_ethnicity", "insurance", "language"}
        assert set(summary["demographics"].keys()) == expected_attrs

    def test_demographics_value_counts(self) -> None:
        """Test that demographics value counts are lists of dicts."""
        df = generate_icu_mortality_data(n_samples=500, seed=42)
        summary = get_data_summary(df)
        for attr, counts in summary["demographics"].items():
            assert isinstance(counts, list)
            for item in counts:
                assert isinstance(item, dict)
                assert attr in item
                assert "count" in item

    def test_n_deaths_matches_y_true_sum(self) -> None:
        """Test that n_deaths equals sum of y_true."""
        df = generate_icu_mortality_data(n_samples=1000, seed=42)
        summary = get_data_summary(df)
        assert summary["n_deaths"] == df["y_true"].sum()

    def test_n_predicted_deaths_matches_y_pred_sum(self) -> None:
        """Test that n_predicted_deaths equals sum of y_pred."""
        df = generate_icu_mortality_data(n_samples=1000, seed=42)
        summary = get_data_summary(df)
        assert summary["n_predicted_deaths"] == df["y_pred"].sum()
