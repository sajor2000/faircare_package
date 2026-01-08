"""
Tests for FairCareAI core statistical functions (statistical.py module).

Tests cover:
- Wilson score confidence interval for proportions
- Clopper-Pearson exact confidence interval
- Newcombe-Wilson CI for difference of proportions
- Sample size status determination
- Sample size warning generation
- Two-proportion z-test
"""

import pytest

from faircareai.core.statistical import (
    clopper_pearson_ci,
    get_sample_status,
    get_sample_warning,
    newcombe_wilson_ci,
    wilson_score_ci,
    z_test_two_proportions,
)

# =============================================================================
# Tests: wilson_score_ci
# =============================================================================


class TestWilsonScoreCICore:
    """Tests for Wilson score confidence interval (statistical.py)."""

    def test_normal_case(self) -> None:
        """Test Wilson CI for typical proportion (80/100 = 0.8)."""
        lower, upper = wilson_score_ci(80, 100)
        assert 0.70 < lower < 0.75
        assert 0.85 < upper < 0.90
        assert lower < 0.8 < upper  # CI contains true proportion

    def test_fifty_percent_proportion(self) -> None:
        """Test Wilson CI for 50% proportion."""
        lower, upper = wilson_score_ci(50, 100)
        assert 0.39 < lower < 0.42
        assert 0.58 < upper < 0.61
        assert lower < 0.5 < upper

    def test_zero_successes(self) -> None:
        """Test Wilson CI when all trials fail (0/100)."""
        lower, upper = wilson_score_ci(0, 100)
        assert lower == pytest.approx(0.0, abs=1e-10)  # Allow floating point tolerance
        assert 0.0 < upper < 0.05  # Upper bound should be small but positive

    def test_all_successes(self) -> None:
        """Test Wilson CI when all trials succeed (100/100)."""
        lower, upper = wilson_score_ci(100, 100)
        assert 0.95 < lower < 1.0  # Lower bound should be close to 1
        assert upper == 1.0

    def test_zero_observations(self) -> None:
        """Test Wilson CI with no observations returns full interval."""
        lower, upper = wilson_score_ci(0, 0)
        assert lower == 0.0
        assert upper == 1.0

    def test_single_success(self) -> None:
        """Test Wilson CI for 1/1 (100% success)."""
        lower, upper = wilson_score_ci(1, 1)
        assert lower > 0.0
        assert upper == 1.0

    def test_single_failure(self) -> None:
        """Test Wilson CI for 0/1 (0% success)."""
        lower, upper = wilson_score_ci(0, 1)
        assert lower == 0.0
        assert upper < 1.0

    def test_negative_successes_raises(self) -> None:
        """Test that negative successes raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            wilson_score_ci(-1, 100)

    def test_negative_trials_raises(self) -> None:
        """Test that negative trials raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            wilson_score_ci(50, -100)

    def test_successes_exceeds_trials_raises(self) -> None:
        """Test that successes > trials raises ValueError."""
        with pytest.raises(ValueError, match="cannot exceed"):
            wilson_score_ci(150, 100)

    def test_confidence_99(self) -> None:
        """Test Wilson CI with 99% confidence level (wider interval)."""
        lower_95, upper_95 = wilson_score_ci(50, 100, confidence=0.95)
        lower_99, upper_99 = wilson_score_ci(50, 100, confidence=0.99)
        # 99% CI should be wider than 95% CI
        assert lower_99 < lower_95
        assert upper_99 > upper_95

    def test_confidence_90(self) -> None:
        """Test Wilson CI with 90% confidence level (narrower interval)."""
        lower_95, upper_95 = wilson_score_ci(50, 100, confidence=0.95)
        lower_90, upper_90 = wilson_score_ci(50, 100, confidence=0.90)
        # 90% CI should be narrower than 95% CI
        assert lower_90 > lower_95
        assert upper_90 < upper_95

    def test_small_sample(self) -> None:
        """Test Wilson CI with small sample size."""
        lower, upper = wilson_score_ci(3, 10)
        assert 0.0 < lower < 0.30
        assert 0.3 < upper < 0.70
        # CI should be wide for small sample

    def test_large_sample(self) -> None:
        """Test Wilson CI with large sample size (narrower interval)."""
        lower_small, upper_small = wilson_score_ci(50, 100)
        lower_large, upper_large = wilson_score_ci(500, 1000)
        # Larger sample should have narrower CI
        assert (upper_small - lower_small) > (upper_large - lower_large)

    def test_returns_tuple(self) -> None:
        """Test that function returns a tuple of two floats."""
        result = wilson_score_ci(50, 100)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_bounds_in_valid_range(self) -> None:
        """Test that bounds are always in [0, 1]."""
        test_cases = [(0, 100), (1, 100), (99, 100), (100, 100), (50, 50)]
        for successes, trials in test_cases:
            lower, upper = wilson_score_ci(successes, trials)
            assert 0.0 <= lower <= 1.0
            assert 0.0 <= upper <= 1.0
            assert lower <= upper


# =============================================================================
# Tests: clopper_pearson_ci
# =============================================================================


class TestClopperPearsonCICore:
    """Tests for Clopper-Pearson exact confidence interval (statistical.py)."""

    def test_normal_case(self) -> None:
        """Test Clopper-Pearson CI for typical proportion."""
        lower, upper = clopper_pearson_ci(80, 100)
        assert 0.70 < lower < 0.75
        assert 0.85 < upper < 0.90
        assert lower < 0.8 < upper

    def test_zero_successes(self) -> None:
        """Test Clopper-Pearson CI when all trials fail."""
        lower, upper = clopper_pearson_ci(0, 100)
        assert lower == 0.0
        assert 0.0 < upper < 0.05

    def test_all_successes(self) -> None:
        """Test Clopper-Pearson CI when all trials succeed."""
        lower, upper = clopper_pearson_ci(100, 100)
        assert 0.95 < lower < 1.0
        assert upper == 1.0

    def test_zero_observations(self) -> None:
        """Test Clopper-Pearson CI with no observations."""
        lower, upper = clopper_pearson_ci(0, 0)
        assert lower == 0.0
        assert upper == 1.0

    def test_negative_successes_raises(self) -> None:
        """Test that negative successes raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            clopper_pearson_ci(-1, 100)

    def test_negative_trials_raises(self) -> None:
        """Test that negative trials raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            clopper_pearson_ci(50, -100)

    def test_successes_exceeds_trials_raises(self) -> None:
        """Test that successes > trials raises ValueError."""
        with pytest.raises(ValueError, match="cannot exceed"):
            clopper_pearson_ci(150, 100)

    def test_confidence_99(self) -> None:
        """Test Clopper-Pearson CI with 99% confidence."""
        lower_95, upper_95 = clopper_pearson_ci(50, 100, confidence=0.95)
        lower_99, upper_99 = clopper_pearson_ci(50, 100, confidence=0.99)
        assert lower_99 < lower_95
        assert upper_99 > upper_95

    def test_confidence_90(self) -> None:
        """Test Clopper-Pearson CI with 90% confidence."""
        lower_95, upper_95 = clopper_pearson_ci(50, 100, confidence=0.95)
        lower_90, upper_90 = clopper_pearson_ci(50, 100, confidence=0.90)
        assert lower_90 > lower_95
        assert upper_90 < upper_95

    def test_extreme_low_proportion(self) -> None:
        """Test Clopper-Pearson CI for very low proportion (1/1000)."""
        lower, upper = clopper_pearson_ci(1, 1000)
        assert lower == pytest.approx(0.0, abs=0.001)
        assert 0.0 < upper < 0.01

    def test_extreme_high_proportion(self) -> None:
        """Test Clopper-Pearson CI for very high proportion (999/1000)."""
        lower, upper = clopper_pearson_ci(999, 1000)
        assert 0.99 < lower < 1.0
        assert upper == pytest.approx(1.0, abs=0.001)

    def test_bounds_in_valid_range(self) -> None:
        """Test that bounds are always in [0, 1]."""
        test_cases = [(0, 100), (1, 100), (99, 100), (100, 100)]
        for successes, trials in test_cases:
            lower, upper = clopper_pearson_ci(successes, trials)
            assert 0.0 <= lower <= 1.0
            assert 0.0 <= upper <= 1.0
            assert lower <= upper


# =============================================================================
# Tests: newcombe_wilson_ci
# =============================================================================


class TestNewcombeWilsonCICore:
    """Tests for Newcombe-Wilson CI for difference of proportions (statistical.py)."""

    def test_difference_proportions(self) -> None:
        """Test CI for difference of two proportions (80% vs 70%)."""
        lower, upper = newcombe_wilson_ci(80, 100, 70, 100)
        # Difference is 70% - 80% = -10%
        assert -0.25 < lower < 0.0
        assert -0.05 < upper < 0.10
        # CI should contain the true difference (-0.10)
        assert lower < -0.10 < upper

    def test_equal_proportions(self) -> None:
        """Test CI when both proportions are equal."""
        lower, upper = newcombe_wilson_ci(50, 100, 50, 100)
        # Difference is 0, CI should contain 0
        assert lower < 0.0 < upper

    def test_no_variation_both_zero(self) -> None:
        """Test CI when both groups have 0% success."""
        lower, upper = newcombe_wilson_ci(0, 100, 0, 100)
        # Both are 0%, difference is 0
        assert lower <= 0.0 <= upper

    def test_no_variation_both_full(self) -> None:
        """Test CI when both groups have 100% success."""
        lower, upper = newcombe_wilson_ci(100, 100, 100, 100)
        # Both are 100%, difference is 0
        assert lower <= 0.0 <= upper

    def test_zero_trials_group1(self) -> None:
        """Test CI when first group has zero trials."""
        lower, upper = newcombe_wilson_ci(0, 0, 50, 100)
        assert lower == -1.0
        assert upper == 1.0

    def test_zero_trials_group2(self) -> None:
        """Test CI when second group has zero trials."""
        lower, upper = newcombe_wilson_ci(50, 100, 0, 0)
        assert lower == -1.0
        assert upper == 1.0

    def test_zero_trials_both(self) -> None:
        """Test CI when both groups have zero trials."""
        lower, upper = newcombe_wilson_ci(0, 0, 0, 0)
        assert lower == -1.0
        assert upper == 1.0

    def test_large_difference(self) -> None:
        """Test CI for large difference in proportions."""
        lower, upper = newcombe_wilson_ci(10, 100, 90, 100)
        # Difference is 90% - 10% = 80%
        # Allow small numerical variation between implementations
        assert 0.69 < lower < 0.80
        assert 0.80 < upper < 0.95  # Allow wider range due to CI width

    def test_negative_difference(self) -> None:
        """Test CI when group2 has lower proportion than group1."""
        lower, upper = newcombe_wilson_ci(90, 100, 10, 100)
        # Difference is 10% - 90% = -80%
        # Allow small numerical variation between implementations
        assert -0.95 < lower < -0.70  # Allow wider range due to CI width
        assert -0.80 < upper < -0.69

    def test_bounds_in_valid_range(self) -> None:
        """Test that bounds are always in [-1, 1]."""
        test_cases = [
            (0, 100, 100, 100),
            (100, 100, 0, 100),
            (50, 100, 50, 100),
        ]
        for s1, t1, s2, t2 in test_cases:
            lower, upper = newcombe_wilson_ci(s1, t1, s2, t2)
            assert -1.0 <= lower <= 1.0
            assert -1.0 <= upper <= 1.0
            assert lower <= upper

    def test_confidence_99(self) -> None:
        """Test Newcombe-Wilson CI with 99% confidence."""
        lower_95, upper_95 = newcombe_wilson_ci(80, 100, 70, 100, confidence=0.95)
        lower_99, upper_99 = newcombe_wilson_ci(80, 100, 70, 100, confidence=0.99)
        assert lower_99 < lower_95
        assert upper_99 > upper_95

    def test_returns_tuple(self) -> None:
        """Test that function returns a tuple of two floats."""
        result = newcombe_wilson_ci(50, 100, 40, 100)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)


# =============================================================================
# Tests: get_sample_status
# =============================================================================


class TestGetSampleStatusCore:
    """Tests for sample size status determination (statistical.py)."""

    def test_adequate_n_100(self) -> None:
        """Test that n >= 50 returns ADEQUATE."""
        assert get_sample_status(100) == "ADEQUATE"
        assert get_sample_status(50) == "ADEQUATE"
        assert get_sample_status(1000) == "ADEQUATE"

    def test_moderate_n_40(self) -> None:
        """Test that 30 <= n < 50 returns MODERATE."""
        assert get_sample_status(40) == "MODERATE"
        assert get_sample_status(30) == "MODERATE"
        assert get_sample_status(49) == "MODERATE"

    def test_low_n_20(self) -> None:
        """Test that 10 <= n < 30 returns LOW."""
        assert get_sample_status(20) == "LOW"
        assert get_sample_status(10) == "LOW"
        assert get_sample_status(29) == "LOW"

    def test_very_low_n_5(self) -> None:
        """Test that n < 10 returns VERY_LOW."""
        assert get_sample_status(5) == "VERY_LOW"
        assert get_sample_status(9) == "VERY_LOW"
        assert get_sample_status(0) == "VERY_LOW"
        assert get_sample_status(1) == "VERY_LOW"

    def test_low_positive_count_overrides(self) -> None:
        """Test that n_positive < 5 overrides to LOW."""
        # n=100 is normally ADEQUATE, but only 3 positives
        assert get_sample_status(100, n_positive=3) == "LOW"
        assert get_sample_status(100, n_positive=4) == "LOW"

    def test_low_negative_count_overrides(self) -> None:
        """Test that n_negative < 5 overrides to LOW."""
        # n=100 is normally ADEQUATE, but only 97 negatives -> 97 negatives is fine
        # but n_positive=97 means n_negative=3
        assert get_sample_status(100, n_positive=97) == "LOW"
        assert get_sample_status(100, n_positive=96) == "LOW"

    def test_moderate_positive_count(self) -> None:
        """Test that 5 <= n_positive < 10 returns MODERATE."""
        assert get_sample_status(100, n_positive=5) == "MODERATE"
        assert get_sample_status(100, n_positive=9) == "MODERATE"

    def test_adequate_with_sufficient_positives(self) -> None:
        """Test ADEQUATE when both n and n_positive are sufficient."""
        assert get_sample_status(100, n_positive=50) == "ADEQUATE"
        assert get_sample_status(100, n_positive=10) == "ADEQUATE"

    def test_boundary_n_50(self) -> None:
        """Test boundary at n=50."""
        assert get_sample_status(50) == "ADEQUATE"
        assert get_sample_status(49) == "MODERATE"

    def test_boundary_n_30(self) -> None:
        """Test boundary at n=30."""
        assert get_sample_status(30) == "MODERATE"
        assert get_sample_status(29) == "LOW"

    def test_boundary_n_10(self) -> None:
        """Test boundary at n=10."""
        assert get_sample_status(10) == "LOW"
        assert get_sample_status(9) == "VERY_LOW"


# =============================================================================
# Tests: get_sample_warning
# =============================================================================


class TestGetSampleWarningCore:
    """Tests for sample size warning message generation (statistical.py)."""

    def test_returns_none_adequate(self) -> None:
        """Test that ADEQUATE sample returns None (no warning)."""
        warning = get_sample_warning("Group A", 100)
        assert warning is None

    def test_returns_none_adequate_with_positives(self) -> None:
        """Test that ADEQUATE sample with positives returns None."""
        warning = get_sample_warning("Group A", 100, n_positive=50)
        assert warning is None

    def test_returns_caution_very_low(self) -> None:
        """Test that VERY_LOW sample returns CAUTION message."""
        warning = get_sample_warning("Group A", 5)
        assert warning is not None
        assert "CAUTION" in warning
        assert "Group A" in warning
        assert "n=5" in warning
        assert "highly uncertain" in warning

    def test_returns_note_low(self) -> None:
        """Test that LOW sample returns Note message."""
        warning = get_sample_warning("Group B", 20)
        assert warning is not None
        assert "Note:" in warning
        assert "Group B" in warning
        assert "n=20" in warning
        assert "caution" in warning.lower()

    def test_returns_moderate_message(self) -> None:
        """Test that MODERATE sample returns appropriate message."""
        warning = get_sample_warning("Group C", 40)
        assert warning is not None
        assert "Group C" in warning
        assert "n=40" in warning
        assert "moderate" in warning.lower()

    def test_with_precomputed_status(self) -> None:
        """Test that precomputed status is used correctly."""
        # Even with n=100, use precomputed VERY_LOW status
        warning = get_sample_warning("Group D", 100, status="VERY_LOW")
        assert warning is not None
        assert "CAUTION" in warning

    def test_with_precomputed_adequate(self) -> None:
        """Test that precomputed ADEQUATE returns None."""
        warning = get_sample_warning("Group E", 5, status="ADEQUATE")
        assert warning is None

    def test_group_name_in_message(self) -> None:
        """Test that group name appears in warning message."""
        warning = get_sample_warning("Custom Group Name", 5)
        assert warning is not None
        assert "Custom Group Name" in warning

    def test_n_positive_affects_warning(self) -> None:
        """Test that n_positive affects warning generation."""
        # n=100 but only 3 positives -> LOW
        warning = get_sample_warning("Group F", 100, n_positive=3)
        assert warning is not None
        assert "Note:" in warning


# =============================================================================
# Tests: z_test_two_proportions
# =============================================================================


class TestZTestTwoProportionsCore:
    """Tests for two-proportion z-test (statistical.py)."""

    def test_significant_difference(self) -> None:
        """Test z-test detects significant difference."""
        # 80% vs 60% with n=100 each - should be significant
        z, p_value = z_test_two_proportions(80, 100, 60, 100)
        assert abs(z) > 2.0  # Significant at p < 0.05
        assert p_value < 0.05

    def test_no_difference(self) -> None:
        """Test z-test for equal proportions."""
        # Both 50% - no difference
        z, p_value = z_test_two_proportions(50, 100, 50, 100)
        assert abs(z) < 0.1
        assert p_value > 0.9

    def test_zero_trials_group1(self) -> None:
        """Test z-test when first group has zero trials."""
        z, p_value = z_test_two_proportions(0, 0, 50, 100)
        assert z == 0.0
        assert p_value == 1.0

    def test_zero_trials_group2(self) -> None:
        """Test z-test when second group has zero trials."""
        z, p_value = z_test_two_proportions(50, 100, 0, 0)
        assert z == 0.0
        assert p_value == 1.0

    def test_zero_trials_both(self) -> None:
        """Test z-test when both groups have zero trials."""
        z, p_value = z_test_two_proportions(0, 0, 0, 0)
        assert z == 0.0
        assert p_value == 1.0

    def test_zero_division_handling(self) -> None:
        """Test z-test handles zero standard error."""
        # All successes in both groups - SE = 0
        z, p_value = z_test_two_proportions(100, 100, 100, 100)
        assert z == 0.0
        assert p_value == 1.0

    def test_all_failures_both_groups(self) -> None:
        """Test z-test when all failures in both groups."""
        z, p_value = z_test_two_proportions(0, 100, 0, 100)
        assert z == 0.0
        assert p_value == 1.0

    def test_positive_z_statistic(self) -> None:
        """Test z-test produces positive z when group2 > group1."""
        z, p_value = z_test_two_proportions(40, 100, 60, 100)
        assert z > 0  # Group 2 has higher proportion

    def test_negative_z_statistic(self) -> None:
        """Test z-test produces negative z when group2 < group1."""
        z, p_value = z_test_two_proportions(60, 100, 40, 100)
        assert z < 0  # Group 2 has lower proportion

    def test_returns_tuple(self) -> None:
        """Test that function returns tuple of two floats."""
        result = z_test_two_proportions(50, 100, 40, 100)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_p_value_bounds(self) -> None:
        """Test that p-value is always in [0, 1]."""
        test_cases = [
            (50, 100, 50, 100),
            (80, 100, 20, 100),
            (0, 100, 100, 100),
            (99, 100, 1, 100),
        ]
        for s1, t1, s2, t2 in test_cases:
            _, p_value = z_test_two_proportions(s1, t1, s2, t2)
            assert 0.0 <= p_value <= 1.0

    def test_large_sample_size(self) -> None:
        """Test z-test with large sample sizes."""
        # 51% vs 49% with large n - should detect small difference
        z, p_value = z_test_two_proportions(5100, 10000, 4900, 10000)
        assert abs(z) > 1.5  # Should detect even small differences

    def test_small_sample_size(self) -> None:
        """Test z-test with small sample sizes."""
        # 6/10 vs 4/10 - unlikely to be significant
        z, p_value = z_test_two_proportions(6, 10, 4, 10)
        assert p_value > 0.3  # Not significant with small samples

    def test_extreme_proportions(self) -> None:
        """Test z-test with extreme proportions (near 0 or 1)."""
        z, p_value = z_test_two_proportions(99, 100, 95, 100)
        assert isinstance(z, float)
        assert isinstance(p_value, float)
        assert 0.0 <= p_value <= 1.0
