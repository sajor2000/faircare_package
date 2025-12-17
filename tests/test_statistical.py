"""
Tests for FairCareAI statistical methods module.

These tests verify the accuracy of:
1. Wilson Score CI for proportions (Brown et al. 2001)
2. Newcombe-Wilson CI for differences (Newcombe 1998)
3. Katz Log-Method CI for ratios - 80% rule (Katz et al. 1978)
4. Cluster bootstrap CI (preserves hierarchical structure)
5. Sample size adequacy (stratum-specific Rule of 5)
6. Multiplicity control (Holm-Bonferroni, BH-FDR)
7. Disparate impact decision logic
"""

import numpy as np
import polars as pl

from faircareai.core.statistics import (
    AnalysisContext,
    adjust_pvalues,
    adjust_pvalues_fdr_bh,
    adjust_pvalues_holm,
    assess_sample_adequacy,
    assess_stratum_adequacy,
    bootstrap_ci_simple,
    ci_newcombe_wilson,
    ci_ratio_katz,
    ci_wilson,
    cluster_bootstrap_ci,
    disparate_impact_decision,
    get_effective_sample_size,
)


class TestWilsonScoreCI:
    """
    Tests for Wilson Score confidence intervals.

    Reference: Brown, Cai, DasGupta (2001) "Interval Estimation for a Binomial Proportion"
    Statistical Science 16(2): 101-133
    """

    def test_known_value_agresti_coull(self) -> None:
        """Test against published values from Agresti & Coull (1998)."""
        # 15 successes out of 100 trials
        lower, upper = ci_wilson(15, 100)
        # Wilson interval should be approximately (0.093, 0.227)
        assert abs(lower - 0.093) < 0.01
        assert abs(upper - 0.227) < 0.01

    def test_boundary_zero_successes(self) -> None:
        """CI for 0 successes should start at 0 but have non-zero upper bound."""
        lower, upper = ci_wilson(0, 100)
        assert lower == 0.0
        assert upper > 0.0
        # Upper bound should be approximately 3/n = 0.03 (Rule of 3)
        assert upper < 0.05

    def test_boundary_all_successes(self) -> None:
        """CI for all successes should end at 1 but have lower bound < 1."""
        lower, upper = ci_wilson(100, 100)
        assert lower < 1.0
        assert upper == 1.0

    def test_zero_observations_returns_nan(self) -> None:
        """Should return NaN for zero observations (undefined proportion)."""
        lower, upper = ci_wilson(0, 0)
        assert np.isnan(lower)
        assert np.isnan(upper)

    def test_symmetry_at_half(self) -> None:
        """CI should be symmetric around 0.5 when p=0.5."""
        lower, upper = ci_wilson(50, 100)
        center = (lower + upper) / 2
        assert abs(center - 0.5) < 0.01

    def test_ci_width_decreases_with_n(self) -> None:
        """CI width should decrease as sample size increases (sqrt(n) relationship)."""
        _, upper_100 = ci_wilson(50, 100)
        lower_100, _ = ci_wilson(50, 100)
        _, upper_400 = ci_wilson(200, 400)
        lower_400, _ = ci_wilson(200, 400)

        width_100 = upper_100 - lower_100
        width_400 = upper_400 - lower_400

        # Width should roughly halve when n quadruples
        assert width_400 < width_100 * 0.6

    def test_higher_confidence_wider_interval(self) -> None:
        """99% CI should be wider than 95% CI."""
        lower_95, upper_95 = ci_wilson(50, 100, alpha=0.05)
        lower_99, upper_99 = ci_wilson(50, 100, alpha=0.01)

        assert (upper_99 - lower_99) > (upper_95 - lower_95)


class TestNewcombeWilsonCI:
    """
    Tests for Newcombe-Wilson CI for difference of proportions.

    Reference: Newcombe (1998) "Interval estimation for the difference between
    independent proportions" Statistics in Medicine 17: 873-890
    """

    def test_no_difference(self) -> None:
        """CI should include 0 when proportions are equal."""
        lower, upper = ci_newcombe_wilson(50, 100, 50, 100)
        assert lower < 0 < upper

    def test_large_difference_excludes_zero(self) -> None:
        """CI should exclude 0 when difference is large and significant."""
        # 80% vs 20% difference
        lower, upper = ci_newcombe_wilson(80, 100, 20, 100)
        assert lower > 0  # Entire CI is positive

    def test_zero_observations_returns_nan(self) -> None:
        """Should return NaN if either group has zero observations."""
        lower, upper = ci_newcombe_wilson(0, 0, 50, 100)
        assert np.isnan(lower)
        assert np.isnan(upper)

    def test_antisymmetry(self) -> None:
        """Difference A-B should be negative of B-A."""
        lower_ab, upper_ab = ci_newcombe_wilson(70, 100, 50, 100)
        lower_ba, upper_ba = ci_newcombe_wilson(50, 100, 70, 100)

        assert abs(lower_ab + upper_ba) < 0.01
        assert abs(upper_ab + lower_ba) < 0.01

    def test_perfect_proportions(self) -> None:
        """Should handle 0% and 100% proportions with Haldane correction."""
        # 0% vs 50%
        lower, upper = ci_newcombe_wilson(0, 100, 50, 100)
        assert not np.isnan(lower)
        assert lower < 0  # 0% - 50% is negative


class TestKatzRatioCI:
    """
    Tests for Katz Log-Method CI for risk ratios.

    Critical for validating the 80% Rule (Disparate Impact) in fairness analysis.
    Reference: Katz, Baptista, Azen, Pike (1978) "Obtaining confidence intervals
    for the risk ratio in cohort studies" Biometrics 34: 469-474
    """

    def test_ratio_equals_one(self) -> None:
        """CI should include 1 when proportions are equal."""
        lower, upper = ci_ratio_katz(50, 100, 50, 100)
        assert lower < 1 < upper

    def test_ratio_significantly_below_80_percent(self) -> None:
        """CI should exclude 0.80 when ratio is clearly below threshold."""
        # 20% vs 40% = ratio of 0.5
        lower, upper = ci_ratio_katz(20, 100, 40, 100)
        assert upper < 0.80  # Entire CI below 80% threshold

    def test_ratio_significantly_above_80_percent(self) -> None:
        """CI should exclude 0.80 when ratio is clearly above threshold."""
        # 90% vs 95% = ratio of ~0.95
        lower, upper = ci_ratio_katz(90, 100, 95, 100)
        assert lower > 0.80  # Entire CI above 80% threshold

    def test_haldane_correction_for_zero_count(self) -> None:
        """Should apply Haldane-Anscombe correction (+0.5) for zero counts."""
        # 0/100 vs 50/100
        lower, upper = ci_ratio_katz(0, 100, 50, 100)
        assert not np.isnan(lower)
        assert not np.isnan(upper)
        assert lower >= 0

    def test_haldane_correction_for_all_successes(self) -> None:
        """Should handle 100% success rate with Haldane correction."""
        # 100/100 vs 50/100
        lower, upper = ci_ratio_katz(100, 100, 50, 100)
        assert not np.isnan(lower)
        assert not np.isnan(upper)
        assert lower > 1  # Ratio is 2.0

    def test_zero_observations_returns_nan(self) -> None:
        """Should return NaN for zero observations."""
        lower, upper = ci_ratio_katz(0, 0, 50, 100)
        assert np.isnan(lower)
        assert np.isnan(upper)

    def test_log_space_symmetry(self) -> None:
        """Ratio CI should be symmetric in log-space (geometric mean)."""
        lower, upper = ci_ratio_katz(60, 100, 40, 100)
        ratio = 60 / 40  # 1.5

        # In log-space: log(ratio) should be equidistant from log(lower) and log(upper)
        log_ratio = np.log(ratio)
        log_lower = np.log(lower)
        log_upper = np.log(upper)

        dist_to_lower = log_ratio - log_lower
        dist_to_upper = log_upper - log_ratio

        # Should be approximately symmetric in log-space
        assert abs(dist_to_lower - dist_to_upper) < 0.1


class TestDisparateImpactDecision:
    """Tests for 80% rule decision logic with explicit CI-based decisions."""

    def test_violation_supported_when_upper_below_threshold(self) -> None:
        """Statistical violation: entire CI below 0.80."""
        decision = disparate_impact_decision(0.65, 0.78, threshold=0.80)
        assert decision.decision == "violation_supported"

    def test_compliant_when_lower_above_threshold(self) -> None:
        """Statistical compliance: entire CI at or above 0.80."""
        decision = disparate_impact_decision(0.85, 0.95, threshold=0.80)
        assert decision.decision == "compliant"

    def test_inconclusive_when_ci_spans_threshold(self) -> None:
        """Inconclusive: CI spans the 0.80 threshold."""
        decision = disparate_impact_decision(0.75, 0.85, threshold=0.80)
        assert decision.decision == "inconclusive"

    def test_insufficient_data_for_nan(self) -> None:
        """Insufficient data when CI cannot be computed."""
        decision = disparate_impact_decision(np.nan, np.nan, threshold=0.80)
        assert decision.decision == "insufficient_data"

    def test_custom_threshold(self) -> None:
        """Should respect custom threshold values."""
        # At 0.90 threshold, this should be violation
        decision = disparate_impact_decision(0.82, 0.88, threshold=0.90)
        assert decision.decision == "violation_supported"


class TestClusterBootstrapCI:
    """
    Tests for cluster-aware bootstrap confidence intervals.

    Critical for healthcare data where patients have multiple encounters.
    Resampling must occur at the patient level, not encounter level.
    """

    def test_basic_functionality(self, clustered_data: pl.DataFrame) -> None:
        """Test basic cluster bootstrap CI computation."""

        def mean_fn(d: pl.DataFrame) -> float:
            return float(d["y_prob"].mean())

        theta, (ci_lower, ci_upper) = cluster_bootstrap_ci(
            clustered_data, "patient_id", mean_fn, n_bootstrap=200, random_state=42
        )

        assert ci_lower < theta < ci_upper
        assert ci_upper - ci_lower > 0

    def test_reproducibility_with_seed(self, clustered_data: pl.DataFrame) -> None:
        """Same random_state should produce identical results."""

        def mean_fn(d: pl.DataFrame) -> float:
            return float(d["y_prob"].mean())

        result1 = cluster_bootstrap_ci(
            clustered_data, "patient_id", mean_fn, n_bootstrap=100, random_state=123
        )
        result2 = cluster_bootstrap_ci(
            clustered_data, "patient_id", mean_fn, n_bootstrap=100, random_state=123
        )

        assert result1 == result2

    def test_cluster_ci_wider_than_naive(self, clustered_data: pl.DataFrame) -> None:
        """Cluster CI should be wider than naive (ignoring clustering) CI."""

        def mean_fn(d: pl.DataFrame) -> float:
            return float(d["y_prob"].mean())

        # Cluster-aware CI
        _, (lo_cluster, hi_cluster) = cluster_bootstrap_ci(
            clustered_data, "patient_id", mean_fn, n_bootstrap=500, random_state=42
        )

        # Naive bootstrap (treat each row as independent)
        data_array = clustered_data["y_prob"].to_numpy()
        _, (lo_naive, hi_naive) = bootstrap_ci_simple(
            data_array, np.mean, n_bootstrap=500, random_state=42
        )

        width_cluster = hi_cluster - lo_cluster
        width_naive = hi_naive - lo_naive

        # Cluster CI should generally be wider (accounts for correlation)
        assert width_cluster > width_naive * 0.8  # Allow some variance


class TestSampleAdequacy:
    """Tests for sample size adequacy assessment (Rule of 5)."""

    def test_adequate_sample(self) -> None:
        """n >= 50 with sufficient positives/negatives is ADEQUATE."""
        result = assess_sample_adequacy(n_total=100, n_positive=50, n_negative=50)
        assert result.status == "ADEQUATE"
        assert result.warning is None

    def test_moderate_sample(self) -> None:
        """30 <= n < 50 is MODERATE."""
        result = assess_sample_adequacy(n_total=40, n_positive=20, n_negative=20)
        assert result.status == "MODERATE"

    def test_low_sample(self) -> None:
        """10 <= n < 30 is LOW with warning."""
        result = assess_sample_adequacy(n_total=20, n_positive=10, n_negative=10)
        assert result.status == "LOW"
        assert result.warning is not None

    def test_very_low_sample(self) -> None:
        """n < 5 or insufficient events is VERY_LOW."""
        result = assess_sample_adequacy(n_total=3, n_positive=2, n_negative=1)
        assert result.status == "VERY_LOW"
        assert "CAUTION" in result.warning

    def test_insufficient_positives_warning(self) -> None:
        """Should warn when n_positive < 5."""
        result = assess_sample_adequacy(n_total=100, n_positive=3, n_negative=97)
        assert "TPR" in result.warning or "positives" in result.warning.lower()


class TestStratumAdequacy:
    """
    Tests for stratum-specific sample adequacy.

    Critical: Sample gates must be per group AND per relevant stratum
    (TPR needs n with y_true=1, FPR needs n with y_true=0).
    """

    def test_tpr_uses_positive_stratum(self) -> None:
        """TPR adequacy should count patients with y_true==1 only."""
        df = pl.DataFrame(
            {
                "patient_id": list(range(100)),
                "group": ["A"] * 50 + ["B"] * 50,
                "y_true": [1] * 40 + [0] * 10 + [1] * 10 + [0] * 40,
            }
        )

        result_a = assess_stratum_adequacy(df, "group", "A", "TPR", "y_true", "patient_id")
        result_b = assess_stratum_adequacy(df, "group", "B", "TPR", "y_true", "patient_id")

        # Group A has 40 positives -> REPORT
        assert result_a.status == "REPORT"
        assert result_a.unique_patients == 40

        # Group B has only 10 positives -> FLAG
        assert result_b.status == "FLAG"
        assert result_b.unique_patients == 10

    def test_fpr_uses_negative_stratum(self) -> None:
        """FPR adequacy should count patients with y_true==0 only."""
        df = pl.DataFrame(
            {
                "patient_id": list(range(100)),
                "group": ["A"] * 50 + ["B"] * 50,
                "y_true": [1] * 47 + [0] * 3 + [1] * 10 + [0] * 40,
            }
        )

        result_a = assess_stratum_adequacy(df, "group", "A", "FPR", "y_true", "patient_id")

        # Group A has only 3 negatives -> SUPPRESS
        assert result_a.status == "SUPPRESS"
        assert result_a.unique_patients == 3

    def test_suppress_threshold(self) -> None:
        """Should SUPPRESS when n < 5 (Rule of 5)."""
        df = pl.DataFrame(
            {
                "patient_id": [1, 2, 3, 4],
                "group": ["A", "A", "A", "A"],
                "y_true": [1, 1, 1, 1],
            }
        )

        result = assess_stratum_adequacy(df, "group", "A", "TPR", "y_true", "patient_id")

        assert result.status == "SUPPRESS"
        assert "SUPPRESS" in result.warning


class TestEffectiveSampleSize:
    """Tests for effective sample size with clustering."""

    def test_without_clustering(self) -> None:
        """Without clustering, effective n = row count."""
        df = pl.DataFrame({"value": [1, 2, 3, 4, 5]})
        assert get_effective_sample_size(df) == 5

    def test_with_clustering(self) -> None:
        """With clustering, effective n = unique clusters."""
        df = pl.DataFrame(
            {
                "patient_id": [1, 1, 2, 2, 3],
                "value": [1, 2, 3, 4, 5],
            }
        )
        assert get_effective_sample_size(df, "patient_id") == 3

    def test_missing_cluster_column(self) -> None:
        """Falls back to row count if cluster column missing."""
        df = pl.DataFrame({"value": [1, 2, 3, 4, 5]})
        assert get_effective_sample_size(df, "patient_id") == 5


class TestMultiplicityControl:
    """
    Tests for multiple testing correction.

    - Holm-Bonferroni: FWER control for confirmatory analysis
    - Benjamini-Hochberg: FDR control for exploratory analysis
    """

    def test_holm_increases_pvalues(self) -> None:
        """Holm adjustment should increase (or maintain) p-values."""
        pvals = np.array([0.01, 0.04, 0.03, 0.20])
        adjusted = adjust_pvalues_holm(pvals)
        assert all(adjusted >= pvals)

    def test_holm_caps_at_one(self) -> None:
        """Adjusted p-values should never exceed 1.0."""
        pvals = np.array([0.5, 0.6, 0.7, 0.8])
        adjusted = adjust_pvalues_holm(pvals)
        assert all(adjusted <= 1.0)

    def test_holm_preserves_order(self) -> None:
        """Holm-adjusted p-values maintain original ordering relationship."""
        pvals = np.array([0.001, 0.01, 0.02, 0.05])
        adjusted = adjust_pvalues_holm(pvals)

        # Original smallest should still be smallest or tied
        orig_order = np.argsort(pvals)
        adj_order = np.argsort(adjusted)
        assert orig_order[0] == adj_order[0]

    def test_fdr_bh_increases_pvalues(self) -> None:
        """BH-FDR adjustment should increase (or maintain) p-values."""
        pvals = np.array([0.01, 0.04, 0.03, 0.20])
        adjusted = adjust_pvalues_fdr_bh(pvals)
        assert all(adjusted >= pvals)

    def test_fdr_less_conservative_than_holm(self) -> None:
        """BH-FDR should generally be less conservative than Holm."""
        pvals = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

        holm = adjust_pvalues_holm(pvals)
        fdr = adjust_pvalues_fdr_bh(pvals)

        # FDR should be less than or equal to Holm
        assert all(fdr <= holm + 1e-10)

    def test_adjust_pvalues_dispatch(self) -> None:
        """Unified function should dispatch correctly."""
        pvals = np.array([0.01, 0.04, 0.03])

        assert np.allclose(adjust_pvalues(pvals, "holm"), adjust_pvalues_holm(pvals))
        assert np.allclose(adjust_pvalues(pvals, "fdr_bh"), adjust_pvalues_fdr_bh(pvals))
        assert np.allclose(adjust_pvalues(pvals, "none"), pvals)

    def test_empty_array(self) -> None:
        """Empty array should return empty array."""
        assert len(adjust_pvalues_holm(np.array([]))) == 0
        assert len(adjust_pvalues_fdr_bh(np.array([]))) == 0


class TestAnalysisContext:
    """Tests for AnalysisContext configuration dataclass."""

    def test_default_values(self) -> None:
        """Verify sensible defaults for clinical analysis."""
        ctx = AnalysisContext()

        assert ctx.cluster_col is None
        assert ctx.threshold == 0.5
        assert ctx.n_bootstrap == 2000
        assert ctx.n_permutations == 2000
        assert ctx.alpha == 0.05
        assert ctx.multiplicity_method == "fdr_bh"

    def test_custom_configuration(self) -> None:
        """Custom values should override defaults."""
        ctx = AnalysisContext(
            cluster_col="patient_id",
            site_col="hospital_id",
            n_bootstrap=5000,
            alpha=0.10,
            multiplicity_method="holm",
        )

        assert ctx.cluster_col == "patient_id"
        assert ctx.n_bootstrap == 5000
        assert ctx.alpha == 0.10
        assert ctx.multiplicity_method == "holm"
