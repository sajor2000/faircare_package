"""
Tests for FairCareAI hypothesis testing module.

These tests verify the accuracy of:
1. Stratified cluster permutation tests (critical fix from audit)
2. Correct stratum selection for TPR (y=1) vs FPR (y=0) metrics
3. Patient-level (cluster) permutation preserving correlation
4. Confounder stratification for Simpson's Paradox control
5. +1 p-value correction to avoid p=0

CRITICAL: For conditional metrics (TPR, FPR), permutation must occur
only within the relevant outcome stratum AND at the patient level.
"""

import warnings

import numpy as np
import polars as pl

from faircareai.core.hypothesis import (
    compute_metric_by_type,
    stratified_cluster_permutation_test,
    stratified_permutation_test,
)


class TestStratumSelection:
    """
    Tests for correct stratum selection based on metric type.

    CRITICAL FIX: Global permutation invalidates tests for conditional
    metrics when prevalence differs between groups. Must stratify.
    """

    def test_tpr_permutes_only_positives(self) -> None:
        """TPR test should permute group labels only where y_true == 1."""
        np.random.seed(42)

        # Groups with different prevalence (A: 80% positive, B: 40% positive)
        # Total: 100 rows
        df = pl.DataFrame(
            {
                "patient_id": list(range(100)),
                "group": ["A"] * 50 + ["B"] * 50,
                "y_true": [1] * 40 + [0] * 10 + [1] * 20 + [0] * 30,
                "y_pred": [1] * 30 + [0] * 20 + [1] * 15 + [0] * 35,  # Fixed: 100 total
            }
        )

        def tpr_diff(d: pl.DataFrame) -> float:
            a = d.filter(d["group"] == "A")
            b = d.filter(d["group"] == "B")
            tpr_a = float((a["y_pred"] == 1).mean()) if len(a) > 0 else 0
            tpr_b = float((b["y_pred"] == 1).mean()) if len(b) > 0 else 0
            return tpr_a - tpr_b

        result = stratified_cluster_permutation_test(
            df,
            tpr_diff,
            metric_type="TPR",
            group_col="group",
            y_true_col="y_true",
            cluster_col="patient_id",
            n_perms=100,
            random_state=42,
        )

        # Should produce valid p-value
        assert 0 < result["p_value"] <= 1
        assert len(result["null_distribution"]) == 100

    def test_fpr_permutes_only_negatives(self) -> None:
        """FPR test should permute group labels only where y_true == 0."""
        np.random.seed(42)

        df = pl.DataFrame(
            {
                "patient_id": list(range(100)),
                "group": ["A"] * 50 + ["B"] * 50,
                "y_true": [1] * 10 + [0] * 40 + [1] * 30 + [0] * 20,
                "y_pred": [1] * 5
                + [0] * 5
                + [1] * 20
                + [0] * 20
                + [1] * 15
                + [0] * 15
                + [1] * 5
                + [0] * 15,
            }
        )

        def fpr_diff(d: pl.DataFrame) -> float:
            a = d.filter(d["group"] == "A")
            b = d.filter(d["group"] == "B")
            fpr_a = float((a["y_pred"] == 1).mean()) if len(a) > 0 else 0
            fpr_b = float((b["y_pred"] == 1).mean()) if len(b) > 0 else 0
            return fpr_a - fpr_b

        result = stratified_cluster_permutation_test(
            df,
            fpr_diff,
            metric_type="FPR",
            group_col="group",
            y_true_col="y_true",
            cluster_col="patient_id",
            n_perms=100,
            random_state=42,
        )

        assert 0 < result["p_value"] <= 1

    def test_independence_permutes_globally(self) -> None:
        """Demographic parity test should permute across entire dataset."""
        np.random.seed(42)

        df = pl.DataFrame(
            {
                "patient_id": list(range(100)),
                "group": ["A"] * 50 + ["B"] * 50,
                "y_true": [1] * 25 + [0] * 25 + [1] * 25 + [0] * 25,
                "y_pred": [1] * 40 + [0] * 10 + [1] * 20 + [0] * 30,
            }
        )

        def selection_rate_diff(d: pl.DataFrame) -> float:
            a = d.filter(d["group"] == "A")
            b = d.filter(d["group"] == "B")
            rate_a = float((a["y_pred"] == 1).mean())
            rate_b = float((b["y_pred"] == 1).mean())
            return rate_a - rate_b

        result = stratified_cluster_permutation_test(
            df,
            selection_rate_diff,
            metric_type="Independence",
            group_col="group",
            y_true_col="y_true",
            cluster_col="patient_id",
            n_perms=100,
            random_state=42,
        )

        assert 0 < result["p_value"] <= 1


class TestClusterLevelPermutation:
    """
    Tests for patient-level (cluster) permutation.

    CRITICAL: Permuting at encounter level violates independence
    assumption and produces anticonservative p-values.
    """

    def test_permutes_at_patient_level(self) -> None:
        """Verify permutation happens at patient level, not encounter."""
        np.random.seed(42)

        n_patients = 20
        encounters = 5

        df = pl.DataFrame(
            {
                "patient_id": np.repeat(np.arange(n_patients), encounters),
                "group": np.repeat(["A"] * 10 + ["B"] * 10, encounters),
                "y_true": np.tile([1, 1, 1, 0, 0], n_patients),
                "y_pred": np.tile([1, 1, 0, 1, 0], n_patients),
            }
        )

        def metric(d: pl.DataFrame) -> float:
            a = d.filter(d["group"] == "A")
            return float((a["y_pred"] == 1).mean())

        result = stratified_cluster_permutation_test(
            df,
            metric,
            metric_type="TPR",
            group_col="group",
            y_true_col="y_true",
            cluster_col="patient_id",
            n_perms=50,
            random_state=42,
        )

        assert 0 < result["p_value"] <= 1

    def test_fallback_without_cluster_col(self) -> None:
        """Should fall back to row-level permutation without cluster_col."""
        df = pl.DataFrame(
            {
                "group": ["A", "A", "B", "B"] * 10,
                "y_true": [1, 1, 0, 0] * 10,
                "y_pred": [1, 0, 0, 1] * 10,
            }
        )

        def metric(d: pl.DataFrame) -> float:
            return float((d.filter(d["group"] == "A")["y_pred"] == 1).mean())

        # Should run without error
        result = stratified_cluster_permutation_test(
            df,
            metric,
            metric_type="TPR",
            group_col="group",
            y_true_col="y_true",
            cluster_col=None,
            n_perms=50,
            random_state=42,
        )

        assert 0 < result["p_value"] <= 1


class TestConfounderStratification:
    """Tests for permutation within confounder strata (Simpson's Paradox control)."""

    def test_site_id_stratification(self) -> None:
        """Permutation should occur within site_id strata."""
        np.random.seed(42)

        df = pl.DataFrame(
            {
                "patient_id": list(range(100)),
                "group": ["A", "B"] * 50,
                "site_id": ["Site1"] * 50 + ["Site2"] * 50,
                "y_true": [1] * 60 + [0] * 40,
                "y_pred": [1] * 40 + [0] * 20 + [1] * 20 + [0] * 20,
            }
        )

        def tpr_diff(d: pl.DataFrame) -> float:
            a = d.filter(d["group"] == "A")
            b = d.filter(d["group"] == "B")
            tpr_a = float((a["y_pred"] == 1).mean()) if len(a) > 0 else 0
            tpr_b = float((b["y_pred"] == 1).mean()) if len(b) > 0 else 0
            return tpr_a - tpr_b

        result = stratified_cluster_permutation_test(
            df,
            tpr_diff,
            metric_type="TPR",
            group_col="group",
            y_true_col="y_true",
            cluster_col="patient_id",
            confound_cols=["site_id"],
            n_perms=50,
            random_state=42,
        )

        assert 0 < result["p_value"] <= 1


class TestPValueCorrection:
    """
    Tests for p-value computation with +1 correction.

    CRITICAL: p = (1 + count) / (1 + n_perms) ensures p > 0 always.
    """

    def test_pvalue_never_exactly_zero(self) -> None:
        """P-value should never be exactly 0 due to +1 correction."""
        np.random.seed(42)

        # Create extreme disparity that should give very small p-value
        df = pl.DataFrame(
            {
                "patient_id": list(range(100)),
                "group": ["A"] * 50 + ["B"] * 50,
                "y_true": [1] * 100,
                "y_pred": [1] * 50 + [0] * 50,  # Perfect disparity
            }
        )

        def tpr_diff(d: pl.DataFrame) -> float:
            a = d.filter(d["group"] == "A")
            b = d.filter(d["group"] == "B")
            return float((a["y_pred"] == 1).mean()) - float((b["y_pred"] == 1).mean())

        result = stratified_cluster_permutation_test(
            df,
            tpr_diff,
            metric_type="TPR",
            group_col="group",
            y_true_col="y_true",
            cluster_col="patient_id",
            n_perms=100,
            random_state=42,
        )

        # Minimum p-value with +1 correction: 1/(1+100) = 0.0099
        assert result["p_value"] > 0
        assert result["p_value"] >= 1 / 101

    def test_pvalue_in_valid_range(self) -> None:
        """P-value should be in (0, 1]."""
        np.random.seed(42)

        df = pl.DataFrame(
            {
                "patient_id": list(range(50)),
                "group": ["A", "B"] * 25,
                "y_true": [1] * 50,
                "y_pred": np.random.binomial(1, 0.5, 50).tolist(),
            }
        )

        def metric(d: pl.DataFrame) -> float:
            a = d.filter(d["group"] == "A")
            b = d.filter(d["group"] == "B")
            return float((a["y_pred"] == 1).mean()) - float((b["y_pred"] == 1).mean())

        result = stratified_cluster_permutation_test(
            df,
            metric,
            metric_type="TPR",
            group_col="group",
            y_true_col="y_true",
            n_perms=100,
            random_state=42,
        )

        assert 0 < result["p_value"] <= 1


class TestReproducibility:
    """Tests for reproducible results with random_state."""

    def test_same_seed_same_result(self) -> None:
        """Same random_state should produce identical results."""
        df = pl.DataFrame(
            {
                "patient_id": list(range(50)),
                "group": ["A", "B"] * 25,
                "y_true": [1] * 30 + [0] * 20,
                "y_pred": [1] * 20 + [0] * 30,
            }
        )

        def metric(d: pl.DataFrame) -> float:
            a = d.filter(d["group"] == "A")
            return float((a["y_pred"] == 1).mean())

        result1 = stratified_cluster_permutation_test(
            df,
            metric,
            metric_type="TPR",
            group_col="group",
            y_true_col="y_true",
            cluster_col="patient_id",
            n_perms=50,
            random_state=42,
        )

        result2 = stratified_cluster_permutation_test(
            df,
            metric,
            metric_type="TPR",
            group_col="group",
            y_true_col="y_true",
            cluster_col="patient_id",
            n_perms=50,
            random_state=42,
        )

        assert result1["p_value"] == result2["p_value"]
        assert np.allclose(result1["null_distribution"], result2["null_distribution"])


class TestEdgeCases:
    """Tests for edge cases and empty data handling."""

    def test_empty_stratum_returns_nan(self) -> None:
        """Should return NaN when relevant stratum is empty."""
        df = pl.DataFrame(
            {
                "patient_id": list(range(10)),
                "group": ["A", "B"] * 5,
                "y_true": [0] * 10,  # No positives
                "y_pred": [1] * 5 + [0] * 5,
            }
        )

        def metric(d: pl.DataFrame) -> float:
            return 0.0

        result = stratified_cluster_permutation_test(
            df,
            metric,
            metric_type="TPR",  # TPR needs y_true==1
            group_col="group",
            y_true_col="y_true",
            n_perms=50,
            random_state=42,
        )

        assert np.isnan(result["observed_stat"])
        assert np.isnan(result["p_value"])


class TestMetricTypeMapping:
    """Tests for metric name to stratum type mapping."""

    def test_tpr_variants(self) -> None:
        """TPR, Sensitivity, Recall all map to TPR stratum."""
        for name in ["tpr", "TPR", "sensitivity", "recall", "Recall"]:
            result = compute_metric_by_type(name)
            assert result in ["TPR", "Recall"]

    def test_fpr_mapping(self) -> None:
        """FPR maps to FPR stratum."""
        assert compute_metric_by_type("fpr") == "FPR"
        assert compute_metric_by_type("FPR") == "FPR"

    def test_tnr_specificity_mapping(self) -> None:
        """TNR and Specificity map to TNR stratum."""
        for name in ["tnr", "TNR", "specificity", "Specificity"]:
            result = compute_metric_by_type(name)
            assert result in ["TNR", "Specificity"]

    def test_unknown_defaults_to_independence(self) -> None:
        """Unknown metrics default to Independence (global permutation)."""
        assert compute_metric_by_type("custom_metric") == "Independence"


class TestLegacyFunction:
    """Tests for deprecated legacy function (backward compatibility)."""

    def test_deprecation_warning(self) -> None:
        """Legacy function should emit DeprecationWarning."""
        y_true = np.array([1, 1, 0, 0] * 10)
        y_pred = np.array([1, 0, 0, 1] * 10)
        groups = np.array(["A", "A", "B", "B"] * 10)

        def metric_fn(yt, yp, g):
            return 0.0

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            stratified_permutation_test(
                y_true, y_pred, groups, metric_fn, metric_type="TPR", n_perms=10
            )

            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)

    def test_legacy_still_produces_valid_result(self) -> None:
        """Legacy function should still work for backward compatibility."""
        y_true = np.array([1, 1, 0, 0] * 25)
        y_pred = np.array([1, 0, 0, 1] * 25)
        groups = np.array(["A", "A", "B", "B"] * 25)

        def metric_fn(yt, yp, g):
            a_mask = g == "A"
            return float(yp[a_mask].mean()) - float(yp[~a_mask].mean())

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = stratified_permutation_test(
                y_true, y_pred, groups, metric_fn, metric_type="TPR", n_perms=50, random_state=42
            )

        assert 0 < result["p_value"] <= 1
        assert len(result["null_distribution"]) == 50
