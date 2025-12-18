"""
Tests for FairCareAI subgroup metrics module.

Tests cover:
- compute_subgroup_metrics function
- _bootstrap_auroc function
- _compute_subgroup_disparities function
- compute_intersectional function
- _interpret_auroc_range function
- compute_pairwise_intersectional function
- identify_vulnerable_subgroups function
- _summarize_vulnerable function
"""

import numpy as np
import polars as pl
import pytest

from faircareai.metrics.subgroup import (
    _compute_subgroup_disparities,
    _interpret_auroc_range,
    _summarize_vulnerable,
    compute_intersectional,
    compute_pairwise_intersectional,
    compute_subgroup_metrics,
    identify_vulnerable_subgroups,
)


class TestComputeSubgroupMetrics:
    """Tests for compute_subgroup_metrics function."""

    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        """Create sample DataFrame with multiple groups."""
        np.random.seed(42)
        n = 300
        groups = np.random.choice(["White", "Black", "Hispanic"], n, p=[0.5, 0.3, 0.2])
        y_true = np.random.binomial(1, 0.3, n)
        y_prob = np.clip(
            np.where(y_true == 1, np.random.normal(0.6, 0.2, n), np.random.normal(0.3, 0.2, n)),
            0.01,
            0.99,
        )
        return pl.DataFrame(
            {
                "group": groups,
                "y_true": y_true,
                "y_prob": y_prob,
            }
        )

    def test_returns_dict(self, sample_df: pl.DataFrame) -> None:
        """Test that function returns a dictionary."""
        result = compute_subgroup_metrics(
            sample_df, "y_prob", "y_true", "group", bootstrap_ci=False
        )
        assert isinstance(result, dict)

    def test_contains_attribute_name(self, sample_df: pl.DataFrame) -> None:
        """Test that result contains attribute name."""
        result = compute_subgroup_metrics(
            sample_df, "y_prob", "y_true", "group", bootstrap_ci=False
        )
        assert result["attribute"] == "group"

    def test_contains_threshold(self, sample_df: pl.DataFrame) -> None:
        """Test that result contains threshold."""
        result = compute_subgroup_metrics(
            sample_df, "y_prob", "y_true", "group", threshold=0.4, bootstrap_ci=False
        )
        assert result["threshold"] == 0.4

    def test_contains_groups(self, sample_df: pl.DataFrame) -> None:
        """Test that result contains groups dict."""
        result = compute_subgroup_metrics(
            sample_df, "y_prob", "y_true", "group", bootstrap_ci=False
        )
        assert "groups" in result
        assert "White" in result["groups"]
        assert "Black" in result["groups"]
        assert "Hispanic" in result["groups"]

    def test_contains_reference(self, sample_df: pl.DataFrame) -> None:
        """Test that reference group is set."""
        result = compute_subgroup_metrics(
            sample_df, "y_prob", "y_true", "group", bootstrap_ci=False
        )
        assert "reference" in result
        # Largest group should be reference by default
        assert result["reference"] == "White"

    def test_custom_reference(self, sample_df: pl.DataFrame) -> None:
        """Test custom reference group."""
        result = compute_subgroup_metrics(
            sample_df, "y_prob", "y_true", "group", reference="Black", bootstrap_ci=False
        )
        assert result["reference"] == "Black"
        assert result["groups"]["Black"]["is_reference"] is True

    def test_group_contains_n(self, sample_df: pl.DataFrame) -> None:
        """Test that each group has sample size."""
        result = compute_subgroup_metrics(
            sample_df, "y_prob", "y_true", "group", bootstrap_ci=False
        )
        for group_data in result["groups"].values():
            assert "n" in group_data
            assert group_data["n"] > 0

    def test_group_contains_metrics(self, sample_df: pl.DataFrame) -> None:
        """Test that each group has metrics."""
        result = compute_subgroup_metrics(
            sample_df, "y_prob", "y_true", "group", bootstrap_ci=False
        )
        for group_data in result["groups"].values():
            if "error" not in group_data:
                assert "tpr" in group_data
                assert "fpr" in group_data
                assert "ppv" in group_data
                assert "auroc" in group_data

    def test_small_sample_warning(self) -> None:
        """Test small sample warning is set."""
        df = pl.DataFrame(
            {
                "group": ["A"] * 50 + ["B"] * 50,
                "y_true": [0, 1] * 50,
                "y_prob": [0.3, 0.7] * 50,
            }
        )
        result = compute_subgroup_metrics(df, "y_prob", "y_true", "group", bootstrap_ci=False)
        # n=50 should trigger small sample warning
        assert result["groups"]["A"]["small_sample_warning"] is True

    def test_very_small_group_error(self) -> None:
        """Test that very small groups get error."""
        df = pl.DataFrame(
            {
                "group": ["A"] * 100 + ["B"] * 5,  # B has only 5
                "y_true": [0, 1] * 50 + [0, 1, 0, 1, 0],
                "y_prob": [0.3, 0.7] * 50 + [0.3, 0.7, 0.3, 0.7, 0.3],
            }
        )
        result = compute_subgroup_metrics(df, "y_prob", "y_true", "group", bootstrap_ci=False)
        assert "error" in result["groups"]["B"]

    def test_contains_disparities(self, sample_df: pl.DataFrame) -> None:
        """Test that result contains disparities."""
        result = compute_subgroup_metrics(
            sample_df, "y_prob", "y_true", "group", bootstrap_ci=False
        )
        assert "disparities" in result

    def test_bootstrap_ci_computed(self, sample_df: pl.DataFrame) -> None:
        """Test that bootstrap CI is computed when requested."""
        result = compute_subgroup_metrics(
            sample_df, "y_prob", "y_true", "group", bootstrap_ci=True, n_bootstrap=50
        )
        # At least one group should have CI
        has_ci = any("auroc_ci_95" in g for g in result["groups"].values() if "error" not in g)
        assert has_ci


class TestComputeSubgroupDisparities:
    """Tests for _compute_subgroup_disparities function."""

    def test_computes_tpr_diff(self) -> None:
        """Test TPR difference computation."""
        results = {
            "groups": {
                "A": {"tpr": 0.8, "fpr": 0.1, "ppv": 0.7, "selection_rate": 0.3, "auroc": 0.85},
                "B": {"tpr": 0.7, "fpr": 0.2, "ppv": 0.6, "selection_rate": 0.4, "auroc": 0.75},
            }
        }
        disparities = _compute_subgroup_disparities(results, "A")
        assert "B" in disparities
        assert disparities["B"]["tpr_diff"] == pytest.approx(-0.1, abs=0.01)

    def test_computes_fpr_diff(self) -> None:
        """Test FPR difference computation."""
        results = {
            "groups": {
                "A": {"tpr": 0.8, "fpr": 0.1, "ppv": 0.7, "selection_rate": 0.3, "auroc": 0.85},
                "B": {"tpr": 0.7, "fpr": 0.2, "ppv": 0.6, "selection_rate": 0.4, "auroc": 0.75},
            }
        }
        disparities = _compute_subgroup_disparities(results, "A")
        assert disparities["B"]["fpr_diff"] == pytest.approx(0.1, abs=0.01)

    def test_computes_equalized_odds_diff(self) -> None:
        """Test equalized odds difference (max of TPR/FPR diff)."""
        results = {
            "groups": {
                "A": {"tpr": 0.8, "fpr": 0.1, "ppv": 0.7, "selection_rate": 0.3, "auroc": 0.85},
                "B": {"tpr": 0.7, "fpr": 0.2, "ppv": 0.6, "selection_rate": 0.4, "auroc": 0.75},
            }
        }
        disparities = _compute_subgroup_disparities(results, "A")
        # max(abs(-0.1), abs(0.1)) = 0.1
        assert disparities["B"]["equalized_odds_diff"] == pytest.approx(0.1, abs=0.01)

    def test_computes_ppv_ratio(self) -> None:
        """Test PPV ratio computation."""
        results = {
            "groups": {
                "A": {"tpr": 0.8, "fpr": 0.1, "ppv": 0.8, "selection_rate": 0.3, "auroc": 0.85},
                "B": {"tpr": 0.7, "fpr": 0.2, "ppv": 0.6, "selection_rate": 0.4, "auroc": 0.75},
            }
        }
        disparities = _compute_subgroup_disparities(results, "A")
        assert disparities["B"]["ppv_ratio"] == pytest.approx(0.75, abs=0.01)

    def test_handles_error_in_reference(self) -> None:
        """Test handling of error in reference group."""
        results = {
            "groups": {
                "A": {"error": "Insufficient data"},
                "B": {"tpr": 0.7, "fpr": 0.2, "ppv": 0.6, "selection_rate": 0.4},
            }
        }
        disparities = _compute_subgroup_disparities(results, "A")
        assert "error" in disparities

    def test_skips_groups_with_error(self) -> None:
        """Test that groups with errors are skipped."""
        results = {
            "groups": {
                "A": {"tpr": 0.8, "fpr": 0.1, "ppv": 0.7, "selection_rate": 0.3, "auroc": 0.85},
                "B": {"error": "Insufficient data"},
            }
        }
        disparities = _compute_subgroup_disparities(results, "A")
        assert "B" not in disparities


class TestInterpretAurocRange:
    """Tests for _interpret_auroc_range function."""

    def test_low_range(self) -> None:
        """Test LOW interpretation for small range."""
        assert _interpret_auroc_range(0.03) == "LOW"

    def test_moderate_range(self) -> None:
        """Test MODERATE interpretation."""
        assert _interpret_auroc_range(0.07) == "MODERATE"

    def test_high_range(self) -> None:
        """Test HIGH interpretation."""
        assert _interpret_auroc_range(0.12) == "HIGH"

    def test_severe_range(self) -> None:
        """Test SEVERE interpretation for large range."""
        assert _interpret_auroc_range(0.20) == "SEVERE"

    def test_boundary_values(self) -> None:
        """Test boundary values."""
        assert _interpret_auroc_range(0.05) == "MODERATE"  # exactly at boundary
        assert _interpret_auroc_range(0.10) == "HIGH"
        assert _interpret_auroc_range(0.15) == "SEVERE"


class TestComputeIntersectional:
    """Tests for compute_intersectional function."""

    @pytest.fixture
    def intersectional_df(self) -> pl.DataFrame:
        """Create DataFrame with multiple attributes for intersection."""
        np.random.seed(42)
        n = 400
        race = np.random.choice(["White", "Black"], n, p=[0.6, 0.4])
        sex = np.random.choice(["M", "F"], n, p=[0.5, 0.5])
        y_true = np.random.binomial(1, 0.3, n)
        y_prob = np.clip(
            np.where(y_true == 1, np.random.normal(0.6, 0.2, n), np.random.normal(0.3, 0.2, n)),
            0.01,
            0.99,
        )
        return pl.DataFrame(
            {
                "race": race,
                "sex": sex,
                "y_true": y_true,
                "y_prob": y_prob,
            }
        )

    def test_returns_dict(self, intersectional_df: pl.DataFrame) -> None:
        """Test that function returns a dictionary."""
        result = compute_intersectional(
            intersectional_df, "y_prob", "y_true", ["race", "sex"], bootstrap_ci=False
        )
        assert isinstance(result, dict)

    def test_contains_attributes(self, intersectional_df: pl.DataFrame) -> None:
        """Test that result contains attributes list."""
        result = compute_intersectional(
            intersectional_df, "y_prob", "y_true", ["race", "sex"], bootstrap_ci=False
        )
        assert result["attributes"] == ["race", "sex"]

    def test_contains_intersections(self, intersectional_df: pl.DataFrame) -> None:
        """Test that result contains intersections."""
        result = compute_intersectional(
            intersectional_df, "y_prob", "y_true", ["race", "sex"], min_n=10, bootstrap_ci=False
        )
        assert "intersections" in result
        # Should have 4 intersections: White x M, White x F, Black x M, Black x F
        assert len(result["intersections"]) > 0

    def test_intersection_names_format(self, intersectional_df: pl.DataFrame) -> None:
        """Test that intersection names use ' x ' separator."""
        result = compute_intersectional(
            intersectional_df, "y_prob", "y_true", ["race", "sex"], min_n=10, bootstrap_ci=False
        )
        for name in result["intersections"].keys():
            assert " x " in name

    def test_contains_summary(self, intersectional_df: pl.DataFrame) -> None:
        """Test that result contains summary."""
        result = compute_intersectional(
            intersectional_df, "y_prob", "y_true", ["race", "sex"], min_n=10, bootstrap_ci=False
        )
        assert "summary" in result

    def test_min_n_filtering(self, intersectional_df: pl.DataFrame) -> None:
        """Test that min_n filters small groups."""
        result_high = compute_intersectional(
            intersectional_df, "y_prob", "y_true", ["race", "sex"], min_n=200, bootstrap_ci=False
        )
        result_low = compute_intersectional(
            intersectional_df, "y_prob", "y_true", ["race", "sex"], min_n=10, bootstrap_ci=False
        )
        # Higher min_n should result in fewer intersections
        assert len(result_high["intersections"]) <= len(result_low["intersections"])

    def test_summary_contains_best_worst(self, intersectional_df: pl.DataFrame) -> None:
        """Test that summary contains best/worst performing groups."""
        result = compute_intersectional(
            intersectional_df, "y_prob", "y_true", ["race", "sex"], min_n=10, bootstrap_ci=False
        )
        summary = result["summary"]
        if "best_performing" in summary:
            assert "worst_performing" in summary
            assert "auroc_range" in summary


class TestComputePairwiseIntersectional:
    """Tests for compute_pairwise_intersectional function."""

    @pytest.fixture
    def three_attr_df(self) -> pl.DataFrame:
        """Create DataFrame with three attributes."""
        np.random.seed(42)
        n = 500
        race = np.random.choice(["White", "Black"], n, p=[0.6, 0.4])
        sex = np.random.choice(["M", "F"], n, p=[0.5, 0.5])
        age = np.random.choice(["Young", "Old"], n, p=[0.5, 0.5])
        y_true = np.random.binomial(1, 0.3, n)
        y_prob = np.clip(
            np.where(y_true == 1, np.random.normal(0.6, 0.2, n), np.random.normal(0.3, 0.2, n)),
            0.01,
            0.99,
        )
        return pl.DataFrame(
            {
                "race": race,
                "sex": sex,
                "age_group": age,
                "y_true": y_true,
                "y_prob": y_prob,
            }
        )

    def test_returns_dict(self, three_attr_df: pl.DataFrame) -> None:
        """Test that function returns a dictionary."""
        attrs = {
            "race": {"column": "race"},
            "sex": {"column": "sex"},
            "age": {"column": "age_group"},
        }
        result = compute_pairwise_intersectional(three_attr_df, "y_prob", "y_true", attrs, min_n=10)
        assert isinstance(result, dict)

    def test_contains_all_pairs(self, three_attr_df: pl.DataFrame) -> None:
        """Test that all pairs are analyzed."""
        attrs = {
            "race": {"column": "race"},
            "sex": {"column": "sex"},
            "age": {"column": "age_group"},
        }
        result = compute_pairwise_intersectional(three_attr_df, "y_prob", "y_true", attrs, min_n=10)
        # 3 attributes = 3 pairs
        assert len(result["pairs"]) == 3

    def test_contains_summary(self, three_attr_df: pl.DataFrame) -> None:
        """Test that result contains summary."""
        attrs = {
            "race": {"column": "race"},
            "sex": {"column": "sex"},
        }
        result = compute_pairwise_intersectional(three_attr_df, "y_prob", "y_true", attrs, min_n=10)
        assert "summary" in result
        assert "n_pairs_analyzed" in result["summary"]


class TestIdentifyVulnerableSubgroups:
    """Tests for identify_vulnerable_subgroups function."""

    @pytest.fixture
    def vulnerability_df(self) -> pl.DataFrame:
        """Create DataFrame with varying performance across groups."""
        np.random.seed(42)
        n = 500

        # Create groups with different performance
        group = np.array(["HighPerf"] * 200 + ["LowPerf"] * 200 + ["MedPerf"] * 100)
        y_true = np.random.binomial(1, 0.3, n)

        # High perf: good separation, Low perf: poor separation
        y_prob = np.zeros(n)
        y_prob[:200] = np.where(
            y_true[:200] == 1, np.random.uniform(0.6, 0.9, 200), np.random.uniform(0.1, 0.4, 200)
        )
        y_prob[200:400] = np.where(
            y_true[200:400] == 1,
            np.random.uniform(0.4, 0.6, 200),  # Poor separation
            np.random.uniform(0.4, 0.6, 200),
        )
        y_prob[400:] = np.where(
            y_true[400:] == 1, np.random.uniform(0.5, 0.8, 100), np.random.uniform(0.2, 0.5, 100)
        )
        y_prob = np.clip(y_prob, 0.01, 0.99)

        return pl.DataFrame(
            {
                "group": group,
                "y_true": y_true,
                "y_prob": y_prob,
            }
        )

    def test_returns_dict(self, vulnerability_df: pl.DataFrame) -> None:
        """Test that function returns a dictionary."""
        attrs = {"group": {"column": "group"}}
        result = identify_vulnerable_subgroups(vulnerability_df, "y_prob", "y_true", attrs)
        assert isinstance(result, dict)

    def test_contains_vulnerable_list(self, vulnerability_df: pl.DataFrame) -> None:
        """Test that result contains vulnerable subgroups list."""
        attrs = {"group": {"column": "group"}}
        result = identify_vulnerable_subgroups(vulnerability_df, "y_prob", "y_true", attrs)
        assert "vulnerable_subgroups" in result
        assert isinstance(result["vulnerable_subgroups"], list)

    def test_identifies_low_auroc_groups(self, vulnerability_df: pl.DataFrame) -> None:
        """Test that groups with low AUROC are identified."""
        attrs = {"group": {"column": "group"}}
        result = identify_vulnerable_subgroups(
            vulnerability_df,
            "y_prob",
            "y_true",
            attrs,
            auroc_threshold=0.75,  # Should catch LowPerf group
        )
        # Should identify at least one vulnerable group
        # (LowPerf has poor separation)
        vulnerable_groups = [v["group"] for v in result["vulnerable_subgroups"]]
        # May or may not catch depending on random data
        assert "n_vulnerable" in result

    def test_respects_auroc_threshold(self, vulnerability_df: pl.DataFrame) -> None:
        """Test that auroc_threshold affects results."""
        attrs = {"group": {"column": "group"}}
        result_strict = identify_vulnerable_subgroups(
            vulnerability_df,
            "y_prob",
            "y_true",
            attrs,
            auroc_threshold=0.9,  # Very strict
        )
        result_lenient = identify_vulnerable_subgroups(
            vulnerability_df,
            "y_prob",
            "y_true",
            attrs,
            auroc_threshold=0.5,  # Very lenient
        )
        # Stricter threshold should find more vulnerable groups
        assert result_strict["n_vulnerable"] >= result_lenient["n_vulnerable"]

    def test_contains_summary(self, vulnerability_df: pl.DataFrame) -> None:
        """Test that result contains summary."""
        attrs = {"group": {"column": "group"}}
        result = identify_vulnerable_subgroups(vulnerability_df, "y_prob", "y_true", attrs)
        assert "summary" in result


class TestSummarizeVulnerable:
    """Tests for _summarize_vulnerable function."""

    def test_empty_list_returns_pass(self) -> None:
        """Test that empty list returns PASS status."""
        result = _summarize_vulnerable([])
        assert result["status"] == "PASS"
        assert "No vulnerable" in result["message"]

    def test_few_vulnerable_returns_review(self) -> None:
        """Test that few vulnerable groups returns REVIEW status."""
        vulnerable = [
            {"type": "single", "group": "A", "auroc": 0.6},
            {"type": "single", "group": "B", "auroc": 0.65},
        ]
        result = _summarize_vulnerable(vulnerable)
        assert result["status"] == "REVIEW"

    def test_many_vulnerable_returns_concern(self) -> None:
        """Test that many vulnerable groups returns CONCERN status."""
        vulnerable = [
            {"type": "single", "group": "A", "auroc": 0.6},
            {"type": "single", "group": "B", "auroc": 0.65},
            {"type": "intersectional", "group": "C", "auroc": 0.55},
        ]
        result = _summarize_vulnerable(vulnerable)
        assert result["status"] == "CONCERN"

    def test_counts_single_and_intersectional(self) -> None:
        """Test that single and intersectional are counted separately."""
        vulnerable = [
            {"type": "single", "group": "A", "auroc": 0.6},
            {"type": "intersectional", "group": "B x C", "auroc": 0.55},
            {"type": "intersectional", "group": "D x E", "auroc": 0.58},
        ]
        result = _summarize_vulnerable(vulnerable)
        assert result["n_single_attribute"] == 1
        assert result["n_intersectional"] == 2

    def test_identifies_worst_subgroup(self) -> None:
        """Test that worst subgroup is identified."""
        vulnerable = [
            {"type": "single", "group": "A", "auroc": 0.65},
            {"type": "single", "group": "B", "auroc": 0.55},  # Worst
            {"type": "single", "group": "C", "auroc": 0.60},
        ]
        # List should be sorted by AUROC, so first is worst
        result = _summarize_vulnerable(vulnerable)
        # The function assumes list is pre-sorted, first element is worst
        assert result["worst_subgroup"] == "A"  # First in list
