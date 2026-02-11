"""
Tests for FairCareAI Van Calster visualization plots module.

Tests cover:
- Alt text generation functions
- Source annotation function
- create_auroc_forest_plot function
- create_calibration_plot_by_subgroup function
- create_decision_curve_by_subgroup function
- create_risk_distribution_plot function
- create_vancalster_dashboard function
"""

import numpy as np
from plotly.graph_objects import Figure

from faircareai.visualization.utils import add_source_annotation
from faircareai.visualization.vancalster_plots import (
    _generate_auroc_forest_alt_text,
    _generate_calibration_alt_text,
    _generate_decision_curve_alt_text,
    _generate_risk_distribution_alt_text,
    create_auroc_forest_plot,
    create_calibration_plot_by_subgroup,
    create_decision_curve_by_subgroup,
    create_risk_distribution_plot,
    create_vancalster_dashboard,
)


class TestGenerateAurocForestAltText:
    """Tests for _generate_auroc_forest_alt_text function."""

    def test_empty_groups(self) -> None:
        """Test with empty groups."""
        result = _generate_auroc_forest_alt_text({"groups": {}}, "Test Title")
        assert "No data available" in result

    def test_no_auroc_values(self) -> None:
        """Test when groups have no AUROC values."""
        results = {"groups": {"Group A": {}, "Group B": {}}}
        result = _generate_auroc_forest_alt_text(results, "Test Title")
        assert "No AUROC values" in result

    def test_normal_auroc_values(self) -> None:
        """Test with normal AUROC values."""
        results = {
            "groups": {
                "Group A": {"auroc": 0.85},
                "Group B": {"auroc": 0.78},
            }
        }
        result = _generate_auroc_forest_alt_text(results, "AUROC by Group")
        assert "AUROC by Group" in result
        assert "2 demographic subgroups" in result
        assert "0.78" in result
        assert "0.85" in result

    def test_clinically_meaningful_difference(self) -> None:
        """Test when AUROC range >= 0.05."""
        results = {
            "groups": {
                "Group A": {"auroc": 0.90},
                "Group B": {"auroc": 0.75},
            }
        }
        result = _generate_auroc_forest_alt_text(results, "Test")
        assert "Clinically meaningful differences" in result

    def test_consistent_performance(self) -> None:
        """Test when AUROC range < 0.05."""
        results = {
            "groups": {
                "Group A": {"auroc": 0.82},
                "Group B": {"auroc": 0.80},
            }
        }
        result = _generate_auroc_forest_alt_text(results, "Test")
        assert "consistent across groups" in result


class TestGenerateCalibrationAltText:
    """Tests for _generate_calibration_alt_text function."""

    def test_empty_groups(self) -> None:
        """Test with empty groups."""
        result = _generate_calibration_alt_text({"groups": {}}, "Test Title")
        assert "No calibration data" in result

    def test_adequate_calibration(self) -> None:
        """Test when all groups are adequately calibrated."""
        results = {
            "groups": {
                "Group A": {"oe_ratio": 1.0},
                "Group B": {"oe_ratio": 0.95},
            }
        }
        result = _generate_calibration_alt_text(results, "Calibration")
        assert "adequate calibration" in result

    def test_miscalibration_overprediction(self) -> None:
        """Test when group has overprediction (O:E < 0.8)."""
        results = {
            "groups": {
                "Group A": {"oe_ratio": 0.5},  # overprediction
            }
        }
        result = _generate_calibration_alt_text(results, "Calibration")
        assert "overprediction" in result

    def test_miscalibration_underprediction(self) -> None:
        """Test when group has underprediction (O:E > 1.2)."""
        results = {
            "groups": {
                "Group A": {"oe_ratio": 1.5},  # underprediction
            }
        }
        result = _generate_calibration_alt_text(results, "Calibration")
        assert "underprediction" in result


class TestGenerateDecisionCurveAltText:
    """Tests for _generate_decision_curve_alt_text function."""

    def test_empty_groups(self) -> None:
        """Test with empty groups."""
        result = _generate_decision_curve_alt_text({"groups": {}}, "Test Title", 0.5)
        assert "No clinical utility data" in result

    def test_useful_range(self) -> None:
        """Test when groups have useful range."""
        results = {
            "groups": {
                "Group A": {"useful_range": {"min": 0.1, "max": 0.4}},
            }
        }
        result = _generate_decision_curve_alt_text(results, "Decision Curves", 0.3)
        assert "clinical benefit" in result
        assert "Group A" in result

    def test_no_useful_range(self) -> None:
        """Test when no groups have useful range."""
        results = {
            "groups": {
                "Group A": {"useful_range": {}},
                "Group B": {},
            }
        }
        result = _generate_decision_curve_alt_text(results, "Decision Curves", 0.3)
        assert "Limited clinical utility" in result

    def test_threshold_in_text(self) -> None:
        """Test that threshold is included in alt text."""
        results = {"groups": {"Group A": {}}}
        result = _generate_decision_curve_alt_text(results, "Test", 0.25)
        assert "25%" in result


class TestGenerateRiskDistributionAltText:
    """Tests for _generate_risk_distribution_alt_text function."""

    def test_empty_groups(self) -> None:
        """Test with empty groups."""
        result = _generate_risk_distribution_alt_text({"groups": {}}, "Test Title")
        assert "No risk distribution data" in result

    def test_adequate_separation(self) -> None:
        """Test when all groups have adequate separation."""
        results = {
            "groups": {
                "Group A": {"discrimination_slope": 0.3},
                "Group B": {"discrimination_slope": 0.25},
            }
        }
        result = _generate_risk_distribution_alt_text(results, "Risk Distribution")
        assert "Adequate separation" in result

    def test_high_overlap(self) -> None:
        """Test when groups have high overlap (low discrimination slope)."""
        results = {
            "groups": {
                "Group A": {"discrimination_slope": 0.15},  # < 0.2 threshold
            }
        }
        result = _generate_risk_distribution_alt_text(results, "Risk Distribution")
        assert "High overlap" in result
        assert "Group A" in result


class TestAddSourceAnnotation:
    """Tests for add_source_annotation function."""

    def test_adds_annotation(self) -> None:
        """Test that function returns figure unchanged (annotations moved to HTML)."""
        import plotly.graph_objects as go

        fig = go.Figure()
        result = add_source_annotation(fig)
        assert result is fig
        # Source annotations now added in HTML report footer, not Plotly inline

    def test_default_source_note(self) -> None:
        """Test that function accepts source note parameter."""
        import plotly.graph_objects as go

        fig = go.Figure()
        result = add_source_annotation(fig)
        assert result is not None
        assert isinstance(result, go.Figure)
        # Source note now handled in HTML report generation

    def test_custom_source_note(self) -> None:
        """Test custom source note parameter."""
        import plotly.graph_objects as go

        fig = go.Figure()
        result = add_source_annotation(fig, "Custom Source")
        assert result is not None
        assert isinstance(result, go.Figure)
        # Custom source notes now added in HTML report footer

    def test_contains_citation(self) -> None:
        """Test that citation parameter is accepted."""
        import plotly.graph_objects as go

        fig = go.Figure()
        result = add_source_annotation(fig, citation="Van Calster et al. (2025) Lancet Digit Health")
        assert result is not None
        assert isinstance(result, go.Figure)
        # Citations now added in HTML report footer


class TestCreateAurocForestPlot:
    """Tests for create_auroc_forest_plot function."""

    def test_returns_figure(self) -> None:
        """Test that function returns a Plotly Figure."""
        results = {"groups": {"Group A": {"auroc": 0.85, "n": 100}}}
        fig = create_auroc_forest_plot(results)
        assert isinstance(fig, Figure)

    def test_empty_groups(self) -> None:
        """Test with empty groups."""
        fig = create_auroc_forest_plot({"groups": {}})
        assert isinstance(fig, Figure)
        # Should have annotation about no data
        assert len(fig.layout.annotations) >= 1

    def test_no_valid_auroc(self) -> None:
        """Test when groups have no valid AUROC."""
        results = {"groups": {"Group A": {"n": 100}, "Group B": {"n": 50}}}
        fig = create_auroc_forest_plot(results)
        assert isinstance(fig, Figure)

    def test_custom_title(self) -> None:
        """Test with custom title."""
        results = {"groups": {"Group A": {"auroc": 0.85, "n": 100}}}
        fig = create_auroc_forest_plot(results, title="Custom Title")
        assert "Custom Title" in fig.layout.title.text

    def test_with_subtitle(self) -> None:
        """Test with subtitle."""
        results = {"groups": {"Group A": {"auroc": 0.85, "n": 100}}}
        fig = create_auroc_forest_plot(results, subtitle="My Subtitle")
        assert "My Subtitle" in fig.layout.title.text

    def test_reference_line_shown(self) -> None:
        """Test that reference line is added when show_reference_line=True."""
        results = {
            "groups": {
                "Group A": {"auroc": 0.85, "n": 100},
                "Group B": {"auroc": 0.75, "n": 50},
            }
        }
        fig = create_auroc_forest_plot(results, show_reference_line=True)
        # Check for vline shape
        assert len(fig.layout.shapes) > 0 or len(fig.data) > 0

    def test_custom_reference_auroc(self) -> None:
        """Test with custom reference AUROC value."""
        results = {"groups": {"Group A": {"auroc": 0.85, "n": 100}}}
        fig = create_auroc_forest_plot(results, reference_auroc=0.80)
        assert isinstance(fig, Figure)

    def test_overall_as_reference(self) -> None:
        """Test that Overall group is used as reference if present."""
        results = {
            "groups": {
                "Overall": {"auroc": 0.82, "n": 200},
                "Group A": {"auroc": 0.85, "n": 100},
            }
        }
        fig = create_auroc_forest_plot(results)
        assert isinstance(fig, Figure)

    def test_ghosting_enabled(self) -> None:
        """Test ghosting for small sample sizes."""
        results = {
            "groups": {
                "Large": {"auroc": 0.85, "n": 1000},
                "Small": {"auroc": 0.75, "n": 10},  # Small sample
            }
        }
        fig = create_auroc_forest_plot(results, enable_ghosting=True)
        assert isinstance(fig, Figure)

    def test_ghosting_disabled(self) -> None:
        """Test with ghosting disabled."""
        results = {"groups": {"Group A": {"auroc": 0.85, "n": 10}}}
        fig = create_auroc_forest_plot(results, enable_ghosting=False)
        assert isinstance(fig, Figure)

    def test_with_confidence_intervals(self) -> None:
        """Test with confidence intervals."""
        results = {
            "groups": {
                "Group A": {"auroc": 0.85, "n": 100, "auroc_ci_95": [0.80, 0.90]},
            }
        }
        fig = create_auroc_forest_plot(results)
        assert isinstance(fig, Figure)
        # Should have traces for CI whiskers and caps

    def test_reference_group_marked(self) -> None:
        """Test that reference group is marked."""
        results = {
            "groups": {
                "Group A": {"auroc": 0.85, "n": 100, "is_reference": True},
            }
        }
        fig = create_auroc_forest_plot(results)
        assert isinstance(fig, Figure)

    def test_custom_source_note(self) -> None:
        """Test with custom source note."""
        results = {"groups": {"Group A": {"auroc": 0.85, "n": 100}}}
        fig = create_auroc_forest_plot(results, source_note="Custom Note")
        assert isinstance(fig, Figure)

    def test_alt_text_in_meta(self) -> None:
        """Test that alt text is included in figure metadata."""
        results = {"groups": {"Group A": {"auroc": 0.85, "n": 100}}}
        fig = create_auroc_forest_plot(results)
        assert "description" in fig.layout.meta


class TestCreateCalibrationPlotBySubgroup:
    """Tests for create_calibration_plot_by_subgroup function."""

    def test_returns_figure(self) -> None:
        """Test that function returns a Plotly Figure."""
        results = {
            "groups": {
                "Group A": {
                    "calibration_curve": {
                        "prob_pred": [0.1, 0.3, 0.5],
                        "prob_true": [0.12, 0.28, 0.52],
                    },
                    "n": 100,
                    "oe_ratio": 1.0,
                    "brier_score": 0.1,
                }
            }
        }
        fig = create_calibration_plot_by_subgroup(results)
        assert isinstance(fig, Figure)

    def test_empty_groups(self) -> None:
        """Test with empty groups."""
        fig = create_calibration_plot_by_subgroup({"groups": {}})
        assert isinstance(fig, Figure)
        assert len(fig.layout.annotations) >= 1

    def test_includes_diagonal(self) -> None:
        """Test that perfect calibration diagonal is included."""
        results = {
            "groups": {
                "Group A": {
                    "calibration_curve": {"prob_pred": [0.1, 0.5], "prob_true": [0.1, 0.5]},
                    "n": 100,
                    "oe_ratio": 1.0,
                    "brier_score": 0.1,
                }
            }
        }
        fig = create_calibration_plot_by_subgroup(results)
        # Should have at least diagonal trace and one group trace
        assert len(fig.data) >= 2

    def test_skips_groups_with_error(self) -> None:
        """Test that groups with errors are skipped."""
        results = {
            "groups": {
                "Group A": {"error": "Not enough data"},
            }
        }
        fig = create_calibration_plot_by_subgroup(results)
        assert isinstance(fig, Figure)

    def test_skips_empty_calibration_data(self) -> None:
        """Test that groups with empty calibration curves are skipped."""
        results = {
            "groups": {
                "Group A": {"calibration_curve": {"prob_pred": [], "prob_true": []}, "n": 100},
            }
        }
        fig = create_calibration_plot_by_subgroup(results)
        assert isinstance(fig, Figure)

    def test_custom_title(self) -> None:
        """Test with custom title."""
        results = {
            "groups": {
                "Group A": {"calibration_curve": {"prob_pred": [0.5], "prob_true": [0.5]}, "n": 100}
            }
        }
        fig = create_calibration_plot_by_subgroup(results, title="Custom Cal Title")
        assert "Custom Cal Title" in fig.layout.title.text

    def test_alt_text_in_meta(self) -> None:
        """Test that alt text is included in metadata."""
        results = {
            "groups": {
                "Group A": {"calibration_curve": {"prob_pred": [0.5], "prob_true": [0.5]}, "n": 100}
            }
        }
        fig = create_calibration_plot_by_subgroup(results)
        assert "description" in fig.layout.meta


class TestCreateDecisionCurveBySubgroup:
    """Tests for create_decision_curve_by_subgroup function."""

    def test_returns_figure(self) -> None:
        """Test that function returns a Plotly Figure."""
        thresholds = np.linspace(0.01, 0.99, 50)
        results = {
            "groups": {
                "Group A": {
                    "decision_curve": {
                        "thresholds": thresholds.tolist(),
                        "net_benefit_model": (0.1 - 0.1 * thresholds).tolist(),
                        "net_benefit_all": (0.08 - 0.05 * thresholds).tolist(),
                        "net_benefit_none": [0] * len(thresholds),
                    },
                    "n": 100,
                }
            },
            "primary_threshold": 0.3,
        }
        fig = create_decision_curve_by_subgroup(results)
        assert isinstance(fig, Figure)

    def test_empty_groups(self) -> None:
        """Test with empty groups."""
        fig = create_decision_curve_by_subgroup({"groups": {}})
        assert isinstance(fig, Figure)

    def test_reference_strategies_shown(self) -> None:
        """Test reference strategies (treat all/none) are shown."""
        thresholds = np.linspace(0.05, 0.5, 20)
        results = {
            "groups": {
                "Group A": {
                    "decision_curve": {
                        "thresholds": thresholds.tolist(),
                        "net_benefit_model": [0.1] * len(thresholds),
                        "net_benefit_all": [0.08] * len(thresholds),
                        "net_benefit_none": [0] * len(thresholds),
                    },
                    "n": 100,
                }
            }
        }
        fig = create_decision_curve_by_subgroup(results, show_reference_strategies=True)
        # Should have traces for treat none, treat all, and model
        trace_names = [t.name for t in fig.data if t.name]
        assert "Treat None" in trace_names
        assert "Treat All" in trace_names

    def test_custom_threshold_range(self) -> None:
        """Test with custom threshold range."""
        thresholds = np.linspace(0.01, 0.99, 50)
        results = {
            "groups": {
                "Group A": {
                    "decision_curve": {
                        "thresholds": thresholds.tolist(),
                        "net_benefit_model": [0.1] * len(thresholds),
                        "net_benefit_all": [0.08] * len(thresholds),
                        "net_benefit_none": [0] * len(thresholds),
                    },
                    "n": 100,
                }
            }
        }
        fig = create_decision_curve_by_subgroup(results, threshold_range=(0.1, 0.4))
        # Compare as list since plotly may return tuple
        assert list(fig.layout.xaxis.range) == [0.1, 0.4]

    def test_primary_threshold_vline(self) -> None:
        """Test that vertical line is added at primary threshold."""
        thresholds = np.linspace(0.05, 0.5, 20)
        results = {
            "groups": {
                "Group A": {
                    "decision_curve": {
                        "thresholds": thresholds.tolist(),
                        "net_benefit_model": [0.1] * len(thresholds),
                        "net_benefit_all": [0.08]
                        * len(thresholds),  # Required for reference strategies
                        "net_benefit_none": [0]
                        * len(thresholds),  # Required for reference strategies
                    },
                    "n": 100,
                }
            },
            "primary_threshold": 0.25,
        }
        fig = create_decision_curve_by_subgroup(results)
        # Should have a vertical line shape
        assert len(fig.layout.shapes) > 0

    def test_skips_groups_with_error(self) -> None:
        """Test that groups with errors are skipped."""
        results = {
            "groups": {
                "Group A": {"error": "Insufficient data"},
            }
        }
        fig = create_decision_curve_by_subgroup(results)
        assert isinstance(fig, Figure)


class TestCreateRiskDistributionPlot:
    """Tests for create_risk_distribution_plot function."""

    def test_returns_figure(self) -> None:
        """Test that function returns a Plotly Figure."""
        results = {
            "groups": {
                "Group A": {
                    "n": 100,
                    "events": {
                        "histogram": {"counts": [10, 20, 30], "bin_centers": [0.3, 0.5, 0.7]}
                    },
                    "non_events": {
                        "histogram": {"counts": [30, 20, 10], "bin_centers": [0.2, 0.4, 0.6]}
                    },
                }
            }
        }
        fig = create_risk_distribution_plot(results)
        assert isinstance(fig, Figure)

    def test_empty_groups(self) -> None:
        """Test with empty groups."""
        fig = create_risk_distribution_plot({"groups": {}})
        assert isinstance(fig, Figure)

    def test_violin_plot_type(self) -> None:
        """Test violin plot type."""
        results = {
            "groups": {
                "Group A": {
                    "n": 100,
                    "events": {"histogram": {"counts": [10, 20], "bin_centers": [0.5, 0.7]}},
                    "non_events": {"histogram": {"counts": [20, 10], "bin_centers": [0.3, 0.5]}},
                }
            }
        }
        fig = create_risk_distribution_plot(results, plot_type="violin")
        assert isinstance(fig, Figure)

    def test_box_plot_type(self) -> None:
        """Test box plot type."""
        results = {
            "groups": {
                "Group A": {
                    "n": 100,
                    "events": {"histogram": {"counts": [10, 20], "bin_centers": [0.5, 0.7]}},
                    "non_events": {"histogram": {"counts": [20, 10], "bin_centers": [0.3, 0.5]}},
                }
            }
        }
        fig = create_risk_distribution_plot(results, plot_type="box")
        assert isinstance(fig, Figure)

    def test_summary_statistics_fallback(self) -> None:
        """Test fallback to summary statistics when no histogram."""
        results = {
            "groups": {
                "Group A": {
                    "n": 100,
                    "events": {"mean": 0.7, "std": 0.15, "n": 50},
                    "non_events": {"mean": 0.3, "std": 0.15, "n": 50},
                }
            }
        }
        fig = create_risk_distribution_plot(results)
        assert isinstance(fig, Figure)

    def test_skips_groups_with_error(self) -> None:
        """Test that groups with errors are skipped."""
        results = {
            "groups": {
                "Group A": {"error": "Insufficient data"},
            }
        }
        fig = create_risk_distribution_plot(results)
        assert isinstance(fig, Figure)

    def test_skips_outcomes_with_error(self) -> None:
        """Test that outcomes with errors are skipped."""
        results = {
            "groups": {
                "Group A": {
                    "n": 100,
                    "events": {"error": "No events"},
                    "non_events": {"mean": 0.3, "std": 0.15, "n": 50},
                }
            }
        }
        fig = create_risk_distribution_plot(results)
        assert isinstance(fig, Figure)

    def test_custom_title(self) -> None:
        """Test with custom title."""
        results = {
            "groups": {
                "Group A": {
                    "n": 100,
                    "events": {"mean": 0.7, "std": 0.15, "n": 50},
                    "non_events": {"mean": 0.3, "std": 0.15, "n": 50},
                }
            }
        }
        fig = create_risk_distribution_plot(results, title="Custom Risk Title")
        assert "Custom Risk Title" in fig.layout.title.text


class TestCreateVancalsterDashboard:
    """Tests for create_vancalster_dashboard function."""

    def test_returns_figure(self) -> None:
        """Test that function returns a Plotly Figure."""
        np.random.seed(42)
        thresholds = np.linspace(0.05, 0.5, 20)
        results = {
            "by_subgroup": {
                "Group A": {
                    "discrimination": {"auroc": 0.85, "auroc_ci_95": [0.80, 0.90]},
                    "calibration": {
                        "calibration_curve": {
                            "prob_pred": [0.1, 0.3, 0.5],
                            "prob_true": [0.12, 0.28, 0.52],
                        }
                    },
                    "clinical_utility": {
                        "decision_curve": {
                            "thresholds": thresholds.tolist(),
                            "net_benefit_model": [0.1] * len(thresholds),
                        }
                    },
                    "risk_distribution": {
                        "events": {"mean": 0.7, "q25": 0.6, "q75": 0.8},
                        "non_events": {"mean": 0.3, "q25": 0.2, "q75": 0.4},
                    },
                }
            }
        }
        fig = create_vancalster_dashboard(results)
        assert isinstance(fig, Figure)

    def test_empty_subgroups(self) -> None:
        """Test with empty subgroups."""
        fig = create_vancalster_dashboard({"by_subgroup": {}})
        assert isinstance(fig, Figure)
        # Should have annotation about no data
        assert len(fig.layout.annotations) >= 1

    def test_custom_title(self) -> None:
        """Test with custom title."""
        results = {
            "by_subgroup": {
                "Group A": {
                    "discrimination": {"auroc": 0.85},
                    "calibration": {"calibration_curve": {}},
                    "clinical_utility": {"decision_curve": {}},
                    "risk_distribution": {},
                }
            }
        }
        fig = create_vancalster_dashboard(results, title="Custom Dashboard")
        assert "Custom Dashboard" in fig.layout.title.text

    def test_multiple_groups(self) -> None:
        """Test with multiple groups."""
        thresholds = np.linspace(0.05, 0.5, 20)
        results = {
            "by_subgroup": {
                "Group A": {
                    "discrimination": {"auroc": 0.85},
                    "calibration": {"calibration_curve": {"prob_pred": [0.5], "prob_true": [0.5]}},
                    "clinical_utility": {
                        "decision_curve": {
                            "thresholds": thresholds.tolist(),
                            "net_benefit_model": [0.1] * len(thresholds),
                        }
                    },
                    "risk_distribution": {
                        "events": {"mean": 0.7, "q25": 0.6, "q75": 0.8},
                        "non_events": {},
                    },
                },
                "Group B": {
                    "discrimination": {"auroc": 0.78},
                    "calibration": {"calibration_curve": {"prob_pred": [0.5], "prob_true": [0.45]}},
                    "clinical_utility": {
                        "decision_curve": {
                            "thresholds": thresholds.tolist(),
                            "net_benefit_model": [0.08] * len(thresholds),
                        }
                    },
                    "risk_distribution": {
                        "events": {"mean": 0.65, "q25": 0.55, "q75": 0.75},
                        "non_events": {},
                    },
                },
            }
        }
        fig = create_vancalster_dashboard(results)
        assert isinstance(fig, Figure)

    def test_handles_missing_auroc(self) -> None:
        """Test handling of missing AUROC values."""
        results = {
            "by_subgroup": {
                "Group A": {
                    "discrimination": {},  # No AUROC
                    "calibration": {"calibration_curve": {}},
                    "clinical_utility": {"decision_curve": {}},
                    "risk_distribution": {},
                }
            }
        }
        fig = create_vancalster_dashboard(results)
        assert isinstance(fig, Figure)

    def test_handles_missing_calibration(self) -> None:
        """Test handling of missing calibration data."""
        results = {
            "by_subgroup": {
                "Group A": {
                    "discrimination": {"auroc": 0.85},
                    "calibration": {"calibration_curve": {"prob_pred": [], "prob_true": []}},
                    "clinical_utility": {"decision_curve": {}},
                    "risk_distribution": {},
                }
            }
        }
        fig = create_vancalster_dashboard(results)
        assert isinstance(fig, Figure)

    def test_handles_missing_decision_curve(self) -> None:
        """Test handling of missing decision curve data."""
        results = {
            "by_subgroup": {
                "Group A": {
                    "discrimination": {"auroc": 0.85},
                    "calibration": {"calibration_curve": {}},
                    "clinical_utility": {
                        "decision_curve": {"thresholds": [], "net_benefit_model": []}
                    },
                    "risk_distribution": {},
                }
            }
        }
        fig = create_vancalster_dashboard(results)
        assert isinstance(fig, Figure)

    def test_handles_risk_distribution_with_error(self) -> None:
        """Test handling of risk distribution with errors."""
        results = {
            "by_subgroup": {
                "Group A": {
                    "discrimination": {"auroc": 0.85},
                    "calibration": {"calibration_curve": {}},
                    "clinical_utility": {"decision_curve": {}},
                    "risk_distribution": {
                        "events": {"error": "No events"},
                        "non_events": {"error": "Too few samples"},
                    },
                }
            }
        }
        fig = create_vancalster_dashboard(results)
        assert isinstance(fig, Figure)

    def test_long_group_name_truncation(self) -> None:
        """Test that long group names are truncated in dashboard."""
        results = {
            "by_subgroup": {
                "Very Long Group Name That Should Be Truncated": {
                    "discrimination": {"auroc": 0.85},
                    "calibration": {"calibration_curve": {}},
                    "clinical_utility": {"decision_curve": {}},
                    "risk_distribution": {
                        "events": {"mean": 0.7, "q25": 0.6, "q75": 0.8},
                        "non_events": {"mean": 0.3, "q25": 0.2, "q75": 0.4},
                    },
                }
            }
        }
        fig = create_vancalster_dashboard(results)
        assert isinstance(fig, Figure)
