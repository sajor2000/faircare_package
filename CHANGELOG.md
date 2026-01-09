# Changelog

All notable changes to FairCareAI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2025-12-17

### Added

- **Two Output Personas**
  - Data Scientist persona: Full technical reports (15-20+ pages) for model validation
  - Governance persona: Streamlined 3-5 page reports for committee review
  - New export methods: `to_governance_html()`, `to_governance_pdf()`
  - Persona parameter: `results.to_html(persona="governance")`

- **Van Calster et al. (2025) Visualizations**
  - 4 overall performance figures: AUROC gauge, Calibration plot, Brier score, Classification metrics
  - 4 subgroup fairness figures: AUROC by group, Sensitivity (TPR), FPR, Selection rate
  - Plain language explanations on every figure for non-technical audiences

- **Plain Language Explanations**
  - Crystal-clear X/Y axis labels (e.g., "Predicted Risk (what the model says)")
  - "What to look for" guidance boxes in reports
  - Clinical interpretation guidance for governance committees
  - Metric glossaries with lay explanations

- **Publication-Ready Typography**
  - Minimum 14px font size throughout all visualizations
  - Publication-compliant styling standards
  - Dynamic chart heights that scale with data
  - Generous margins preventing text truncation

- **CHAI RAIC Framework Alignment**
  - 7 governance report sections mapped to CHAI checkpoints
  - Advisory disclaimer banner: "Package SUGGESTS, humans DECIDE"
  - Detection-based language ("No Issues Detected" vs "Ready for Deployment")

### Changed

- Improved all figure legend text for clarity and accessibility
- Updated disclaimer fonts from 10-13px to 14px minimum
- Enhanced table font sizes for better readability
- Refined axis labels for clinical interpretation

### Fixed

- Font size consistency across all governance dashboard figures
- Table header and cell font sizes for publication standards
- Waterfall chart subtitle legibility

---

## [0.2.0] - 2024-12-09

### Added

- **Core Fairness Auditing**
  - `FairCareAudit` class for end-to-end fairness analysis
  - Automatic detection of sensitive attributes (race, sex, insurance, language)
  - Support for Polars DataFrames, Parquet, and CSV files
  - Bootstrap confidence intervals for all metrics

- **CHAI-Aligned Governance Framework**
  - Use-case based fairness metric recommendations
  - Impossibility theorem acknowledgment in decision tree
  - Advisory-only outputs with clear disclaimers
  - Go/no-go governance recommendations

- **Fairness Metrics**
  - Demographic Parity (selection rate equality)
  - Equalized Odds (TPR and FPR equality)
  - Equal Opportunity (TPR equality)
  - Predictive Parity (PPV equality)
  - Calibration by group

- **Performance Metrics (TRIPOD+AI Compliant)**
  - AUROC with bootstrap CI
  - AUPRC (Average Precision)
  - Brier Score
  - Calibration slope, intercept, E/O ratio
  - Decision Curve Analysis
  - Threshold sensitivity analysis

- **Statistical Methods**
  - Wilson Score CI for proportions
  - Newcombe-Wilson CI for differences
  - Katz Log-Method CI for ratios (80% rule)
  - Cluster-aware bootstrap for hierarchical data
  - Stratified permutation tests for hypothesis testing
  - Multiplicity control (Holm-Bonferroni, BH-FDR)

- **Visualization (WCAG 2.1 AA Compliant)**
  - Executive summary dashboard
  - Forest plots with confidence intervals
  - ROC and Precision-Recall curves
  - Calibration plots
  - Subgroup comparison charts
  - Okabe-Ito colorblind-safe palette

- **Report Generation**
  - Interactive HTML reports
  - PDF reports for governance review
  - PowerPoint decks for board presentations
  - JSON export for programmatic use

- **Interactive Dashboard**
  - Streamlit-based multi-page application
  - Data upload and validation
  - Interactive fairness analysis
  - Dual-audience support (executives vs. data scientists)

- **Command-Line Interface**
  - `faircareai dashboard` - Launch interactive dashboard
  - `faircareai audit` - Run audit from command line
  - `faircareai info` - Display fairness metrics guide

### Security

- Local-only processing (HIPAA-friendly)
- No external API calls or cloud dependencies
- PHI-safe error messages

## [0.1.0] - 2024-11-01

### Added

- Initial development release
- Basic fairness metrics computation
- Prototype visualization

---

## Release Notes

### Upgrading to 0.2.0

This is a major refactoring release. Key changes:

1. **New Primary Class**: Use `FairCareAudit` instead of `FairAudit`
2. **New Results Container**: `AuditResults` replaces `AuditResult`
3. **Configuration Required**: Must set `primary_fairness_metric` and `fairness_justification`

```python
# Old API (deprecated)
from faircareai import FairAudit
audit = FairAudit(df, threshold=0.5)

# New API
from faircareai import FairCareAudit, FairnessConfig
from faircareai.core.config import FairnessMetric

audit = FairCareAudit(data=df, pred_col="risk", target_col="outcome")
audit.config = FairnessConfig(
    model_name="My Model",
    primary_fairness_metric=FairnessMetric.EQUALIZED_ODDS,
    fairness_justification="Model triggers intervention..."
)
results = audit.run()
```

### Governance Philosophy

> **Package SUGGESTS, humans DECIDE**

All FairCareAI outputs are advisory. Final deployment decisions rest with clinical stakeholders and governance committees.
