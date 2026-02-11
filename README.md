# FairCareAI

[![CI](https://github.com/sajor2000/faircare_package/actions/workflows/ci.yml/badge.svg)](https://github.com/sajor2000/faircare_package/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/faircareai.svg)](https://pypi.org/project/faircareai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![WCAG 2.1 AA](https://img.shields.io/badge/WCAG-2.1%20AA-green.svg)](https://www.w3.org/WAI/WCAG21/quickref/)

**Healthcare AI Fairness Auditing for Clinical Decision Support**

FairCareAI is a Python package for auditing machine learning models for fairness in clinical contexts. Built on the **Van Calster et al. (2025)** methodology and aligned with the **CHAI RAIC** governance framework, it helps health system data scientists present evidence-based fairness analysis to governance stakeholders.

---

## Table of Contents

- [Key Features](#key-features)
- [Governance Philosophy](#governance-philosophy)
- [Installation](#installation)
- [Data Preparation Guide](#data-preparation-guide)
- [Quick Start](#quick-start)
- [Output Personas](#output-personas)
- [Data Requirements](#data-requirements)
- [Fairness Visualizations](#fairness-visualizations)
- [Fairness Metrics](#fairness-metrics)
- [Use Cases](#use-cases)
- [API Reference](#api-reference)
- [Accessibility Features](#accessibility-features)
- [Governance Compliance (CHAI RAIC)](#governance-compliance-chai-raic)
- [Interactive Dashboard](#interactive-dashboard)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [Citation](#citation)
- [References](#references)

---

## Key Features

- **Two Output Personas**: Full technical reports for data scientists, streamlined 3-5 page reports for governance committees
- **Van Calster et al. (2025) Methodology**: 4 recommended fairness visualizations (AUROC, Calibration, Sensitivity/TPR, Selection Rate)
- **CHAI RAIC Framework**: Aligned with Coalition for Health AI governance standards
- **Plain Language Explanations**: Every visualization includes clear explanations of axes, metrics, and clinical significance
- **Publication-Ready Typography**: Minimum 14px fonts
- **Accessibility-First Design**: WCAG 2.1 AA compliant, colorblind-safe palettes
- **Multiple Export Formats**: HTML dashboards, PDF reports, PowerPoint decks
- **HIPAA-Friendly**: All computation runs locally, no cloud dependencies
- **TRIPOD+AI Compliant**: Scientifically validated performance metrics

---

## Governance Philosophy

> **Package SUGGESTS, humans DECIDE**

All outputs are ADVISORY, not mandates. FairCareAI computes metrics and presents visualizations per established clinical AI fairness methodology. Final deployment decisions rest with clinical stakeholders and governance committees who understand the local context, organizational values, and patient populations.

---

## Installation

### Basic Installation

```bash
pip install faircareai
```

### With Export Capabilities (PDF/PPTX)

```bash
pip install "faircareai[export]"
python -m playwright install chromium  # Required for PDF generation
```

**Note**: PDF generation uses Playwright for cross-platform compatibility. See [docs/PDF_SETUP_GUIDE.md](docs/PDF_SETUP_GUIDE.md) for details.

### Development Installation

```bash
git clone https://github.com/sajor2000/faircare_package.git
cd faircareai
pip install -e ".[dev]"
```

### Platform Support

FairCareAI is tested and supported on:

| Platform | Python Versions | Status |
|----------|-----------------|--------|
| **macOS** (Intel & Apple Silicon) | 3.10, 3.11, 3.12 | ✅ Fully Supported |
| **Windows** (x64) | 3.10, 3.11, 3.12 | ✅ Fully Supported |
| **Linux** (Ubuntu, Debian, RHEL, Arch) | 3.10, 3.11, 3.12 | ✅ Fully Supported |

**Notes:**
- PDF generation requires Playwright Chromium browser (auto-installed with `python -m playwright install chromium`)
- No system dependencies required (all Python packages)
- Identical setup process on all platforms
- CI/CD tested on all platform combinations

### Requirements

- Python >= 3.10
- polars >= 0.20.0
- plotly >= 5.18.0
- streamlit >= 1.30.0
- scipy >= 1.11.0
- statsmodels >= 0.14.0

See `pyproject.toml` for complete dependencies.

---

## Data Preparation Guide

Before using FairCareAI, prepare a dataset with your model's predictions. This section covers what data scientists need to bring and common preparation steps.

### What You Need to Bring

| Required | Column | Type | Description |
|----------|--------|------|-------------|
| Yes | **Predictions** | float [0.0, 1.0] | Model-generated risk probabilities |
| Yes | **Outcomes** | int (0 or 1) | Actual binary outcomes |
| Recommended | **Sensitive Attributes** | string/categorical | Demographics (auto-detected or custom) |

### Supported File Formats

FairCareAI accepts multiple input formats:

| Format | Extension | Example |
|--------|-----------|---------|
| **Parquet** | `.parquet` | `FairCareAudit(data="predictions.parquet", ...)` |
| **CSV** | `.csv` | `FairCareAudit(data="data.csv", ...)` |
| **Polars DataFrame** | - | `FairCareAudit(data=pl_df, ...)` |
| **Pandas DataFrame** | - | `FairCareAudit(data=pd_df, ...)` |

### Prediction Column Requirements

Your **prediction column** must contain:
- **Probabilities** in range [0.0, 1.0] (NOT logits, NOT raw scores)
- One value per patient/observation

```python
# Correct: Probabilities from model.predict_proba()
y_prob = model.predict_proba(X_test)[:, 1]  # Second column for positive class

# Wrong: Logits (unbounded values)
logits = model.decision_function(X_test)  # Must convert to probabilities first

# Wrong: Binary predictions
y_pred = model.predict(X_test)  # Use predict_proba(), not predict()
```

### Outcome Column Requirements

Your **outcome/target column** must contain:
- **Binary values**: exactly 0 or 1
- 1 = event occurred (e.g., readmission, mortality)
- 0 = event did not occur

### CSV File Format

For CSV files, ensure:
- **Header row required** with column names
- **UTF-8 encoding** (standard)
- **Comma-delimited** (standard CSV)
- No special quoting needed for most data

```csv
patient_id,risk_score,readmit_30d,race,sex,insurance
P001,0.72,1,White,Female,Medicare
P002,0.31,0,Black,Male,Medicaid
P003,0.85,1,Hispanic,Female,Commercial
```

### Parquet Files (Recommended for Large Datasets)

Parquet is recommended for:
- Datasets > 100,000 rows
- Faster loading times
- Smaller file sizes
- Better type preservation

```python
# Save your predictions as Parquet
import polars as pl

df = pl.DataFrame({
    "risk_score": y_prob,
    "outcome": y_test,
    "race": race_values,
    "sex": sex_values,
})
df.write_parquet("predictions.parquet")
```

### Common Data Preparation Mistakes

| Mistake | Problem | Solution |
|---------|---------|----------|
| Using logits instead of probabilities | Values outside [0,1] | Apply sigmoid: `1 / (1 + np.exp(-logits))` |
| Using `model.predict()` | Returns 0/1, not probabilities | Use `model.predict_proba()[:, 1]` |
| Missing values in predictions | Audit will fail | Impute or remove rows with NaN |
| Outcome not binary | Validation error | Ensure values are exactly 0 or 1 |
| Predictions on training data | Overfitting bias | Use held-out test set predictions |

### Pre-Audit Checklist

Before running FairCareAI, ensure you have:

- [ ] **Model predictions** as probabilities [0.0, 1.0]
- [ ] **Actual outcomes** as binary (0 or 1)
- [ ] **Held-out test set** (not training data)
- [ ] **Sensitive attribute columns** (race, sex, age_group, etc.)
- [ ] **Clinical context** (threshold, use case type)

### Minimum Viable Dataset

```python
# Minimum required columns
required_columns = {
    "prediction_column": "risk_score",   # Probabilities [0, 1]
    "outcome_column": "readmit_30d",     # Binary 0/1
}

# Recommended: At least one sensitive attribute
recommended_columns = {
    "race": ["White", "Black", "Hispanic", "Asian", "Other"],
    "sex": ["Male", "Female"],
}
```

---

## Quick Start

### Step 1: Load Your Data

FairCareAI accepts multiple input formats. Choose the one that fits your workflow:

```python
from faircareai import FairCareAudit, FairnessConfig
from faircareai.core.config import FairnessMetric, UseCaseType

# Option A: Parquet file (recommended for large datasets)
audit = FairCareAudit(
    data="predictions.parquet",
    pred_col="risk_score",
    target_col="readmit_30d"
)

# Option B: CSV file
audit = FairCareAudit(
    data="patient_predictions.csv",
    pred_col="risk_score",
    target_col="readmit_30d"
)

# Option C: Polars DataFrame
import polars as pl
df = pl.read_csv("data.csv")
audit = FairCareAudit(data=df, pred_col="risk_score", target_col="readmit_30d")

# Option D: Pandas DataFrame
import pandas as pd
df = pd.read_csv("data.csv")
audit = FairCareAudit(data=df, pred_col="risk_score", target_col="readmit_30d")
```

### Step 2: Configure Sensitive Attributes

```python
# See suggested sensitive attributes
audit.suggest_attributes()
# Output: Detected race_ethnicity, sex, insurance columns...

# Accept suggestions (1-indexed)
audit.accept_suggested_attributes([1, 2, 3])
```

### Step 3: Set Fairness Configuration

```python
# Get fairness metric recommendation based on use case
audit.config.use_case_type = UseCaseType.INTERVENTION_TRIGGER
recommendation = audit.suggest_fairness_metric()
print(recommendation)

# Configure the audit
audit.config = FairnessConfig(
    model_name="Readmission Risk Model v2.0",
    model_version="2.0.0",
    intended_use="Trigger care management outreach for high-risk patients",
    intended_population="Adult patients discharged from acute care",
    primary_fairness_metric=FairnessMetric.EQUALIZED_ODDS,
    fairness_justification=(
        "Model triggers intervention (care management). "
        "Equalized odds ensures equal TPR/FPR across groups, "
        "preventing differential access to beneficial intervention."
    ),
    use_case_type=UseCaseType.INTERVENTION_TRIGGER,
)
```

### Step 4: Run Audit and Export Reports

```python
# Run the audit
results = audit.run()

# View executive summary
results.plot_executive_summary()

# Export for Data Scientist (full technical output - default)
results.to_html("fairness_report.html")
results.to_pdf("fairness_report.pdf")

# Export for Governance (streamlined 3-5 page output)
results.to_governance_html("governance.html")
results.to_governance_pdf("governance.pdf")
# Or equivalently:
results.to_html("governance.html", persona="governance")
results.to_pdf("governance.pdf", persona="governance")

# PowerPoint deck (always governance-focused)
results.to_pptx("governance_deck.pptx")
```

---

## Output Personas

FairCareAI supports two output personas to serve different audiences with tailored content:

### Data Scientist (Default)

**Purpose**: Full technical validation and model documentation
**Audience**: Data scientists, ML engineers, statisticians
**Content**:
- All 7 report sections (Executive Summary, Descriptive Stats, Overall Performance, Subgroup Performance, Fairness Assessment, Flags, Governance Decision)
- Complete metric tables with confidence intervals
- All visualizations (~15-20 figures)
- Technical terminology and statistical detail
- Bootstrap confidence intervals
- **Length**: 15-20+ pages

**Use when**:
- Documenting model validation for regulatory submission
- Internal technical review and methodology audit
- Research publication and reproducibility
- Detailed debugging of fairness issues

### Governance

**Purpose**: Streamlined executive review for decision-making
**Audience**: Governance committees, clinical leadership, non-technical stakeholders
**Content**:
- 5 key sections (Executive Summary, Overall Performance, Subgroup Performance, Flags, Decision Block)
- 8 key figures following Van Calster et al. (2025):
  - 4 overall performance figures (AUROC, Calibration, Brier Score, Classification Metrics)
  - 4 subgroup fairness figures per attribute (AUROC, Sensitivity/TPR, FPR, Selection Rate)
- Plain language summaries with clinical interpretation
- Clear pass/warning/flag indicators
- **Minimum 14px fonts** for readability
- **WCAG 2.1 AA compliant** visualizations
- **Length**: 3-5 pages

**Use when**:
- Presenting to governance committees or IRB
- Board meeting materials and executive briefings
- Clinical deployment approval process
- Non-technical stakeholder communication

### API Examples

```python
# Data Scientist output (default)
results.to_html("full_report.html")
results.to_pdf("full_report.pdf")

# Governance output (method 1: convenience methods)
results.to_governance_html("governance.html")
results.to_governance_pdf("governance.pdf")

# Governance output (method 2: persona parameter)
results.to_html("governance.html", persona="governance")
results.to_pdf("governance.pdf", persona="governance")

# PowerPoint (always governance-focused)
results.to_pptx("governance_deck.pptx")
```

---

## Data Requirements

### Required Columns

Your data must include:

| Column | Description | Example |
|--------|-------------|---------|
| **pred_col** | Model predictions (probabilities 0-1) | `risk_score`, `predicted_prob`, `y_prob` |
| **target_col** | Actual outcomes (binary 0/1) | `readmit_30d`, `mortality`, `outcome` |

### Auto-Detected Sensitive Attributes

FairCareAI automatically detects common healthcare demographic columns:

| Attribute | Detected Column Names | Default Reference |
|-----------|----------------------|-------------------|
| **Race/Ethnicity** | `race`, `ethnicity`, `race_eth`, `patient_race`, `race_cd`, `race_ethnicity` | White |
| **Sex** | `sex`, `gender`, `patient_sex`, `sex_cd`, `birth_sex` | Male |
| **Age Group** | `age_group`, `age_cat`, `age_bucket`, `age_band`, `age_category` | (largest group) |
| **Insurance** | `insurance`, `payer`, `insurance_type`, `coverage`, `payer_type`, `payer_category` | Commercial |
| **Language** | `language`, `primary_language`, `lang`, `language_cd`, `preferred_language` | English |
| **Disability** | `disability`, `disabled`, `disability_status`, `functional_status` | No |

### Custom Sensitive Attributes

Add your own attributes beyond auto-detection:

```python
# Add custom sensitive attribute
audit.add_sensitive_attribute(
    name="rural_urban",
    column="geographic_type",
    reference="Urban",
    clinical_justification="Rural patients may face access barriers"
)

# Add intersection for subgroup analysis
audit.add_intersection(["race", "sex"])  # Analyzes race × sex combinations
```

### Example Data Schema

```
patient_id | risk_score | readmit_30d | race    | sex    | insurance  | age_group
-----------|------------|-------------|---------|--------|------------|----------
P001       | 0.72       | 1           | White   | Female | Medicare   | 65-74
P002       | 0.31       | 0           | Black   | Male   | Medicaid   | 45-54
P003       | 0.85       | 1           | Hispanic| Female | Commercial | 35-44
...
```

---

## Fairness Visualizations

FairCareAI implements the **Van Calster et al. (2025)** 4-visualization framework for healthcare AI fairness assessment. Each visualization includes plain language explanations suitable for non-technical audiences.

### Overall Performance (4 Figures)

#### 1. AUROC (Discrimination)

**What it shows**: The X-axis shows false positive rate (0-100%), Y-axis shows true positive rate (0-100%). The area under the ROC curve (AUROC) measures the model's ability to separate high-risk from low-risk patients.

**Plain language**: Think of AUROC as the model's ability to rank patients correctly. A score of 0.5 means random guessing (coin flip), while 1.0 means perfect ranking. Healthcare standard: 0.7+ is acceptable, 0.8+ is strong.

**Why it matters**: Poor discrimination means the model cannot reliably identify who will and won't experience the outcome.

#### 2. Calibration Curve

**What it shows**: X-axis shows predicted risk from the model (0-100%), Y-axis shows actual observed rate (0-100%). Points should fall near the diagonal line for good calibration.

**Plain language**: If the model predicts 20% risk for a group, do about 20% actually experience the outcome? Points closer to the diagonal line mean more trustworthy risk estimates.

**Why it matters**: Under/over-estimating risk leads to wrong treatment decisions. Calibration slope should be 0.8-1.2 (ideal: 1.0).

#### 3. Brier Score (Overall Accuracy)

**What it shows**: A gauge showing Brier score from 0 (perfect) to 0.5 (poor). Lower is better.

**Plain language**: Think of it as the "error" in risk predictions. Score <0.15 = excellent, 0.15-0.25 = acceptable, >0.25 = needs improvement.

**Why it matters**: High Brier scores indicate the model's probability estimates are not reliable for clinical decisions.

#### 4. Classification Metrics at Threshold

**What it shows**: Bar chart showing Sensitivity (% of actual cases detected), Specificity (% correctly identified as negative), and PPV (% flagged who truly have condition).

**Plain language**: At the chosen risk threshold, these metrics show what happens to patients. Higher sensitivity means fewer missed cases, higher specificity means fewer false alarms.

**Why it matters**: Thresholds involve tradeoffs between catching all cases vs. avoiding unnecessary interventions.

### Subgroup Fairness (4 Figures per Attribute)

For each sensitive attribute (race, sex, insurance, etc.), FairCareAI generates:

#### 1. AUROC by Subgroup

**What it shows**: Bar chart comparing model discrimination across demographic groups. X-axis shows groups, Y-axis shows AUROC (0.5-1.0).

**Plain language**: Does the model perform equally well across all demographic groups? All bars should be similar height (difference <0.05 is ideal). Lower bars mean the model is less accurate for that group.

**Why it matters**: We want the model to work well for everyone, not just some groups.

#### 2. Sensitivity (TPR) by Subgroup

**What it shows**: Bar chart showing true positive rate for each group. X-axis shows groups, Y-axis shows percentage of actual cases correctly identified (0-100%).

**Plain language**: Of patients who actually develop the outcome, what percentage does the model correctly identify in each group? Large differences mean the model "misses" more cases in certain groups.

**Fairness goal**: Differences between groups should be <10 percentage points.

#### 3. False Positive Rate by Subgroup

**What it shows**: Bar chart showing FPR for each group. X-axis shows groups, Y-axis shows percentage incorrectly flagged as high-risk (0-50%).

**Plain language**: Of patients who DON'T have the outcome, what percentage are incorrectly flagged as high-risk in each group? Lower is better (fewer false alarms).

**Fairness concern**: Higher FPR means a group gets unnecessary interventions/worry.

#### 4. Selection Rate by Subgroup

**What it shows**: Bar chart showing what percentage of each group is flagged as "high-risk". X-axis shows groups, Y-axis shows selection rate (0-100%).

**Plain language**: This shows which groups the model identifies for intervention. Large differences may indicate disparate treatment even if clinically justified.

**Consider**: Should intervention rates differ by demographics?

### Typography and Accessibility

All visualizations follow publication-ready standards:

- **Minimum 14px font size** for all text (publication-compliant)
- **WCAG 2.1 AA contrast ratios** for text and UI elements
- **Colorblind-safe palettes** (tested with CVD simulation)
- **Alt text** for screen readers on all figures
- **Clear axis labels** with units and context
- **High-resolution export** suitable for print publication

---

## Fairness Metrics

FairCareAI supports multiple fairness definitions, acknowledging the **impossibility theorem** (Chouldechova 2017, Kleinberg et al. 2017) - no single metric is universally correct.

| Metric | Definition | Best For | Clinical Interpretation |
|--------|------------|----------|------------------------|
| **Demographic Parity** | Equal selection rates across groups:<br>P(Ŷ=1\|A=a) = P(Ŷ=1\|A=b) | Resource allocation | Equal proportion of each group receives intervention |
| **Equalized Odds** | Equal TPR and FPR across groups:<br>P(Ŷ=1\|Y=y,A=a) = P(Ŷ=1\|Y=y,A=b) | Intervention triggers | Equal error rates - no group systematically favored/disadvantaged |
| **Equal Opportunity** | Equal TPR across groups only:<br>P(Ŷ=1\|Y=1,A=a) = P(Ŷ=1\|Y=1,A=b) | Screening programs | Equal detection of true positives (disease cases) |
| **Predictive Parity** | Equal PPV across groups:<br>P(Y=1\|Ŷ=1,A=a) = P(Y=1\|Ŷ=1,A=b) | Risk communication | When flagged positive, same likelihood across groups |
| **Calibration** | Equal calibration curves across groups:<br>E[Y\|Ŷ=p,A=a] = p for all groups | Shared decision-making | Risk predictions are accurate within each group |

### Metric Selection Guidance

Use `audit.suggest_fairness_metric()` for context-specific recommendations based on your use case type.

**Example recommendations**:

- **Intervention Trigger** (care management, outreach) → Equalized Odds
- **Risk Communication** (patient counseling) → Calibration
- **Resource Allocation** (limited slots) → Demographic Parity
- **Screening** (disease detection) → Equal Opportunity

---

## Use Cases

### Intervention Trigger Models

Models that determine who receives an intervention (care management, outreach):
- **Recommended metric**: Equalized Odds
- **Key concern**: Equal access to beneficial interventions
- **Example**: Post-discharge care management referral

### Risk Communication

Models that communicate risk to patients/providers:
- **Recommended metric**: Calibration
- **Key concern**: Trustworthy probabilities for shared decisions
- **Example**: 10-year cardiovascular risk calculator

### Resource Allocation

Models that allocate limited resources:
- **Recommended metric**: Demographic Parity
- **Key concern**: Proportional distribution of resources
- **Example**: Care coordination slot assignment

### Screening Programs

Models used for disease screening:
- **Recommended metric**: Equal Opportunity
- **Key concern**: Equal detection rates for those with disease
- **Example**: Diabetic retinopathy screening

---

## API Reference

### Core Classes

#### `FairCareAudit`

Main orchestration class for conducting fairness audits.

```python
audit = FairCareAudit(
    data: pl.DataFrame | pd.DataFrame | str | Path,
    pred_col: str,
    target_col: str,
    config: FairnessConfig | None = None,
    threshold: float = 0.5
)
```

**Key Methods**:

```python
# Attribute management
audit.suggest_attributes() -> list[dict]
audit.accept_suggested_attributes(selections: list[int | str])
audit.add_sensitive_attribute(name, column, reference, categories, clinical_justification)
audit.add_intersection(attributes: list[str])

# Configuration
audit.suggest_fairness_metric() -> dict
audit.config.validate() -> list[str]

# Execution
results = audit.run(bootstrap_ci=True, n_bootstrap=1000) -> AuditResults
```

#### `FairnessConfig`

Configuration for fairness audit following CHAI RAIC framework.

```python
config = FairnessConfig(
    model_name: str,                              # Required
    model_version: str = "1.0.0",
    model_type: ModelType = ModelType.BINARY_CLASSIFIER,
    intended_use: str = "",                       # Recommended
    intended_population: str = "",                # Recommended
    primary_fairness_metric: FairnessMetric | None = None,  # Required
    fairness_justification: str = "",             # Required (CHAI AC1.CR93)
    use_case_type: UseCaseType | None = None,
    thresholds: dict = {...},                     # Configurable
    decision_thresholds: list[float] = [0.5],
)
```

**Thresholds** (configurable by health system):

```python
thresholds = {
    "min_subgroup_n": 100,                    # Minimum subgroup size
    "demographic_parity_ratio": (0.8, 1.25),  # EEOC 80% rule
    "equalized_odds_diff": 0.1,               # Max TPR/FPR difference
    "calibration_diff": 0.05,                 # Max calibration error
    "min_auroc": 0.65,                        # Minimum acceptable AUROC
    "max_missing_rate": 0.10,                 # Max missing data rate
}
```

#### `AuditResults`

Results container with visualization and export capabilities.

**Attributes**:
- `config`: FairnessConfig used for the audit
- `audit_id`: Unique identifier for this audit run
- `run_timestamp`: ISO timestamp when the audit executed
- `descriptive_stats`: Cohort characteristics (Table 1)
- `overall_performance`: TRIPOD+AI metrics
- `subgroup_performance`: Performance by demographic group
- `fairness_metrics`: Fairness metrics per attribute
- `intersectional`: Intersectional analysis results
- `flags`: List of metrics outside thresholds
- `governance_recommendation`: Summary statistics

**Visualization Methods**:

```python
# Executive summaries
results.plot_executive_summary()              # Single-page governance overview
results.plot_go_nogo_scorecard()              # Checklist-style scorecard
results.plot_fairness_dashboard()             # 4-panel comprehensive dashboard

# Overall performance
results.plot_discrimination()                 # ROC and PR curves
results.plot_overall_calibration()            # Calibration curve
results.plot_threshold_analysis()             # Threshold sensitivity
results.plot_decision_curve()                 # Decision curve analysis

# Subgroup analysis
results.plot_subgroup_performance(metric="auroc")  # Subgroup comparison
results.plot_calibration(by="race")           # Stratified calibration
```

**Export Methods**:

```python
# Data Scientist output (full technical)
results.to_html("report.html")
results.to_pdf("report.pdf")

# Governance output (streamlined)
results.to_html("gov.html", persona="governance")
results.to_pdf("gov.pdf", persona="governance")
results.to_governance_html("gov.html")        # Convenience method
results.to_governance_pdf("gov.pdf")          # Convenience method

# PowerPoint (always governance-focused)
results.to_pptx("deck.pptx")

# JSON for programmatic use
results.to_json("metrics.json")

# Open in browser
results.to_html("report.html", open_browser=True)
```

Reports include an **Audit Trail** section with audit ID, audit run timestamp, report generated time, model/version, and configuration context. JSON exports include `audit_metadata` with `audit_id` and `run_timestamp`.

### Enums

#### `OutputPersona`

```python
from faircareai.core.config import OutputPersona

OutputPersona.DATA_SCIENTIST  # Full technical output
OutputPersona.GOVERNANCE      # Streamlined 3-5 page output
```

#### `FairnessMetric`

```python
from faircareai.core.config import FairnessMetric

FairnessMetric.DEMOGRAPHIC_PARITY
FairnessMetric.EQUALIZED_ODDS
FairnessMetric.EQUAL_OPPORTUNITY
FairnessMetric.PREDICTIVE_PARITY
FairnessMetric.CALIBRATION
FairnessMetric.INDIVIDUAL_FAIRNESS
```

#### `UseCaseType`

```python
from faircareai.core.config import UseCaseType

UseCaseType.INTERVENTION_TRIGGER
UseCaseType.RISK_COMMUNICATION
UseCaseType.RESOURCE_ALLOCATION
UseCaseType.SCREENING
UseCaseType.DIAGNOSIS_SUPPORT
```

---

## Accessibility Features

FairCareAI is designed with accessibility as a core principle, not an afterthought.

### WCAG 2.1 AA Compliance

All visualizations meet or exceed WCAG 2.1 Level AA standards:

- **Text Contrast**: Minimum 4.5:1 contrast ratio for normal text, 3:1 for large text
- **Color Independence**: Information never conveyed by color alone (icons, patterns, labels used)
- **Font Size**: Minimum 14px for all text in governance reports, 12px for data scientist reports
- **Zoom Support**: Layouts remain usable at 200% zoom
- **Screen Reader Support**: Alt text provided for all figures via Plotly metadata

### Colorblind-Safe Palettes

FairCareAI uses colorblind-safe palettes tested with CVD (color vision deficiency) simulation:

- **Primary Palette**: Blue (#1f77b4) and Orange (#ff7f0e) - distinguishable for all CVD types
- **Status Colors**: Green (success), Yellow (warning), Red (error) - supplemented with icons
- **Subgroup Colors**: Carefully selected categorical palette avoiding red-green confusion

Tested with:
- Protanopia (red-blind)
- Deuteranopia (green-blind)
- Tritanopia (blue-blind)
- Monochromacy (grayscale)

### Publication-Ready Typography

Governance reports follow publication style guidelines:

- **Body text**: 14px (minimum)
- **Headings**: 16-20px, bold
- **Chart labels**: 14px
- **Annotations**: 14px
- **Font family**: Sans-serif (Arial, Helvetica, system default)

### Plain Language

All governance outputs use plain language principles:

- **Active voice**: "The model predicts..." not "Predictions are made..."
- **Short sentences**: <25 words average
- **Defined jargon**: Technical terms explained on first use
- **Concrete examples**: "20% of patients" not "0.2 probability"
- **Clear headings**: Descriptive section titles

---

## Governance Compliance (CHAI RAIC)

FairCareAI is aligned with the [Coalition for Health AI (CHAI) RAIC Framework](https://www.coalitionforhealthai.org/) Checkpoint 1: Fairness Assessment.

### CHAI Assurance Criteria Mapping

| CHAI Criterion | FairCareAI Feature | Implementation |
|----------------|-------------------|----------------|
| **AC1.CR1** | Model identification | `model_name`, `model_version`, `intended_use` in config |
| **AC1.CR3** | Intended population | `intended_population` field with validation |
| **AC1.CR68** | Data quality checks | Missing data rate flags, minimum subgroup size checks |
| **AC1.CR82** | Sample size adequacy | Configurable `min_subgroup_n` threshold with warnings |
| **AC1.CR92** | Fairness metric selection | `primary_fairness_metric` with context-based recommendations |
| **AC1.CR93** | Fairness justification | Required `fairness_justification` field (blocks run if missing) |
| **AC1.CR100** | Out-of-scope documentation | `out_of_scope` list in config |

### Governance Report Sections

Generated reports include 7 CHAI-aligned sections:

1. **Executive Summary** - Go/no-go advisory with key findings
2. **Descriptive Statistics** - Cohort characteristics (Table 1)
3. **Overall Performance** - AUROC, AUPRC, calibration metrics (TRIPOD+AI)
4. **Subgroup Performance** - Performance by demographic group
5. **Fairness Assessment** - Disparity analysis with confidence intervals
6. **Limitations & Flags** - Warnings and considerations
7. **Governance Decision Block** - Sign-off section for stakeholders

### Threshold Configuration

Default thresholds are evidence-based starting points. Health systems should adjust based on context:

```python
config = FairnessConfig(
    model_name="My Model",
    thresholds={
        "min_subgroup_n": 100,                    # Adjust based on power requirements
        "demographic_parity_ratio": (0.8, 1.25),  # EEOC 80% rule
        "equalized_odds_diff": 0.1,               # Adjust based on clinical impact
        "calibration_diff": 0.05,                 # Adjust based on decision context
        "min_auroc": 0.65,                        # Adjust based on use case
        "max_missing_rate": 0.10,                 # Adjust based on data quality standards
    }
)
```

---

## Interactive Dashboard

### Command-Line Interface (CLI)

FairCareAI provides a CLI for running audits directly from the terminal:

```bash
# Audit a Parquet file (recommended for large datasets)
faircareai audit predictions.parquet -p risk_score -t outcome -o report.html

# Audit a CSV file
faircareai audit patient_data.csv -p risk_score -t readmit_30d -a race -a sex

# Specify output format explicitly
faircareai audit data.parquet -p prob -t label --format html --output fairness_report.html

# Generate governance PDF report
faircareai audit data.csv -p risk_score -t outcome --persona governance --format pdf -o governance.pdf

# Run with custom threshold
faircareai audit predictions.parquet -p risk_score -t outcome --threshold 0.3
```

**CLI Options**:
| Option | Description | Example |
|--------|-------------|---------|
| `-p`, `--pred-col` | Prediction column name | `-p risk_score` |
| `-t`, `--target-col` | Target/outcome column name | `-t readmit_30d` |
| `-a`, `--attribute` | Sensitive attribute (repeatable) | `-a race -a sex` |
| `-o`, `--output` | Output file path | `-o report.html` |
| `--format` | Output format (html, pdf, json) | `--format pdf` |
| `--persona` | Output persona (data_scientist, governance) | `--persona governance` |
| `--threshold` | Decision threshold (0-1) | `--threshold 0.3` |

### Streamlit Dashboard

Launch the interactive Streamlit dashboard for visual analysis:

```python
import faircareai
faircareai.launch()
```

Or from the command line:

```bash
faircareai dashboard
```

**Dashboard Features**:
- Upload data via file or connect to database
- Interactive attribute selection
- Real-time visualization updates
- Threshold adjustment sliders
- Export reports directly from UI
- Side-by-side persona comparison

---

## Configuration

### Complete Example

```python
from faircareai import FairCareAudit, FairnessConfig
from faircareai.core.config import FairnessMetric, UseCaseType, ModelType

audit = FairCareAudit(
    data="predictions.parquet",
    pred_col="risk_score",
    target_col="readmit_30d",
    threshold=0.3  # Adjust based on operating point
)

# Auto-detect and accept attributes
audit.suggest_attributes()
audit.accept_suggested_attributes([1, 2, 3])  # race, sex, insurance

# Or manually configure
audit.add_sensitive_attribute(
    name="language",
    column="primary_language",
    reference="English",
    clinical_justification="Language barriers affect care coordination"
)

# Configure audit
audit.config = FairnessConfig(
    # Model identity (required)
    model_name="30-Day Readmission Risk Model",
    model_version="2.1.0",
    model_type=ModelType.BINARY_CLASSIFIER,

    # Intended use (CHAI recommended)
    intended_use="Identify high-risk patients for care management outreach within 24h of discharge",
    intended_population="Adult patients (18+) discharged from medicine or surgery services",
    out_of_scope=[
        "Pediatric patients",
        "Psychiatric admissions",
        "Patients in hospice care"
    ],

    # Fairness configuration (CHAI required)
    primary_fairness_metric=FairnessMetric.EQUALIZED_ODDS,
    fairness_justification=(
        "This model triggers a beneficial intervention (care management). "
        "Equalized odds ensures equal true positive rates (benefit access) "
        "and false positive rates (intervention burden) across demographic groups. "
        "This prevents systematic advantage/disadvantage in accessing care coordination."
    ),
    use_case_type=UseCaseType.INTERVENTION_TRIGGER,

    # Thresholds (organization-specific)
    thresholds={
        "min_subgroup_n": 150,  # Higher threshold for external validity
        "demographic_parity_ratio": (0.75, 1.33),  # Relaxed for intervention
        "equalized_odds_diff": 0.08,  # Stricter due to high-stakes decision
        "calibration_diff": 0.05,
        "min_auroc": 0.70,
        "max_missing_rate": 0.05,  # Stricter data quality
    },

    # Report metadata
    organization_name="Example Health System",
    report_date="2025-01-15",
    include_chai_mapping=True,
)

# Run audit with bootstrap confidence intervals
results = audit.run(bootstrap_ci=True, n_bootstrap=1000)

# Export both personas
results.to_pdf("technical_validation.pdf")  # Data scientist
results.to_governance_pdf("governance_review.pdf")  # Governance
results.to_pptx("committee_presentation.pptx")  # PowerPoint
```

---

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) and ensure:

1. **Tests pass**: Run `pytest tests/` before submitting
2. **Code follows style**: Use `ruff check src/` and `mypy src/`
3. **Documentation updated**: Update README and docstrings
4. **Scientific claims cited**: Include references for methodology

### Development Setup

```bash
# Clone repository
git clone https://github.com/sajor2000/faircare_package.git
cd faircareai

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Type checking
mypy src/faircareai

# Linting
ruff check src/
```

### Areas for Contribution

- Additional fairness metrics (individual fairness, counterfactual fairness)
- Support for multiclass classification and regression
- Integration with MLOps platforms (MLflow, Weights & Biases)
- Additional export formats (Word, LaTeX)
- Translations for internationalization

---

## Citation

If you use FairCareAI in your research or clinical implementation, please cite:

```bibtex
@software{faircareai,
  title = {FairCareAI: Healthcare AI Fairness Auditing},
  author = {FairCareAI Contributors},
  year = {2024},
  url = {https://github.com/sajor2000/faircare_package},
  version = {0.2.1},
  note = {Python package for auditing ML fairness in healthcare}
}
```

---

## References

### Fairness Theory

- **Chouldechova, A.** (2017). Fair prediction with disparate impact: A study of bias in recidivism prediction instruments. *Big Data*, 5(2), 153-163.
- **Hardt, M., Price, E., & Srebro, N.** (2016). Equality of opportunity in supervised learning. *Advances in Neural Information Processing Systems*, 29.
- **Kleinberg, J., Mullainathan, S., & Raghavan, M.** (2017). Inherent trade-offs in the fair determination of risk scores. *Proceedings of the 8th Innovations in Theoretical Computer Science Conference*.

### Clinical ML Reporting

- **Collins, G.S., Dhiman, P., Andaur Navarro, C.L., et al.** (2024). Protocol for development of a reporting guideline (TRIPOD-AI) and risk of bias tool (PROBAST-AI) for diagnostic and prognostic prediction model studies based on artificial intelligence. *BMJ Open*, 11(7), e048008.
- **Van Calster, B., Collins, G.S., Vickers, A.J., et al.** (2025). Evaluation of performance measures in predictive artificial intelligence models to support medical decisions: overview and guidance. *The Lancet Digital Health*. https://doi.org/10.1016/j.landig.2025.100916

### Governance Frameworks

- **Coalition for Health AI (CHAI)**. (2024). RAIC Framework: Responsible AI in Healthcare. Retrieved from https://www.coalitionforhealthai.org/
- **FDA.** (2021). Artificial Intelligence/Machine Learning (AI/ML)-Based Software as a Medical Device (SaMD) Action Plan.

### Healthcare Disparities

- **Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S.** (2019). Dissecting racial bias in an algorithm used to manage the health of populations. *Science*, 366(6464), 447-453.
- **Rajkomar, A., Hardt, M., Howell, M.D., Corrado, G., & Chin, M.H.** (2018). Ensuring fairness in machine learning to advance health equity. *Annals of Internal Medicine*, 169(12), 866-872.

---

## Support

- **Documentation**: See [docs/](docs/) folder for detailed guides
  - [USAGE.md](docs/USAGE.md) - Quickstart and end-to-end usage
  - [METHODOLOGY.md](docs/METHODOLOGY.md) - Scientific foundation and fairness theory
  - [PDF_SETUP_GUIDE.md](docs/PDF_SETUP_GUIDE.md) - PDF export setup
  - [ARCHITECTURE.md](ARCHITECTURE.md) - System design and data flow
  - [CONTRIBUTING.md](CONTRIBUTING.md) - Development guidelines
  - [Reports & audits](docs/reports/) - Internal reviews and verification artifacts
- **Issues**: [GitHub Issues](https://github.com/sajor2000/faircare_package/issues)
- **Discussions**: [GitHub Discussions](https://github.com/sajor2000/faircare_package/discussions)
- **Code of Conduct**: See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- **Security**: See [SECURITY.md](SECURITY.md)

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Disclaimer

FairCareAI provides CHAI-grounded guidance for fairness auditing. All outputs are **ADVISORY**. Final deployment decisions rest with the health system, clinical governance committees, and regulatory authorities who understand local context, patient populations, and organizational values.

This software is provided "as is" without warranty of any kind. Healthcare organizations are responsible for validating all outputs in their specific clinical context before deployment.
