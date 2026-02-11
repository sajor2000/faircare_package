# FairCareAI Usage Guide

Complete guide to using FairCareAI for healthcare AI fairness auditing.

---

## Installation

### Basic Installation

```bash
pip install faircareai
```

### With Export Capabilities (PDF/PPTX)

```bash
pip install "faircareai[export]"
```

### Development Installation

```bash
git clone https://github.com/sajor2000/faircare_package.git
cd faircareai
pip install -e ".[dev]"
```

### Verify Installation

```bash
faircareai version
```

---

## Quick Start (5 Minutes)

```python
from faircareai import FairCareAudit, FairnessConfig
from faircareai.core.config import FairnessMetric, UseCaseType

# 1. Load your model predictions
audit = FairCareAudit(
    data="predictions.parquet",  # or DataFrame
    pred_col="risk_score",
    target_col="outcome"
)

# 2. Review suggested sensitive attributes
audit.suggest_attributes()

# 3. Accept the ones you want to analyze
audit.accept_suggested_attributes([1, 2])  # 1-indexed

# 4. Configure the audit
audit.config = FairnessConfig(
    model_name="Readmission Risk v2",
    primary_fairness_metric=FairnessMetric.EQUALIZED_ODDS,
    fairness_justification="Model triggers intervention...",
    use_case_type=UseCaseType.INTERVENTION_TRIGGER
)

# 5. Run and export
results = audit.run()
results.to_html("report.html")
```

---

## Output Personas

FairCareAI supports two output personas tailored for different audiences:

### Data Scientist Persona (Default)

Full technical reports for model validation teams (15-20+ pages).

**Includes:**
- Complete statistical methodology
- Bootstrap confidence intervals
- All fairness metrics with CIs
- Technical visualizations (ROC, PR curves, DCA)
- Detailed subgroup analysis
- Raw data tables

**Usage:**
```python
# Default - full technical output
results.to_html("report.html")
results.to_pdf("report.pdf")

# Explicit persona parameter
results.to_html("report.html", persona="data_scientist")
```

### Governance Persona

Streamlined 3-5 page reports for governance committees and clinical leadership.

**Includes:**
- Executive summary with traffic light status
- 4 key performance figures (Van Calster methodology)
- 4 subgroup fairness figures per attribute
- Plain language explanations on every figure
- Go/No-Go checklist
- Advisory disclaimer banner

**Usage:**
```python
# Governance-focused output
results.to_html("gov_report.html", persona="governance")
results.to_pdf("gov_report.pdf", persona="governance")

# Convenience shortcuts
results.to_governance_html("gov_report.html")
results.to_governance_pdf("gov_report.pdf")

# PowerPoint (always governance-focused)
results.to_pptx("presentation.pptx")
```

### Comparison

| Feature | Data Scientist | Governance |
|---------|---------------|------------|
| Page count | 15-20+ | 3-5 |
| Technical detail | Full | Simplified |
| Confidence intervals | All metrics | Key metrics only |
| Plain language | Minimal | Extensive |
| Figure explanations | Technical | Plain language |
| Decision support | Raw metrics | Traffic light status |
| Use case | Model validation | Committee review |

---

## Python API Reference

### FairCareAudit

Main class for conducting fairness audits.

#### Constructor

```python
FairCareAudit(
    data: pl.DataFrame | pd.DataFrame | str | Path,
    pred_col: str,
    target_col: str,
    config: FairnessConfig | None = None,
    threshold: float = 0.5
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | DataFrame or path | Model predictions with outcomes. Accepts Polars, pandas, or path to parquet/csv. |
| `pred_col` | str | Column name for predicted probabilities (0-1). |
| `target_col` | str | Column name for actual outcomes (0 or 1). |
| `config` | FairnessConfig | Audit configuration (optional, can set later). |
| `threshold` | float | Decision threshold for classification (default: 0.5). |

**Example:**

```python
# From DataFrame
audit = FairCareAudit(df, pred_col="risk", target_col="outcome")

# From file
audit = FairCareAudit("predictions.parquet", pred_col="risk", target_col="outcome")

# With config
audit = FairCareAudit(df, pred_col="risk", target_col="outcome",
                       config=my_config, threshold=0.3)
```

#### Methods

##### suggest_attributes()

```python
suggest_attributes(display: bool = True) -> list[dict]
```

Auto-detect sensitive attributes from column names.

**Parameters:**
- `display`: Print suggestions to console (default: True)

**Returns:** List of suggestion dictionaries with keys: `suggested_name`, `detected_column`, `suggested_reference`, `clinical_justification`

**Example:**

```python
suggestions = audit.suggest_attributes()
# Output:
# 1. race (column: race_ethnicity) - Reference: White
# 2. sex (column: patient_sex) - Reference: Male
```

##### accept_suggested_attributes()

```python
accept_suggested_attributes(
    selections: list[int | str],
    modify: dict | None = None
) -> FairCareAudit
```

Accept auto-detected attributes for analysis.

**Parameters:**
- `selections`: List of indices (1-based) or names to accept
- `modify`: Override reference groups or other settings

**Returns:** self (for method chaining)

**Example:**

```python
# Accept by index
audit.accept_suggested_attributes([1, 2, 3])

# Accept by name
audit.accept_suggested_attributes(["race", "sex"])

# Override reference group
audit.accept_suggested_attributes([1], modify={"race": {"reference": "Black"}})
```

##### add_sensitive_attribute()

```python
add_sensitive_attribute(
    name: str,
    column: str | None = None,
    reference: str | None = None,
    categories: list[str] | None = None,
    clinical_justification: str | None = None
) -> FairCareAudit
```

Add a custom sensitive attribute.

**Parameters:**
- `name`: Display name (e.g., "race", "sex")
- `column`: Column name in data (defaults to name)
- `reference`: Reference group for comparisons
- `categories`: Expected category values
- `clinical_justification`: CHAI-required justification

**Returns:** self (for method chaining)

**Example:**

```python
audit.add_sensitive_attribute(
    name="rural_urban",
    column="geographic_type",
    reference="Urban",
    clinical_justification="Rural patients may face access barriers"
)
```

##### add_intersection()

```python
add_intersection(attributes: list[str]) -> FairCareAudit
```

Add intersectional analysis (e.g., race x sex).

**Parameters:**
- `attributes`: List of attribute names to intersect

**Returns:** self (for method chaining)

**Example:**

```python
audit.add_intersection(["race", "sex"])
```

##### suggest_fairness_metric()

```python
suggest_fairness_metric() -> dict
```

Get fairness metric recommendation based on use case.

**Returns:** Dictionary with recommended metric and context

**Example:**

```python
audit.config.use_case_type = UseCaseType.INTERVENTION_TRIGGER
recommendation = audit.suggest_fairness_metric()
print(recommendation["recommended_metric"])  # "equalized_odds"
```

##### run()

```python
run(
    bootstrap_ci: bool = True,
    n_bootstrap: int = 1000
) -> AuditResults
```

Execute the fairness audit.

**Parameters:**
- `bootstrap_ci`: Calculate bootstrap confidence intervals
- `n_bootstrap`: Number of bootstrap iterations

**Returns:** AuditResults with all metrics and visualization methods

**Example:**

```python
results = audit.run()
results = audit.run(bootstrap_ci=False)  # Faster, no CIs
results = audit.run(n_bootstrap=500)     # Fewer iterations
```

---

### FairnessConfig

Configuration dataclass for fairness audit.

#### Constructor

```python
FairnessConfig(
    model_name: str,
    model_version: str = "1.0.0",
    model_type: ModelType = ModelType.BINARY_CLASSIFIER,
    intended_use: str = "",
    intended_population: str = "",
    out_of_scope: list[str] = [],
    primary_fairness_metric: FairnessMetric | None = None,
    fairness_justification: str = "",
    use_case_type: UseCaseType | None = None,
    thresholds: dict = {...},
    decision_thresholds: list[float] = [0.5],
    include_chai_mapping: bool = True,
    organization_name: str = "",
    report_date: str | None = None
)
```

#### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `model_name` | str | Display name of the model |
| `primary_fairness_metric` | FairnessMetric | Selected fairness metric (required for run) |
| `fairness_justification` | str | Why this metric was chosen (required for run) |

#### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_version` | str | "1.0.0" | Model version identifier |
| `model_type` | ModelType | BINARY_CLASSIFIER | Type of ML model |
| `intended_use` | str | "" | Clinical use description |
| `intended_population` | str | "" | Target patient population |
| `out_of_scope` | list[str] | [] | Out-of-scope uses |
| `use_case_type` | UseCaseType | None | Clinical use case category |
| `thresholds` | dict | {...} | Fairness thresholds |
| `decision_thresholds` | list[float] | [0.5] | Classification thresholds |
| `organization_name` | str | "" | Health system name |

#### Thresholds Dictionary

```python
thresholds = {
    "min_subgroup_n": 100,           # Minimum subgroup size
    "demographic_parity_ratio": (0.8, 1.25),  # Selection rate range
    "equalized_odds_diff": 0.1,      # Max TPR/FPR difference
    "calibration_diff": 0.05,        # Max calibration error
    "min_auroc": 0.65,               # Minimum AUROC
    "max_missing_rate": 0.10,        # Max missing data rate
}
```

#### Methods

##### validate()

```python
validate() -> list[str]
```

Validate configuration and return issues.

**Returns:** List of "ERROR:" or "WARNING:" messages

##### has_errors()

```python
has_errors() -> bool
```

Check if configuration has blocking errors.

##### get_threshold()

```python
get_threshold(key: str, default: float | None = None) -> float | None
```

Get threshold value with optional default.

#### Example

```python
config = FairnessConfig(
    model_name="Readmission Risk v2",
    model_version="2.0.0",
    intended_use="Trigger care management outreach",
    intended_population="Adult patients discharged from acute care",
    primary_fairness_metric=FairnessMetric.EQUALIZED_ODDS,
    fairness_justification=(
        "Model triggers intervention. Equalized odds ensures "
        "equal TPR/FPR, preventing differential access."
    ),
    use_case_type=UseCaseType.INTERVENTION_TRIGGER,
    thresholds={
        "min_subgroup_n": 200,  # Custom threshold
        "equalized_odds_diff": 0.05,  # Stricter
    }
)

# Validate
issues = config.validate()
if config.has_errors():
    print("Fix these:", [i for i in issues if i.startswith("ERROR")])
```

---

### AuditResults

Container for audit results with export and visualization methods.

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `config` | FairnessConfig | Configuration used |
| `audit_id` | str | Unique identifier for this audit run |
| `run_timestamp` | str \| None | ISO timestamp when the audit executed |
| `descriptive_stats` | dict | Section 1: Cohort summary |
| `overall_performance` | dict | Section 2: TRIPOD+AI metrics |
| `subgroup_performance` | dict | Section 3: By-group performance |
| `fairness_metrics` | dict | Section 4: Fairness metrics |
| `intersectional` | dict | Intersectional analysis |
| `flags` | list[dict] | Metrics outside thresholds |
| `governance_recommendation` | dict | Section 7: Summary |

#### Visualization Methods

##### plot_executive_summary()

```python
plot_executive_summary() -> plotly.graph_objects.Figure
```

Single-page governance summary with traffic light status.

##### plot_discrimination()

```python
plot_discrimination() -> plotly.graph_objects.Figure
```

ROC and Precision-Recall curves (TRIPOD+AI 2.1).

##### plot_overall_calibration()

```python
plot_overall_calibration() -> plotly.graph_objects.Figure
```

Calibration curve for overall model (TRIPOD+AI 2.2).

##### plot_calibration()

```python
plot_calibration(by: str | None = None) -> plotly.graph_objects.Figure
```

Calibration curve(s), optionally stratified by attribute.

##### plot_threshold_analysis()

```python
plot_threshold_analysis(selected_threshold: float | None = None) -> plotly.graph_objects.Figure
```

Interactive threshold sensitivity analysis.

##### plot_decision_curve()

```python
plot_decision_curve() -> plotly.graph_objects.Figure
```

Decision Curve Analysis for clinical utility.

##### plot_fairness_dashboard()

```python
plot_fairness_dashboard() -> plotly.graph_objects.Figure
```

Comprehensive 4-panel fairness dashboard.

##### plot_subgroup_performance()

```python
plot_subgroup_performance(metric: str = "auroc") -> plotly.graph_objects.Figure
```

Subgroup comparison for specified metric.

##### plot_go_nogo_scorecard()

```python
plot_go_nogo_scorecard() -> plotly.graph_objects.Figure
```

Checklist-style scorecard for governance.

#### Export Methods

##### to_html()

```python
to_html(
    path: str | Path,
    open_browser: bool = False,
    persona: str = "data_scientist"
) -> Path
```

Export interactive HTML report.

Reports include an **Audit Trail** section with audit ID, audit run timestamp, report generated time, model/version, and configuration context.

**Parameters:**
- `path`: Output file path
- `open_browser`: Open in browser after export
- `persona`: "data_scientist" (default) or "governance"

##### to_governance_html()

```python
to_governance_html(path: str | Path, open_browser: bool = False) -> Path
```

Convenience method for governance HTML export.

##### to_pdf()

```python
to_pdf(path: str | Path, persona: str = "data_scientist") -> Path
```

Export PDF report (requires `[export]`).

##### to_governance_pdf()

```python
to_governance_pdf(path: str | Path) -> Path
```

Convenience method for governance PDF export.

##### to_pptx()

```python
to_pptx(path: str | Path) -> Path
```

Export PowerPoint deck (requires `[export]`). Always uses governance persona.

##### to_json()

```python
to_json(path: str | Path) -> Path
```

Export metrics as JSON.

JSON includes `audit_metadata` with `audit_id` and `run_timestamp`.

#### Other Methods

##### summary()

```python
summary() -> str
```

Get formatted text summary.

##### print_table1()

```python
print_table1() -> str
```

Print Table 1 descriptive statistics.

##### get_table1_dataframe()

```python
get_table1_dataframe() -> pl.DataFrame
```

Get Table 1 as DataFrame.

#### Example

```python
results = audit.run()

# View summary
print(results.summary())

# Visualize
fig = results.plot_executive_summary()
fig.show()

# Export all formats
results.to_html("report.html", open_browser=True)
results.to_pdf("report.pdf")
results.to_pptx("deck.pptx")
results.to_json("metrics.json")
```

---

### Enums

#### FairnessMetric

```python
from faircareai.core.config import FairnessMetric

FairnessMetric.DEMOGRAPHIC_PARITY   # Equal selection rates
FairnessMetric.EQUALIZED_ODDS       # Equal TPR and FPR
FairnessMetric.EQUAL_OPPORTUNITY    # Equal TPR only
FairnessMetric.PREDICTIVE_PARITY    # Equal PPV
FairnessMetric.CALIBRATION          # Equal calibration
FairnessMetric.INDIVIDUAL_FAIRNESS  # Similar → similar
```

#### UseCaseType

```python
from faircareai.core.config import UseCaseType

UseCaseType.INTERVENTION_TRIGGER   # Triggers intervention
UseCaseType.RISK_COMMUNICATION     # Communicates risk
UseCaseType.RESOURCE_ALLOCATION    # Allocates resources
UseCaseType.SCREENING              # Disease screening
UseCaseType.DIAGNOSIS_SUPPORT      # Diagnosis support
```

#### ModelType

```python
from faircareai.core.config import ModelType

ModelType.BINARY_CLASSIFIER  # Binary classification
ModelType.RISK_SCORE         # Risk score
ModelType.MULTICLASS         # Multiclass (future)
```

---

### SensitiveAttribute

Dataclass for sensitive attribute configuration.

```python
from faircareai.core.config import SensitiveAttribute

attr = SensitiveAttribute(
    name="race",
    column="race_ethnicity",
    reference="White",
    categories=["White", "Black", "Hispanic", "Asian", "Other"],
    attr_type="categorical",
    is_protected=True,
    clinical_justification="Required for health equity analysis"
)
```

---

## CLI Reference

### faircareai dashboard

Launch the interactive Streamlit dashboard.

```bash
faircareai dashboard [OPTIONS]
```

**Options:**
- `--port`: Port number (default: 8501)

**Example:**

```bash
faircareai dashboard
faircareai dashboard --port 8080
```

### faircareai audit

Run audit from command line. Supports both **Parquet** and **CSV** input files.

```bash
faircareai audit DATA_PATH [OPTIONS]
```

**Arguments:**
- `DATA_PATH`: Path to predictions file (`.parquet` or `.csv`)

**Options:**

| Option | Description | Example |
|--------|-------------|---------|
| `-p`, `--pred-col` | Prediction column name (required) | `-p risk_score` |
| `-t`, `--target-col` | Target/outcome column name (required) | `-t readmit_30d` |
| `-a`, `--attribute` | Sensitive attribute (repeatable) | `-a race -a sex` |
| `-o`, `--output` | Output file path | `-o report.html` |
| `--format` | Output format (html, pdf, json) | `--format pdf` |
| `--persona` | Output persona (data_scientist, governance) | `--persona governance` |
| `--threshold` | Decision threshold (0-1) | `--threshold 0.3` |
| `--model-name` | Model display name | `--model-name "Risk v2"` |

**Examples:**

```bash
# Audit a Parquet file (recommended for large datasets)
faircareai audit predictions.parquet -p risk_score -t outcome -o report.html

# Audit a CSV file
faircareai audit patient_data.csv -p risk_score -t readmit_30d -a race -a sex

# Generate governance PDF report from CSV
faircareai audit data.csv -p risk_score -t outcome --persona governance --format pdf -o governance.pdf

# Full example with all options
faircareai audit predictions.parquet \
  -p risk_score \
  -t outcome \
  -o report.html \
  --model-name "Readmission Risk v2" \
  -a race \
  -a sex \
  --threshold 0.3

# Run audit on Parquet with custom threshold
faircareai audit predictions.parquet -p risk_score -t outcome --threshold 0.3
```

### faircareai info

Display fairness metrics guide.

```bash
faircareai info
```

### faircareai version

Show version information.

```bash
faircareai version
```

---

## Dashboard Guide

### Page 1: Data Upload

1. **Upload File**: Drag & drop parquet/csv or use demo data
2. **Column Selection**: Select prediction and target columns
3. **Preview**: View first rows of data
4. **Validation**: Automatic data validation with warnings

### Page 2: Analysis Configuration

1. **Sensitive Attributes**: Select from auto-detected or add custom
2. **Fairness Metric**: Choose based on use case guidance
3. **Thresholds**: Adjust fairness thresholds as needed
4. **Run Audit**: Execute analysis with progress indicator

### Page 3: Governance Report

1. **Executive Summary**: Traffic light status
2. **Performance Metrics**: AUROC, calibration, etc.
3. **Fairness Analysis**: Disparity charts with CIs
4. **Flags & Warnings**: Items outside thresholds
5. **Export**: Download HTML/PDF/PPTX

### Page 4: Settings

1. **Export Format**: Select output format
2. **Bootstrap Options**: Configure CI computation
3. **Theme**: Toggle light/dark mode
4. **About**: Version and documentation links

---

## Common Patterns

### Loading Different Data Formats

FairCareAI accepts multiple input formats:

| Format | Extension | Best For |
|--------|-----------|----------|
| **Parquet** | `.parquet` | Large datasets (>100k rows), faster loading |
| **CSV** | `.csv` | Simple data exchange, human-readable |
| **Polars DataFrame** | - | In-memory analysis with Polars |
| **Pandas DataFrame** | - | In-memory analysis with pandas |

```python
# Option 1: Parquet file (recommended for large datasets)
audit = FairCareAudit("predictions.parquet", pred_col="risk", target_col="outcome")

# Option 2: CSV file
audit = FairCareAudit("patient_data.csv", pred_col="risk", target_col="outcome")

# Option 3: Polars DataFrame
import polars as pl
df = pl.read_parquet("data.parquet")
audit = FairCareAudit(df, pred_col="risk", target_col="outcome")

# Option 4: Pandas DataFrame
import pandas as pd
df = pd.read_csv("data.csv")
audit = FairCareAudit(df, pred_col="risk", target_col="outcome")

# Saving predictions as Parquet (for future use)
import polars as pl
df.write_parquet("predictions.parquet")
```

### Custom Fairness Thresholds

```python
config = FairnessConfig(
    model_name="My Model",
    thresholds={
        "min_subgroup_n": 50,      # Smaller minimum
        "equalized_odds_diff": 0.15,  # More lenient
        "demographic_parity_ratio": (0.7, 1.4),  # Wider range
    }
)
```

### Intersectional Analysis

```python
audit.add_sensitive_attribute("race", column="race_eth", reference="White")
audit.add_sensitive_attribute("sex", column="patient_sex", reference="Male")
audit.add_intersection(["race", "sex"])

results = audit.run()
print(results.intersectional)  # race x sex combinations
```

### Multiple Thresholds

```python
for threshold in [0.3, 0.5, 0.7]:
    audit = FairCareAudit(df, pred_col="risk", target_col="outcome",
                          threshold=threshold)
    # ... configure ...
    results = audit.run()
    results.to_html(f"report_thresh_{threshold}.html")
```

### Batch Processing

```python
models = ["model_a.parquet", "model_b.parquet", "model_c.parquet"]

for model_path in models:
    audit = FairCareAudit(model_path, pred_col="risk", target_col="outcome")
    audit.accept_suggested_attributes([1, 2])
    audit.config = config
    results = audit.run()
    results.to_html(f"{Path(model_path).stem}_report.html")
```

---

## Troubleshooting

### Common Errors

#### "primary_fairness_metric is required"

```python
# Fix: Set the metric before running
audit.config.primary_fairness_metric = FairnessMetric.EQUALIZED_ODDS
```

#### "fairness_justification is required"

```python
# Fix: Provide justification
audit.config.fairness_justification = "Model triggers intervention..."
```

#### "At least one sensitive attribute required"

```python
# Fix: Accept or add attributes
audit.accept_suggested_attributes([1])
# or
audit.add_sensitive_attribute("race", column="race_col")
```

#### "Missing required columns"

```python
# Fix: Check column names match your data
print(df.columns)  # Verify pred_col and target_col exist
```

#### "Predictions must be probabilities in [0, 1]"

```python
# Fix: Apply sigmoid if using logits
import scipy.special
df = df.with_columns(pl.col("logits").map_elements(scipy.special.expit).alias("prob"))
```

### Performance Issues

#### Slow Bootstrap CI

```python
# Reduce iterations
results = audit.run(n_bootstrap=500)

# Or disable
results = audit.run(bootstrap_ci=False)
```

#### Large Dataset Memory

```python
# Use Polars lazy evaluation
df = pl.scan_parquet("large_data.parquet").collect()

# Or sample
df = df.sample(n=100000, seed=42)
```

### Export Issues

#### PDF Export Fails

```bash
# Install export dependencies
pip install "faircareai[export]"

# On macOS, may need:
brew install pango
```

#### PPTX Template Issues

```python
# Ensure python-pptx is installed
pip install python-pptx
```

---

## Next Steps

- [Architecture](../ARCHITECTURE.md) - System design
- [Methodology](METHODOLOGY.md) - Scientific foundation
- [Contributing](../CONTRIBUTING.md) - Development guide
