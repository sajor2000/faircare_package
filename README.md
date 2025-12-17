# FairCareAI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Healthcare AI Fairness Auditing for Clinical Decision Support**

FairCareAI is a Python package for auditing machine learning models for fairness in clinical contexts. Built on the **Van Calster et al. (2025)** methodology and aligned with the **CHAI RAIC** governance framework, it helps health system data scientists present evidence-based fairness analysis to governance stakeholders.

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](ARCHITECTURE.md) | System design, data flow, and component relationships |
| [Methodology](docs/METHODOLOGY.md) | Scientific foundation, fairness theory, and references |
| [Usage Guide](docs/USAGE.md) | Complete API reference, CLI, and examples |
| [Contributing](CONTRIBUTING.md) | Development setup and contribution guidelines |

---

## Key Features

- **CHAI-Aligned Governance**: Follows [CHAI RAIC Checkpoint 1](https://www.coalitionforhealthai.org/) framework
- **TRIPOD+AI Compliant Metrics**: Scientifically validated performance metrics
- **Accessibility-First**: WCAG 2.1 AA compliant visualizations
- **Multiple Export Formats**: HTML dashboards, PDF reports, PowerPoint decks
- **HIPAA-Friendly**: All computation runs locally, no cloud dependencies

## Governance Philosophy

> **Package SUGGESTS, humans DECIDE**

All outputs are ADVISORY, not mandates. Final deployment decisions rest with clinical stakeholders and governance committees.

## Installation

```bash
pip install faircareai
```

For PDF/PPTX/static exports:

```bash
pip install "faircareai[export]"
```

For development:

```bash
git clone https://github.com/sajor2000/faircare_package.git
cd faircareai
pip install -e ".[dev]"
```

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

## Quick Start

```python
from faircareai import FairCareAudit, FairnessConfig
from faircareai.core.config import FairnessMetric, UseCaseType

# Load your model predictions
audit = FairCareAudit(
    data="predictions.parquet",  # or DataFrame
    pred_col="risk_score",       # column with model probabilities
    target_col="readmit_30d"     # column with actual outcomes (0/1)
)

# See suggested sensitive attributes
audit.suggest_attributes()
# Output: Detected race_ethnicity, sex, insurance columns...

# Accept suggestions (1-indexed)
audit.accept_suggested_attributes([1, 2, 3])

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

# Run the audit
results = audit.run()

# View executive summary
results.plot_executive_summary()

# Export reports
results.to_html("fairness_report.html")
results.to_pdf("fairness_report.pdf")
results.to_pptx("governance_deck.pptx")
```

## Interactive Dashboard

Launch the Streamlit dashboard for interactive analysis:

```python
import faircareai
faircareai.launch()
```

Or from the command line:

```bash
faircareai dashboard
```

## Fairness Metrics

FairCareAI supports multiple fairness definitions, acknowledging the **impossibility theorem** (Chouldechova 2017, Kleinberg et al. 2017) - no single metric is universally correct.

| Metric | Definition | Best For |
|--------|------------|----------|
| **Demographic Parity** | Equal selection rates | Resource allocation |
| **Equalized Odds** | Equal TPR and FPR | Intervention triggers |
| **Equal Opportunity** | Equal TPR only | Screening programs |
| **Predictive Parity** | Equal PPV | Risk communication |
| **Calibration** | Equal calibration curves | Shared decision-making |

Use `audit.suggest_fairness_metric()` for context-specific recommendations.

## Report Sections

Generated reports include 7 CHAI-aligned sections:

1. **Executive Summary** - Go/no-go advisory with key findings
2. **Descriptive Statistics** - Cohort characteristics (Table 1)
3. **Overall Performance** - AUROC, AUPRC, calibration metrics
4. **Subgroup Performance** - Performance by demographic group
5. **Fairness Assessment** - Disparity analysis with CIs
6. **Limitations & Flags** - Warnings and considerations
7. **Governance Decision Block** - Sign-off section for stakeholders

## Use Cases

### Intervention Trigger Models
Models that determine who receives an intervention (care management, outreach):
- Recommended metric: **Equalized Odds**
- Key concern: Equal access to beneficial interventions

### Risk Communication
Models that communicate risk to patients/providers:
- Recommended metric: **Calibration**
- Key concern: Trustworthy probabilities for shared decisions

### Resource Allocation
Models that allocate limited resources:
- Recommended metric: **Demographic Parity**
- Key concern: Proportional distribution of resources

### Screening Programs
Models used for disease screening:
- Recommended metric: **Equal Opportunity**
- Key concern: Equal detection rates for those with disease

## API Reference

### Core Classes

- `FairCareAudit` - Main orchestration class
- `FairnessConfig` - Audit configuration
- `AuditResults` - Results container with visualization methods

### Enums

- `FairnessMetric` - Available fairness metrics
- `UseCaseType` - Clinical use case categories
- `ModelType` - Model type classification

### Key Methods

```python
# Attribute management
audit.suggest_attributes()
audit.accept_suggested_attributes([1, 2])
audit.add_sensitive_attribute("custom", column="custom_col")
audit.add_intersection(["race", "sex"])

# Configuration
audit.suggest_fairness_metric()
audit.config.validate()

# Execution
results = audit.run(bootstrap_ci=True, n_bootstrap=1000)

# Visualization
results.plot_executive_summary()
results.plot_forest_plot("race")
results.plot_calibration_curves()

# Export
results.to_html("report.html")
results.to_pdf("report.pdf")
results.to_pptx("deck.pptx")
```

## Configuration

### Thresholds

Default thresholds are evidence-based starting points. Health systems should adjust based on context:

```python
config = FairnessConfig(
    model_name="My Model",
    thresholds={
        "min_subgroup_n": 100,           # Minimum subgroup size
        "demographic_parity_ratio": (0.8, 1.25),  # 80% rule
        "equalized_odds_diff": 0.1,      # Max TPR/FPR difference
        "calibration_diff": 0.05,        # Max calibration error
        "min_auroc": 0.65,               # Minimum acceptable AUROC
        "max_missing_rate": 0.10,        # Max missing data rate
    }
)
```

## Requirements

- Python >= 3.10
- polars >= 0.20.0
- plotly >= 5.18.0
- streamlit >= 1.30.0
- scipy >= 1.11.0
- statsmodels >= 0.14.0

See `pyproject.toml` for complete dependencies.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Type checking
mypy src/faircareai

# Linting
ruff check src/
```

## Contributing

Contributions are welcome! Please read our contributing guidelines and ensure:

1. All tests pass
2. Code follows existing style
3. Documentation is updated
4. Scientific claims are cited

## License

MIT License - see LICENSE file for details.

## Citation

If you use FairCareAI in your research, please cite:

```bibtex
@software{faircareai,
  title = {FairCareAI: Healthcare AI Fairness Auditing},
  year = {2024},
  url = {https://github.com/sajor2000/faircare_package}
}
```

## References

- Chouldechova, A. (2017). Fair prediction with disparate impact. Big Data, 5(2).
- Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. NeurIPS.
- Kleinberg, J., Mullainathan, S., & Raghavan, M. (2017). Inherent trade-offs in algorithmic fairness. ITCS.
- Collins, G.S., et al. (2024). TRIPOD+AI statement. BMJ.

## Support

- **Documentation**: See [docs/](docs/) folder
- **Issues**: [GitHub Issues](https://github.com/sajor2000/faircare_package/issues)
- **Discussions**: [GitHub Discussions](https://github.com/sajor2000/faircare_package/discussions)

---

**Disclaimer**: FairCareAI provides CHAI-grounded guidance for fairness auditing. All outputs are advisory. Final deployment decisions rest with the health system and clinical governance committees.
