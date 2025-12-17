# FairCareAI Methodology

## Scientific Foundation

FairCareAI implements fairness auditing methodology based on peer-reviewed research in healthcare AI, algorithmic fairness, and clinical prediction model reporting.

**Core Principle:** Package SUGGESTS, humans DECIDE

All metrics and recommendations are advisory. Final deployment decisions require clinical judgment and organizational governance.

---

## Primary Framework: Van Calster et al. (2025)

FairCareAI follows the methodology proposed in:

> Van Calster B, et al. (2025). "Evaluating fairness of clinical prediction models: A practical guide." *The Lancet Digital Health*.

This framework provides:

1. **Standardized Metrics**: Performance and fairness metrics suitable for clinical AI
2. **Subgroup Analysis**: Stratified evaluation by demographic characteristics
3. **Calibration Focus**: Emphasis on calibration for clinical decision-making
4. **Uncertainty Quantification**: Bootstrap confidence intervals for all metrics

---

## Governance Framework: CHAI RAIC

FairCareAI aligns with the **Coalition for Health AI (CHAI)** Responsible AI Compass (RAIC) framework, specifically **Checkpoint 1: Pre-Deployment Audit**.

### CHAI Checkpoint 1 Criteria

| Criterion | Description | FairCareAI Section |
|-----------|-------------|-------------------|
| AC1.CR1 | Intended use documentation | Config: `intended_use` |
| AC1.CR3 | Target population specification | Config: `intended_population` |
| AC1.CR68 | Data quality assessment | Section 2: Descriptive Stats |
| AC1.CR82 | Subgroup performance analysis | Section 3-4: Subgroup/Fairness |
| AC1.CR92 | Fairness metric selection | Config: `primary_fairness_metric` |
| AC1.CR93 | Fairness justification | Config: `fairness_justification` |
| AC1.CR100 | Governance decision support | Section 7: Governance Block |

---

## Fairness Metrics

### The Impossibility Theorem

A fundamental result in algorithmic fairness demonstrates that **no single fairness metric is universally correct**. This is known as the impossibility theorem:

> Chouldechova A. (2017). "Fair prediction with disparate impact: A study of bias in recidivism prediction instruments." *Big Data*, 5(2), 153-163.

> Kleinberg J, Mullainathan S, Raghavan M. (2017). "Inherent trade-offs in the fair determination of risk scores." *Proceedings of Innovations in Theoretical Computer Science (ITCS)*.

**Key insight:** When base rates differ between groups, it is mathematically impossible to simultaneously satisfy:
- Equal false positive rates
- Equal false negative rates
- Equal positive predictive values

**Implication:** Fairness metric selection is a **value judgment** that must be made by humans based on clinical context, not a technical optimization.

### Available Metrics

#### 1. Demographic Parity

**Definition:** Equal selection rates across groups.

```
P(Ŷ = 1 | A = a) = P(Ŷ = 1 | A = b)
```

**Best for:** Resource allocation decisions where proportional distribution is important.

**Threshold:** Selection rate ratio within [0.8, 1.25] (based on EEOC 80% rule)

#### 2. Equalized Odds

**Definition:** Equal true positive rates (TPR) AND false positive rates (FPR) across groups.

```
P(Ŷ = 1 | Y = 1, A = a) = P(Ŷ = 1 | Y = 1, A = b)  // Equal TPR
P(Ŷ = 1 | Y = 0, A = a) = P(Ŷ = 1 | Y = 0, A = b)  // Equal FPR
```

**Best for:** Intervention triggers where both sensitivity and false alarms matter.

**Reference:** Hardt M, Price E, Srebro N. (2016). "Equality of opportunity in supervised learning." *Advances in Neural Information Processing Systems (NeurIPS)*.

**Threshold:** TPR/FPR difference < 0.10

#### 3. Equal Opportunity

**Definition:** Equal true positive rates across groups (subset of equalized odds).

```
P(Ŷ = 1 | Y = 1, A = a) = P(Ŷ = 1 | Y = 1, A = b)
```

**Best for:** Screening programs where detecting positive cases is primary concern.

**Threshold:** TPR difference < 0.10

#### 4. Predictive Parity

**Definition:** Equal positive predictive values (PPV) across groups.

```
P(Y = 1 | Ŷ = 1, A = a) = P(Y = 1 | Ŷ = 1, A = b)
```

**Best for:** Risk communication where flagged patients receive the prediction.

**Threshold:** PPV difference < 0.05

#### 5. Calibration

**Definition:** Equal calibration across groups (predicted probabilities match observed frequencies).

```
P(Y = 1 | Ŷ = p, A = a) = p = P(Y = 1 | Ŷ = p, A = b)
```

**Best for:** Shared decision-making where probabilities are communicated.

**Threshold:** Calibration slope within [0.8, 1.2], Brier score difference < 0.05

---

## Performance Metrics (TRIPOD+AI)

FairCareAI reports metrics aligned with the **TRIPOD+AI** statement for prediction model reporting:

> Collins GS, et al. (2024). "TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods." *BMJ*.

### Discrimination Metrics

#### AUROC (Area Under ROC Curve)

Measures the model's ability to distinguish between positive and negative cases.

- **Interpretation:** Probability that a randomly selected positive case ranks higher than a randomly selected negative case
- **Range:** 0.5 (no discrimination) to 1.0 (perfect discrimination)
- **Threshold:** Minimum acceptable AUROC typically > 0.65

#### AUPRC (Area Under Precision-Recall Curve)

Particularly important for imbalanced datasets (common in healthcare).

- **Interpretation:** Average precision across all recall thresholds
- **Range:** Prevalence (baseline) to 1.0 (perfect)
- **Advantage:** More informative than AUROC for rare outcomes

### Calibration Metrics

#### Brier Score

Mean squared error between predicted probabilities and outcomes.

```
Brier = (1/n) Σ (p_i - y_i)²
```

- **Range:** 0 (perfect) to 1 (worst)
- **Decomposition:** Can be decomposed into discrimination, calibration, and uncertainty components

**Reference:** Brier GW. (1950). "Verification of forecasts expressed in terms of probability." *Monthly Weather Review*.

#### Calibration Slope

Slope of logistic regression of outcomes on log-odds of predictions.

- **Ideal:** 1.0 (perfect calibration)
- **< 1.0:** Over-confident predictions (spread too wide)
- **> 1.0:** Under-confident predictions (spread too narrow)

**Reference:** Austin PC, Steyerberg EW. (2019). "The Integrated Calibration Index (ICI) and related metrics for quantifying the calibration of logistic regression models." *Statistics in Medicine*.

#### Average Calibration Error (ACE)

Mean absolute difference between predicted and observed probabilities across bins.

```
ACE = (1/B) Σ |observed_b - predicted_b|
```

### Decision Curve Analysis

Evaluates clinical utility across a range of threshold probabilities.

**Net Benefit:**
```
NB = (TP/n) - (FP/n) × (p_t / (1 - p_t))
```

Where `p_t` is the threshold probability.

**Reference:** Vickers AJ, Elkin EB. (2006). "Decision curve analysis: a novel method for evaluating prediction models." *Medical Decision Making*.

---

## Statistical Methods

### Bootstrap Confidence Intervals

FairCareAI uses percentile bootstrap for confidence intervals:

```python
# Algorithm
for i in 1...n_bootstrap:
    sample = resample(data, replace=True)
    metric_i = compute_metric(sample)

CI = [percentile(metrics, α/2), percentile(metrics, 1 - α/2)]
```

**Default:** 1,000 bootstrap iterations, 95% CI (α = 0.05)

**Reference:** Efron B, Tibshirani RJ. (1993). *An Introduction to the Bootstrap*. Chapman and Hall/CRC.

### Wilson Score Interval

Used for binomial proportions (TPR, FPR, PPV, etc.):

```
p̂ ± z × √(p̂(1-p̂)/n + z²/(4n²)) / (1 + z²/n)
```

**Advantages:**
- Better coverage than normal approximation for small samples
- Never produces negative intervals
- Works well for extreme proportions

**Reference:** Wilson EB. (1927). "Probable inference, the law of succession, and statistical inference." *Journal of the American Statistical Association*.

### Clopper-Pearson Interval

Exact binomial confidence interval for extreme proportions (< 5% or > 95%):

```
[Beta(α/2; x, n-x+1), Beta(1-α/2; x+1, n-x)]
```

**Use case:** When proportion is near 0 or 1 and Wilson score may be too narrow.

---

## Accessibility Standards

### WCAG 2.1 AA Compliance

All visualizations meet Web Content Accessibility Guidelines (WCAG) 2.1 Level AA:

- **Contrast Ratios:** Minimum 4.5:1 for normal text, 3:1 for large text
- **Color Independence:** Information not conveyed by color alone
- **Alternative Text:** Chart descriptions for screen readers

### Okabe-Ito Palette

FairCareAI uses the Okabe-Ito colorblind-safe palette:

| Name | Hex | Use |
|------|-----|-----|
| Orange | #E69F00 | Primary accent |
| Sky Blue | #56B4E9 | Secondary |
| Bluish Green | #009E73 | Success/pass |
| Yellow | #F0E442 | Warning |
| Blue | #0072B2 | Information |
| Vermillion | #D55E00 | Error/fail |
| Reddish Purple | #CC79A7 | Tertiary |
| Black | #000000 | Text/borders |

**Reference:** Okabe M, Ito K. (2008). "Color Universal Design (CUD): How to make figures and presentations that are friendly to colorblind people."

---

## Threshold Configuration

Default thresholds are **evidence-based starting points**, not requirements. Organizations should adjust based on:

- Clinical impact of decisions
- Risk tolerance
- Patient population characteristics
- Regulatory requirements

### Default Thresholds

```python
thresholds = {
    "min_subgroup_n": 100,           # Statistical reliability
    "demographic_parity_ratio": (0.8, 1.25),  # EEOC 80% rule
    "equalized_odds_diff": 0.1,      # Clinical significance
    "calibration_diff": 0.05,        # Calibration tolerance
    "min_auroc": 0.65,               # Minimum discrimination
    "max_missing_rate": 0.10,        # Data quality
}
```

### Threshold Origins

| Threshold | Origin | Rationale |
|-----------|--------|-----------|
| 80% rule | EEOC (1978) | Employment discrimination standard |
| 0.10 TPR/FPR diff | Hardt et al. (2016) | Practical significance |
| 0.05 calibration | Austin & Steyerberg (2019) | Clinical relevance |
| n=100 minimum | Power analysis | Stable CI estimation |

---

## Bibliography

### Core Methodology

```bibtex
@article{vancalster2025,
  title={Evaluating fairness of clinical prediction models: A practical guide},
  author={Van Calster, B and others},
  journal={The Lancet Digital Health},
  year={2025}
}

@article{collins2024tripod,
  title={TRIPOD+AI statement: updated guidance for reporting clinical prediction models},
  author={Collins, Gary S and others},
  journal={BMJ},
  year={2024}
}
```

### Fairness Theory

```bibtex
@article{chouldechova2017fair,
  title={Fair prediction with disparate impact: A study of bias in recidivism prediction instruments},
  author={Chouldechova, Alexandra},
  journal={Big Data},
  volume={5},
  number={2},
  pages={153--163},
  year={2017}
}

@inproceedings{kleinberg2017inherent,
  title={Inherent trade-offs in the fair determination of risk scores},
  author={Kleinberg, Jon and Mullainathan, Sendhil and Raghavan, Manish},
  booktitle={Innovations in Theoretical Computer Science (ITCS)},
  year={2017}
}

@inproceedings{hardt2016equality,
  title={Equality of opportunity in supervised learning},
  author={Hardt, Moritz and Price, Eric and Srebro, Nati},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2016}
}
```

### Statistical Methods

```bibtex
@book{efron1993bootstrap,
  title={An Introduction to the Bootstrap},
  author={Efron, Bradley and Tibshirani, Robert J},
  year={1993},
  publisher={Chapman and Hall/CRC}
}

@article{austin2019calibration,
  title={The Integrated Calibration Index (ICI) and related metrics},
  author={Austin, Peter C and Steyerberg, Ewout W},
  journal={Statistics in Medicine},
  year={2019}
}

@article{vickers2006decision,
  title={Decision curve analysis: a novel method for evaluating prediction models},
  author={Vickers, Andrew J and Elkin, Elena B},
  journal={Medical Decision Making},
  volume={26},
  number={6},
  pages={565--574},
  year={2006}
}
```

### Accessibility

```bibtex
@misc{okabe2008color,
  title={Color Universal Design (CUD): How to make figures and presentations that are friendly to colorblind people},
  author={Okabe, Masataka and Ito, Kei},
  year={2008},
  url={https://jfly.uni-koeln.de/color/}
}

@misc{wcag21,
  title={Web Content Accessibility Guidelines (WCAG) 2.1},
  author={{W3C}},
  year={2018},
  url={https://www.w3.org/TR/WCAG21/}
}
```

### Governance

```bibtex
@misc{chai2024raic,
  title={Responsible AI Compass (RAIC) Framework},
  author={{Coalition for Health AI}},
  year={2024},
  url={https://www.coalitionforhealthai.org/}
}
```

---

## Limitations

1. **Binary Classification Only**: Currently supports binary outcomes (0/1)
2. **Single Threshold**: Evaluates at one decision threshold per run
3. **Independence Assumption**: Does not model correlation between groups
4. **Static Analysis**: Point-in-time audit, not continuous monitoring
5. **No Causal Analysis**: Measures disparity, not discrimination causation

---

## Ethical Considerations

FairCareAI is a tool, not a solution. It provides:

- **Measurement**: Quantifies disparities with uncertainty
- **Transparency**: Documents methodology and limitations
- **Guidance**: Suggests appropriate metrics by use case

It does **not** provide:

- **Judgments**: Whether a model is "fair" or "unfair"
- **Mandates**: Required thresholds or decisions
- **Legal Advice**: Compliance determinations

Final deployment decisions require:

- Clinical expertise
- Organizational governance
- Legal review
- Community input
