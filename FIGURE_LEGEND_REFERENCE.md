# FairCareAI Figure Legend Quick Reference

## Quick Reference for Report Authors

This guide provides the improved messaging for all FairCareAI figures. Use these templates when creating custom reports or presentations.

---

## Van Calster et al. (2025) Core Implementation

FairCareAI implements the **4 RECOMMENDED** metrics from Van Calster et al. (2025):

| Metric | Module Function | Van Calster Classification |
|--------|-----------------|---------------------------|
| **AUROC** | `compute_auroc_by_subgroup()` | "The key discrimination measure" |
| **Calibration** | `compute_calibration_by_subgroup()` | "Most insightful approach" |
| **Net Benefit (DCA)** | `compute_net_benefit_by_subgroup()` | "Essential to report" |
| **Risk Distribution** | `compute_risk_distribution_by_subgroup()` | "Provides valuable insights" |

**Reference:**
> Van Calster B, et al. Evaluation of performance measures in predictive AI models. Lancet Digit Health 2025. doi:10.1016/j.landig.2025.100916

---

## Overall Performance Figures

### 1. AUROC Gauge

**Title:** Model Discrimination (AUROC)

**Explanation (Governance):**
> AUROC measures how well the model separates high-risk from low-risk patients. Think of it as the model's ability to rank patients correctly. Score of 0.5 = random guessing (coin flip). Score of 1.0 = perfect ranking. Healthcare standard: 0.7 or higher is acceptable, 0.8+ is strong.

**Explanation (Data Science):**
> AUROC (Area Under ROC Curve) measures discrimination - the model's ability to rank patients by risk. Range: 0.5 (random) to 1.0 (perfect). Clinical threshold: ≥0.7 acceptable, ≥0.8 strong.

**Axis Labels:**
- Gauge range: 0.5 to 1.0
- Threshold line at 0.7: "Acceptable minimum"
- Color zones: Red (<0.7), Yellow (0.7-0.8), Green (>0.8)

**Interpretation:**
- <0.7: Below standard, improve model before deployment
- 0.7-0.8: Acceptable, monitor performance
- ≥0.8: Excellent, no action required

---

### 2. Calibration Plot

**Title:** Model Calibration: Do Predicted Risks Match Reality?

**Explanation (Governance):**
> Calibration checks if predicted risks match actual outcomes. If the model predicts 20% risk for a group, do about 20% actually experience the outcome? Points closer to the diagonal line = more trustworthy risk estimates. Why it matters: Under/over-estimating risk can lead to wrong treatment decisions.

**Explanation (Data Science):**
> Calibration plot comparing predicted probabilities to observed outcome frequencies. Perfect calibration follows the diagonal (y=x). Deviation indicates systematic over/under-prediction.

**Axis Labels:**
- **X-axis:** "Predicted Risk (what the model says)" [0-100%]
- **Y-axis:** "Observed Rate (what actually happened)" [0-100%]
- Reference line: Perfect calibration (diagonal)

**Key Metrics:**
- **Calibration Slope:** Ideal = 1.0, acceptable = 0.8-1.2
- **Brier Score:** Excellent <0.15, acceptable <0.25

---

### 3. Brier Score Gauge

**Title:** Brier Score (Prediction Accuracy)

**Explanation (Governance):**
> Brier Score measures overall prediction accuracy (0 = perfect, 0.25 = poor). Lower is better. Think of it as the "error" in risk predictions. Score <0.15 = excellent calibration. Score 0.15-0.25 = acceptable. Score >0.25 = needs improvement.

**Explanation (Data Science):**
> Brier score measures mean squared error of probabilistic predictions. Range: 0 (perfect calibration) to 0.25 (reference model). Threshold: <0.15 excellent, <0.25 acceptable.

**Axis Labels:**
- Gauge range: 0 to 0.5
- Color zones: Green (0-0.15), Yellow (0.15-0.25), Red (>0.25)

**Interpretation:**
- <0.15: Excellent calibration
- 0.15-0.25: Acceptable calibration
- >0.25: Recalibration required

---

### 4. Classification Metrics Bar Chart

**Title:** Classification Metrics at Threshold X.XX

**Explanation (Governance):**
> At the chosen risk threshold, these metrics show what happens to patients: Sensitivity = % of actual cases correctly identified. Specificity = % without the condition correctly identified. PPV = When flagged positive, % who actually have the condition.

**Explanation (Data Science):**
> Performance metrics at operating threshold. Sensitivity (TPR), Specificity (TNR), PPV (Precision), and selection rate.

**Axis Labels:**
- **X-axis:** "Performance Metric"
- **Y-axis:** "Percentage of Patients" [0-100%]

**Metrics Definitions:**
- **Sensitivity:** % of actual cases correctly identified (TPR)
- **Specificity:** % without condition correctly identified (TNR)
- **PPV:** When flagged positive, % truly positive (Precision)
- **% Flagged:** Proportion identified for intervention

---

## Subgroup Performance Figures (Van Calster 4)

### 1. AUROC by Subgroup

**Title:** Model Accuracy (AUROC) by Demographic Group

**Explanation (Governance):**
> AUROC by Subgroup: Does the model perform equally well across all demographic groups? All bars should be similar height (difference <0.05 is ideal). Lower bars mean the model is less accurate for that group. Why it matters: We want the model to work well for everyone, not just some groups.

**Explanation (Data Science):**
> AUROC discrimination performance stratified by demographic subgroup. Ideal: difference between groups <0.05. Larger differences indicate potential fairness concerns per Van Calster et al. (2025).

**Axis Labels:**
- **X-axis:** "Demographic Group"
- **Y-axis:** "AUROC (Model Accuracy Score)" [0.5-1.0]
- Threshold line at 0.7: "Acceptable minimum (0.7)"

**Interpretation:**
- All groups ≥0.7: Acceptable across all groups
- Difference between groups <0.05: Fair performance
- Difference between groups ≥0.05: Review for fairness

---

### 2. Sensitivity (TPR) by Subgroup

**Title:** Sensitivity: % of Actual Cases Detected by Group

**Explanation (Governance):**
> Sensitivity (True Positive Rate): Of patients who actually develop the outcome, what percentage does the model correctly identify in each group? Large differences mean the model 'misses' more cases in certain groups. Fairness goal: Differences between groups should be <10 percentage points.

**Explanation (Data Science):**
> True Positive Rate (Sensitivity, Recall) by demographic subgroup. Equal Opportunity fairness metric. Threshold: difference between groups <0.10 (10 percentage points).

**Axis Labels:**
- **X-axis:** "Demographic Group"
- **Y-axis:** "True Positive Rate (%)" [0-100%]

**Interpretation:**
- Difference <10pp: Satisfies Equal Opportunity
- Difference 10-20pp: Review for clinical significance
- Difference >20pp: Significant fairness concern

---

### 3. FPR by Subgroup

**Title:** False Alarms: % Incorrectly Flagged by Group

**Explanation (Governance):**
> False Positive Rate: Of patients who DON'T have the outcome, what percentage are incorrectly flagged as high-risk in each group? Lower is better (fewer false alarms). Fairness concern: Higher FPR means a group gets unnecessary interventions/worry.

**Explanation (Data Science):**
> False Positive Rate by demographic subgroup. Component of Equalized Odds. Lower indicates fewer unnecessary interventions. Fairness threshold: difference <0.10.

**Axis Labels:**
- **X-axis:** "Demographic Group"
- **Y-axis:** "False Positive Rate (%)" [0-50%]

**Interpretation:**
- All groups <10%: Low false alarm rate
- Difference <10pp: Fair false positive burden
- Large differences: Unequal burden of false alarms

---

### 4. Selection Rate by Subgroup

**Title:** Intervention Rate: % Flagged as High-Risk by Group

**Explanation (Governance):**
> Selection Rate: What percentage of each group is flagged as 'high-risk' by the model? This shows which groups the model identifies for intervention. Large differences may indicate disparate treatment even if clinically justified. Consider: Should intervention rates differ by demographics?

**Explanation (Data Science):**
> Selection rate (positive prediction rate) by subgroup. Measures Demographic Parity. Clinical context required: differences may be justified by base rate variation or unjustified disparate treatment.

**Axis Labels:**
- **X-axis:** "Demographic Group"
- **Y-axis:** "Selection Rate (% flagged)" [0-100%]

**Interpretation:**
- Equal rates: Demographic parity satisfied
- Different rates: May reflect base rate differences (justified) or bias (unjustified)
- Clinical review required for interpretation

---

## Technical Visualization Improvements

### ROC Curve

**Axis Labels:**
- **X-axis:** "False Positive Rate (% without outcome incorrectly flagged)" [0-100%]
- **Y-axis:** "True Positive Rate (% with outcome correctly identified)" [0-100%]
- Reference line: Random classifier (diagonal)

---

### Precision-Recall Curve

**Axis Labels:**
- **X-axis:** "Recall (% of actual positives detected)" [0-100%]
- **Y-axis:** "Precision (% of flagged cases that are true positives)" [0-100%]
- Reference line: Prevalence (horizontal)

---

### Decision Curve Analysis

**Title:** Decision Curve Analysis: When is the Model Clinically Useful?

**Axis Labels:**
- **X-axis:** "Threshold Probability (risk level for action)" [0-100%]
- **Y-axis:** "Net Benefit (clinical value per 100 patients)"

**Interpretation:**
> Model is useful when its curve is above both 'Treat All' and 'Treat None' strategies.

---

## Report Section Guidance

### Section 3: Overall Model Performance

**What to include:**
1. Discrimination metrics with interpretation
2. Calibration metrics with interpretation
3. Classification metrics at threshold
4. Clear thresholds for "acceptable" vs "needs review"

**Key phrases:**
- "0.5 = random guessing, 1.0 = perfect"
- "Lower is better" (for Brier)
- "At this risk cutoff, here's what happens to patients:"

---

### Section 4: Subgroup Performance

**What to include:**
1. Performance stratified by demographics
2. Clear "what to look for" guidance
3. Interpretation thresholds
4. Glossary of metrics

**Key phrases:**
- "Performance should be similar across all demographic groups"
- "Large differences in AUROC (>0.05) or TPR/FPR (>10pp) may indicate fairness concerns"
- "Higher is better" (AUROC, TPR)
- "Lower is better" (FPR)

---

### Section 5: Fairness Assessment

**What to include:**
1. Primary fairness metric with justification
2. Differences from reference group
3. Interpretation of fairness thresholds
4. Impossibility theorem context

**Key phrases:**
- "Differences <0.10 are typically acceptable"
- "Large differences mean the model 'misses' more cases in certain groups"
- "When base rates differ, trade-offs are necessary"
- "No model can satisfy all fairness criteria simultaneously"

---

## Color Coding (WCAG 2.1 AA Compliant)

### Status Colors
- **Green (#009E73):** Pass, Excellent, Within threshold
- **Yellow (#C9B900):** Warning, Review, Near threshold
- **Red (#D55E00):** Error, Critical, Outside threshold

### Subgroup Colors (Okabe-Ito Palette)
- Primary group (reference): Blue (#0072B2)
- Comparison groups: Orange (#E69F00), Green (#009E73), etc.

---

## Common Pitfalls to Avoid

### ❌ DON'T:
- Use generic axis labels like "Value", "Score"
- Present numbers without interpretation
- Assume technical knowledge
- Use jargon without explanation
- Leave thresholds unexplained

### ✅ DO:
- Use descriptive axis labels explaining what and why
- Provide clear interpretation guidance
- Include plain language for governance
- Define all technical terms
- State thresholds explicitly with context

---

## Templates

### Governance Figure Caption Template
```
[METRIC NAME]: [PLAIN LANGUAGE WHAT]

What it shows: [PLAIN LANGUAGE HOW TO READ]
Why it matters: [CLINICAL/OPERATIONAL IMPACT]
What to look for: [GOOD VS BAD]
Threshold: [SPECIFIC VALUE] ([WHY THIS VALUE])
```

### Data Science Figure Caption Template
```
[METRIC NAME] by [STRATIFICATION]

Methodology: [STATISTICAL METHOD]
Interpretation: [TECHNICAL DEFINITION]
Threshold: [VALUE] (Reference: [CITATION])
Clinical context: [APPLICATION]
```

---

## Quick Metric Reference

| Metric | Range | Good | Acceptable | Poor | Why it matters |
|--------|-------|------|------------|------|----------------|
| AUROC | 0.5-1.0 | ≥0.8 | 0.7-0.8 | <0.7 | Model's ability to rank patients |
| Brier Score | 0-0.25 | <0.15 | 0.15-0.25 | >0.25 | Overall prediction error |
| Cal. Slope | 0-2.0 | 0.8-1.2 | 0.6-1.5 | Other | Agreement between predicted/observed |
| Sensitivity | 0-100% | >80% | 60-80% | <60% | % of actual cases caught |
| Specificity | 0-100% | >90% | 70-90% | <70% | % correctly identified as negative |
| PPV | 0-100% | >70% | 50-70% | <50% | When flagged positive, % truly positive |

---

## Typography Standards (v0.2.1+)

All FairCareAI figures now meet publication-ready standards:

| Element | Minimum | Standard |
|---------|---------|----------|
| Body/explanation text | 14px | 14-16px |
| Axis labels | 14px | 14-16px |
| Axis titles | 14px | 16px |
| Chart titles | 16px | 20px |
| Table headers | 14px | 14px |
| Table cells | 14px | 14px |
| Legend text | 14px | 14px |
| Annotations | 14px | 14-16px |
| Disclaimer | 14px | 14px |

**Standards Applied:**
- Scientific publication typography guidelines
- WCAG 2.1 AA accessibility compliance
- Minimum 4.5:1 contrast ratio for all text
- Okabe-Ito colorblind-safe palette

---

## Version
Last updated: 2025-12-17
FairCareAI v0.2.1
Van Calster et al. (2025) methodology
Publication-ready typography (14px minimum)
