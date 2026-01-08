"""Generate test PDFs to inspect chart quality."""

from pathlib import Path

import numpy as np
import polars as pl

from faircareai import FairCareAudit
from faircareai.core.config import FairnessConfig, FairnessMetric, UseCaseType

# Create test data
np.random.seed(42)
n = 400

df = pl.DataFrame({
    "outcome": np.random.binomial(1, 0.3, n),
    "probability": np.random.beta(2, 5, n),
    "race": np.random.choice(["White", "Black", "Asian", "Hispanic"], n),
    "sex": np.random.choice(["Male", "Female"], n),
})

# Initialize audit
audit = FairCareAudit(
    data=df,
    pred_col="probability",
    target_col="outcome",
)

# Configure attributes
audit.add_sensitive_attribute(name="race", column="race", reference="White")
audit.add_sensitive_attribute(name="sex", column="sex", reference="Male")

# Set config
audit.config = FairnessConfig(
    model_name="Test Model - Chart Verification",
    primary_fairness_metric=FairnessMetric.EQUALIZED_ODDS,
    fairness_justification="Verifying static chart rendering in PDFs",
    use_case_type=UseCaseType.SCREENING,
)

# Run audit
print("Running audit...")
results = audit.run(bootstrap_ci=False)

# Output directory
output_dir = Path.cwd()
print(f"\nOutput directory: {output_dir}")

# Generate Governance PDF
gov_pdf = output_dir / "test_governance_report_with_charts.pdf"
print(f"\n📄 Generating governance PDF...")
results.to_governance_pdf(gov_pdf)
print(f"✅ Saved: {gov_pdf}")
print(f"   Size: {gov_pdf.stat().st_size / 1024:.1f} KB")

# Generate Data Scientist PDF
sci_pdf = output_dir / "test_scientist_report.pdf"
print(f"\n📊 Generating data scientist PDF...")
results.to_pdf(sci_pdf)
print(f"✅ Saved: {sci_pdf}")
print(f"   Size: {sci_pdf.stat().st_size / 1024:.1f} KB")

# Generate HTML for comparison
html_file = output_dir / "test_report_interactive.html"
print(f"\n🌐 Generating HTML report...")
results.to_html(html_file)
print(f"✅ Saved: {html_file}")
print(f"   Size: {html_file.stat().st_size / 1024:.1f} KB")

print("\n✅ All reports generated successfully!")
print("\nTo view:")
print(f"  PDF (governance): open {gov_pdf}")
print(f"  PDF (scientist):  open {sci_pdf}")
print(f"  HTML:             open {html_file}")
