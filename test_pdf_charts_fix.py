"""Test script to verify charts are embedded in governance PDF reports."""

import tempfile
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
    model_name="Test Model for PDF Charts",
    primary_fairness_metric=FairnessMetric.EQUALIZED_ODDS,
    fairness_justification="Testing charts in governance PDF reports",
    use_case_type=UseCaseType.SCREENING,
)

# Run audit
print("Running audit...")
results = audit.run(bootstrap_ci=False)

# Generate Governance PDF report
with tempfile.TemporaryDirectory() as tmp_dir:
    output_path = Path(tmp_dir) / "test_governance_report.pdf"
    print(f"\n📄 Generating governance PDF report to {output_path}...")

    try:
        pdf_path = results.to_governance_pdf(output_path)
        print(f"✅ Governance PDF generated successfully at: {pdf_path}")
        print(f"   File size: {pdf_path.stat().st_size / 1024:.1f} KB")

        # Check if file exists and has reasonable size
        if pdf_path.exists():
            size_kb = pdf_path.stat().st_size / 1024
            if size_kb > 50:  # PDF with charts should be larger than 50KB
                print(f"✅ PDF file size ({size_kb:.1f} KB) suggests charts were embedded")
            else:
                print(f"⚠️  PDF file size ({size_kb:.1f} KB) seems small - charts may be missing")
        else:
            print("❌ PDF file was not created")

    except Exception as e:
        print(f"❌ Error generating governance PDF: {e}")
        import traceback
        traceback.print_exc()

# Generate Data Scientist PDF report
with tempfile.TemporaryDirectory() as tmp_dir:
    output_path = Path(tmp_dir) / "test_scientist_report.pdf"
    print(f"\n📊 Generating data scientist PDF report to {output_path}...")

    try:
        pdf_path = results.to_pdf(output_path)
        print(f"✅ Data scientist PDF generated successfully at: {pdf_path}")
        print(f"   File size: {pdf_path.stat().st_size / 1024:.1f} KB")

        # Check if file exists and has reasonable size
        if pdf_path.exists():
            size_kb = pdf_path.stat().st_size / 1024
            if size_kb > 30:  # PDF with charts should be larger than 30KB
                print(f"✅ PDF file size ({size_kb:.1f} KB) suggests charts were embedded")
            else:
                print(f"⚠️  PDF file size ({size_kb:.1f} KB) seems small - charts may be missing")
        else:
            print("❌ PDF file was not created")

    except Exception as e:
        print(f"❌ Error generating data scientist PDF: {e}")
        import traceback
        traceback.print_exc()

print("\n✅ PDF chart fix test complete!")
