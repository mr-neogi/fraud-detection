
"""Markdown and PDF evaluation report exports."""
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def build_markdown(results, recommendation, n_total, n_fraud):
    lines = [
        "# Credit Card Fraud Detection — Evaluation Report",
        "",
        "## Dataset and imbalance",
        f"- Transactions evaluated: {n_total:,}",
        f"- Fraud transactions: {n_fraud:,} ({(100*n_fraud/max(n_total,1)):.4f}%)",
        "- Accuracy is misleading for rare fraud: predicting every transaction as legitimate can appear highly accurate while detecting no fraud.",
        "- Prioritize fraud-class recall, precision, F1 and PR-AUC; review false positives alongside missed fraud.",
        "",
        "## Model comparison",
        "| Model | Recall | Precision | F1 | PR-AUC | ROC-AUC | Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|"
    ]
    for name, m in results.items():
        lines.append(f"| {name} | {m['recall']:.4f} | {m['precision']:.4f} | {m['f1']:.4f} | {m['pr_auc']:.4f} | {m['roc_auc']:.4f} | {m['accuracy']:.4f} |")
    lines += [
        "", "## Confusion matrices", "Rows are actual [Legit, Fraud]; columns are predicted [Legit, Fraud]."
    ]
    for name, m in results.items():
        lines += [f"- **{name}:** `{m['confusion_matrix']}`"]
    lines += [
        "", "## Business impact",
        "- **False negatives (missed fraud):** potential direct financial loss, chargebacks, investigation cost and customer trust impact.",
        "- **False positives (legitimate transactions blocked):** declined purchases, customer friction, support contacts and lost revenue.",
        "- Thresholds should be selected with an explicit operational precision/recall and review-capacity target, not accuracy alone.",
        "", "## Recommendation", recommendation,
        "", "## Limitations and next steps",
        "- Synthetic fallback data is for software demonstration only; validate on the official dataset and then on time-separated production-like data.",
        "- Kaggle V1–V28 are anonymized; feature explanations are not equivalent to causal explanations.",
        "- SMOTE can create synthetic points that do not reflect real fraud patterns. Compare against class-weighted training and cost-sensitive objectives.",
        "- Monitor drift, calibration, threshold stability, latency, subgroup impacts and feedback from fraud analysts.",
        "- SHAP explanations are optional and should be checked for stability before operational use."
    ]
    return "\n".join(lines)

def save_pdf(markdown_text, output_path):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SmallBody", parent=styles["BodyText"], leading=13, spaceAfter=7))
    doc = SimpleDocTemplate(str(output_path), pagesize=letter, rightMargin=42, leftMargin=42)
    story = []
    for line in markdown_text.splitlines():
        if not line.strip():
            story.append(Spacer(1, 5)); continue
        if line.startswith("# "):
            story.append(Paragraph(line[2:], styles["Title"]))
        elif line.startswith("## "):
            story.append(Paragraph(line[3:], styles["Heading2"]))
        elif line.startswith("|"):
            story.append(Paragraph(line.replace("|", " &nbsp; | &nbsp; ").replace("---", "—"), styles["SmallBody"]))
        else:
            clean = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(clean, styles["SmallBody"]))
    doc.build(story)
    return Path(output_path)
