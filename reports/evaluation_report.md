# Credit Card Fraud Detection — Evaluation Report

This report is populated with actual held-out test metrics after training from the app's **Evaluation Report** page.

## Imbalance and metric selection

The Kaggle dataset contains 284,807 transactions and 492 frauds (about 0.173%). A classifier that predicts every transaction as legitimate can achieve approximately 99.83% accuracy while detecting zero fraud. Accuracy therefore hides the minority-class failure mode.

Prioritize fraud-class recall, precision, F1 and PR-AUC. Recall measures the share of fraud caught; precision measures how many alerts are truly fraud; F1 balances the two.

## Models

- Isolation Forest: unsupervised anomaly detector; contamination is set near the observed fraud prevalence.
- XGBoost + SMOTE: supervised model trained on a resampled training partition.
- Decision thresholds are selected using a validation subset of training data, never the test labels.

## Business impact

False negatives are missed fraud and can create direct losses, chargebacks and trust impacts. False positives block legitimate customers and can create friction, lost sales and support costs. The correct threshold depends on the relative cost of these outcomes and the review team's capacity.

## Recommendation

Use the model with the strongest validation-aligned and held-out fraud recall/F1 trade-off as a candidate for further validation. Do not choose by accuracy alone. Evaluate on time-separated data, inspect precision at the required recall, and quantify expected cost before deployment.

## Limitations and next steps

Synthetic fallback data is for software demonstration only. Anonymized V1–V28 features limit business interpretation. SMOTE may create synthetic samples unlike real fraud. Compare class weighting, cost-sensitive learning, calibration, drift monitoring and explainability before production use.
