# Fraud Transaction Detection

Course project for **CSE437 – Data Science: Coding with Real World Data**.

This project trains and compares seven supervised classifiers, Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, Support Vector Machine, Naive Bayes, and AdaBoost, to detect fraudulent transactions in a synthetic transaction dataset. The central finding is not which algorithm "wins," it's that two of the raw features near-perfectly reconstruct the target label, so the project is built around auditing and quantifying that leakage rather than reporting an inflated accuracy number at face value.

## Problem Statement

Given transaction-level details (amount, merchant category, device, authentication method, account history, and behavioral flags), predict whether a transaction is fraudulent (`Fraud_Label`: 1 = fraud, 0 = legitimate). This is a binary classification task on a moderately imbalanced target (32.13% fraud).

## Key Finding: Target Leakage

Two features, `Risk_Score` and `Failed_Transaction_Count_7d`, were found during EDA to correlate far more strongly with the label than anything else in the dataset. A simple hand-written rule:

```python
leak_rule = (Failed_Transaction_Count_7d == 4) | (Risk_Score > 0.85)
```

reconstructs `Fraud_Label` with **100.00% accuracy across all 50,000 rows**. Because of this, every model in this project is trained and evaluated **twice**:

- **Naive pipeline** — all features included (the leak is present)
- **Leakage-aware pipeline** — both leaking features removed

The naive pipeline lets tree-based models reach a literal 1.000 on every metric; the leakage-aware pipeline shows every model collapsing to chance-level performance (ROC-AUC ≈ 0.49–0.51), proving this synthetic dataset contains no learnable fraud signal outside of the leak.

## Methodology

1. **Cleaning:** drop identifier columns (`Transaction_ID`, `User_ID`); engineer `Timestamp` into `Hour` and `DayOfWeek` rather than encoding a near-unique raw value.
2. **Train/test split:** 70/30 stratified split on `Fraud_Label`, performed once and reused for both feature sets so the naive vs. leakage-aware comparison is apples-to-apples.
3. **Preprocessing:** a `ColumnTransformer` applies `StandardScaler` to numeric features and `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` to categoricals, fit only on the training split.
4. **Modeling:** all 7 classifiers, with `class_weight='balanced'` applied wherever scikit-learn supports it natively (Logistic Regression, Decision Tree, Random Forest, SVM), and depth-regularized trees (`max_depth=10`) to prevent overfitting.
5. **Evaluation:** accuracy, precision, recall, F1, ROC-AUC, and PR-AUC (average precision), the latter reported throughout since it's the more honest metric under class imbalance.
6. **Cost-sensitive threshold optimization:** sweeps the classification threshold using each transaction's real dollar amount as the false-negative cost, then honestly cross-checks the naive pipeline's apparent savings against the same sweep run on the leakage-aware pipeline.
7. **Explainability:** SHAP (`TreeExplainer` on Random Forest) for both pipelines, visually confirming the leak dominates the naive model's attributions and that nothing replaces it once removed.

## Results

| Model | Naive Accuracy | Naive ROC-AUC | Leakage-Aware Accuracy | Leakage-Aware ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.796 | 0.894 | 0.505 | 0.496 |
| KNN | 0.818 | 0.885 | 0.612 | 0.506 |
| Decision Tree | 1.000 | 1.000 | 0.477 | 0.497 |
| Random Forest | 1.000 | 1.000 | 0.572 | 0.497 |
| SVM | 0.975 | 0.999 | 0.509 | 0.493 |
| Naive Bayes | 0.886 | 0.944 | 0.679 | 0.494 |
| AdaBoost | 1.000 | 1.000 | 0.679 | 0.500 |

**Cost-sensitive thresholding** (naive Logistic Regression, using real transaction amounts as false-negative cost): moving from the default 0.50 threshold to the cost-optimal 0.18 reduces total cost from $108,355.62 to $23,651.86. Running the identical sweep on the leakage-aware model shows this saving does not survive removing the leak, the leakage-aware "optimum" simply degenerates to flagging 100% of transactions.

## Repository Structure

```
.
├── Fraud_Transaction_Detection.ipynb        # Full notebook: EDA, leakage investigation, both pipelines, cost analysis, SHAP
├── fraud_detection_analysis.py              # Same pipeline as a standalone script
├── fraud_transactions.csv                   # Dataset
├── Fraud_Transaction_Detection_Report.pdf    # Full written report (A to Z walkthrough with figures)
└── README.md
```

## Requirements

- Python 3.10+
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- shap

Install with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn shap
```

## Usage

Run the full pipeline as a script:

```bash
python fraud_detection_analysis.py
```

Or open the notebook for a step-by-step walkthrough with inline plots:

```bash
jupyter notebook Fraud_Transaction_Detection.ipynb
```

Both expect `fraud_transactions.csv` to be in the same directory.

## Key Takeaways

- A model reaching 100% accuracy should prompt suspicion before celebration; the correct response is to audit the inputs for shortcuts, not report the number.
- Applying class-imbalance handling and regularization consistently across every model, rather than favoring one, is what makes a naive-vs-leakage-aware comparison meaningful instead of misleading.
- Cost-sensitive threshold optimization should be checked against a leakage-free version of the model before its dollar figures are trusted as a deployment estimate.

## Author

Project submitted for CSE437 – Data Science: Coding with Real World Data.
