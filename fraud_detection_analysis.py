"""
Fraud Transaction Detection

Compares seven classifiers on a synthetic transaction dataset to predict
Fraud_Label. Central to this analysis: two of the raw features
(Risk_Score, Failed_Transaction_Count_7d) turn out to near-perfectly
reconstruct the label, so every model is trained and evaluated twice,
once on the full ("naive") feature set and once with those two features
removed ("leakage-aware"), to separate genuine model skill from a
shortcut baked into the data. The leakage-aware results are then used
for a cost-sensitive threshold analysis and SHAP-based explainability.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay
)

sns.set(style="whitegrid")
RANDOM_STATE = 42

# ===========================================================================
# 1. LOAD DATA
# ===========================================================================
df = pd.read_csv("fraud_transactions.csv")
print("Shape:", df.shape)
print("Missing values:\n", df.isna().sum().sum(), "total")   # dataset has none
print("Fraud rate:", df['Fraud_Label'].mean().round(4))

# ===========================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ===========================================================================
# Target balance + two of the strongest-looking categorical splits, grouped
# into one figure instead of scattering many separate plots.
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

df['Fraud_Label'].value_counts().plot.pie(
    labels=['Not Fraud', 'Fraud'], autopct='%1.1f%%', colors=['steelblue', 'crimson'],
    ax=axes[0], ylabel='')
axes[0].set_title("Target Balance")

sns.countplot(x='Merchant_Category', hue='Fraud_Label', data=df, palette='coolwarm', ax=axes[1])
axes[1].set_title("Fraud by Merchant Category")
axes[1].tick_params(axis='x', rotation=45)

sns.countplot(x='Authentication_Method', hue='Fraud_Label', data=df, palette='coolwarm', ax=axes[2])
axes[2].set_title("Fraud by Authentication Method")

plt.tight_layout()
plt.savefig("fig_eda_categorical.png", dpi=110)
plt.close()

# Numeric feature distributions split by label, most relevant for spotting
# separation between classes at a glance.
num_cols_preview = ['Transaction_Amount', 'Account_Balance', 'Risk_Score', 'Failed_Transaction_Count_7d']
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
for ax, col in zip(axes, num_cols_preview):
    sns.boxplot(x='Fraud_Label', y=col, data=df, palette='Set2', ax=ax)
    ax.set_title(col)
plt.tight_layout()
plt.savefig("fig_eda_numeric_by_label.png", dpi=110)
plt.close()

# Correlation heatmap: numeric features against each other and the target.
plt.figure(figsize=(12, 9))
corr = df.select_dtypes(include='number').corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.4, center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("fig_correlation_heatmap.png", dpi=110)
plt.close()
print("\nCorrelation with Fraud_Label (sorted):")
print(corr['Fraud_Label'].sort_values(ascending=False))

# ===========================================================================
# 3. LEAKAGE INVESTIGATION
# ===========================================================================
# The correlation table above already singles out two features as unusually
# predictive on their own: Risk_Score and Failed_Transaction_Count_7d. This
# section checks whether they leak the label outright rather than merely
# correlating with it.
print("\nFraud rate by Failed_Transaction_Count_7d:")
print(df.groupby('Failed_Transaction_Count_7d')['Fraud_Label'].mean())

# A simple two-feature rule tests whether the label is trivially recoverable.
leak_rule = ((df['Failed_Transaction_Count_7d'] == 4) | (df['Risk_Score'] > 0.85)).astype(int)
leak_rule_accuracy = (leak_rule == df['Fraud_Label']).mean()
print(f"\nAccuracy of a 2-feature hand-written rule: {leak_rule_accuracy:.4f}")

# Visualizing the same thing: fraud (red) and non-fraud (blue) points
# separate almost perfectly along Risk_Score once Failed_Transaction_Count_7d == 4
# is set aside, which is exactly what a tree-based model finds in one split.
plt.figure(figsize=(9, 5))
sns.stripplot(x='Failed_Transaction_Count_7d', y='Risk_Score', hue='Fraud_Label',
              data=df, palette=['steelblue', 'crimson'], alpha=0.3, jitter=0.3)
plt.axhline(0.85, color='black', linestyle='--', linewidth=1, label='Risk_Score = 0.85')
plt.title("Near-perfect Label Separation via Risk_Score and Failed_Transaction_Count_7d")
plt.legend(title='Fraud_Label')
plt.tight_layout()
plt.savefig("fig_leakage_evidence.png", dpi=110)
plt.close()

print(
    f"\nConclusion: a hand-written rule using only 2 of 18 features reaches "
    f"{leak_rule_accuracy:.2%} accuracy. Any model given these two features "
    f"is not learning fraud patterns, it is rediscovering this rule. "
    f"All models below are therefore trained twice: once with the full "
    f"feature set ('naive'), and once with Risk_Score and "
    f"Failed_Transaction_Count_7d removed ('leakage-aware')."
)

# ===========================================================================
# 4. CLEANING & FEATURE ENGINEERING
# ===========================================================================
# Drop identifier columns (no predictive meaning, near-unique per row).
df_model = df.drop(columns=['Transaction_ID', 'User_ID'])

# Timestamp is a near-unique string per row; label-encoding it directly would
# just assign each row an arbitrary large number. Extract the components
# that could plausibly carry signal (hour of day, day of week) instead.
ts = pd.to_datetime(df_model['Timestamp'])
df_model['Hour'] = ts.dt.hour
df_model['DayOfWeek'] = ts.dt.dayofweek
df_model = df_model.drop(columns=['Timestamp'])

y = df_model['Fraud_Label']
X_full = df_model.drop(columns=['Fraud_Label'])

LEAK_COLS = ['Risk_Score', 'Failed_Transaction_Count_7d']
X_leakage_aware = X_full.drop(columns=LEAK_COLS)

print("\nNaive feature set:", X_full.shape[1], "columns")
print("Leakage-aware feature set:", X_leakage_aware.shape[1], "columns")

# ===========================================================================
# 5. TRAIN / TEST SPLIT + PREPROCESSING
# ===========================================================================
# One shared split (by row index) is reused for both feature sets so the
# naive vs. leakage-aware comparison later is apples-to-apples.
idx_train, idx_test = train_test_split(
    X_full.index, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)
y_train, y_test = y.loc[idx_train], y.loc[idx_test]


def build_preprocessor(X):
    """Numeric columns are scaled; nominal categoricals are one-hot encoded.
    Fitting happens only on the training split (done by the caller) to avoid
    leaking test-set categories or scale into preprocessing. sparse_output is
    pinned to False explicitly: mixing a dense StandardScaler output with a
    sparse OneHotEncoder output inside ColumnTransformer is version/threshold
    -dependent, and SHAP/seaborn downstream expect dense arrays."""
    num_cols = X.select_dtypes(include='number').columns.tolist()
    cat_cols = X.select_dtypes(exclude='number').columns.tolist()
    return ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
    ])


def make_train_test(X):
    X_train, X_test = X.loc[idx_train], X.loc[idx_test]
    pre = build_preprocessor(X)
    X_train_p = pre.fit_transform(X_train)
    X_test_p = pre.transform(X_test)
    return X_train_p, X_test_p, pre


Xtr_naive, Xte_naive, pre_naive = make_train_test(X_full)
Xtr_leak, Xte_leak, pre_leak = make_train_test(X_leakage_aware)
print("Naive processed shape:", Xtr_naive.shape)
print("Leakage-aware processed shape:", Xtr_leak.shape)

# ===========================================================================
# 6. MODEL TRAINING & EVALUATION FRAMEWORK
# ===========================================================================
def get_models():
    """Fresh model instances each call, since the same models are trained
    twice (naive and leakage-aware) and must not share fitted state.
    Random Forest is depth-limited for the same reason as the Decision Tree:
    left unconstrained, it grows trees averaging depth ~38 on the
    leakage-aware features (verified directly), fitting noise rather than
    signal, and this also makes exact SHAP explanation impractically slow.
    class_weight='balanced' is applied wherever a model supports it (Logistic
    Regression, Decision Tree, Random Forest, SVM); AdaBoost, Naive Bayes,
    and KNN have no such parameter in scikit-learn."""
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE),
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10, min_samples_leaf=20, class_weight='balanced', random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(
            max_depth=10, min_samples_leaf=10, class_weight='balanced', random_state=RANDOM_STATE),
        'SVM': SVC(class_weight='balanced', random_state=RANDOM_STATE),
        'Naive Bayes': GaussianNB(),
        'AdaBoost': AdaBoostClassifier(random_state=RANDOM_STATE),
    }


def score_of(model, X):
    """Ranking score for ROC/AUC: predicted probability where available,
    decision function otherwise (SVC without probability=True)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def train_and_evaluate(X_train, X_test, y_train, y_test, label):
    """Trains every model in get_models() and returns a results dict keyed
    by model name, each holding the fitted model, predictions, scores, and
    core metrics. `label` ('Naive' / 'Leakage-Aware') is stored for later
    side-by-side comparison plots."""
    results = {}
    for name, model in get_models().items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = score_of(model, X_test)
        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'y_score': y_score,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, pos_label=1),
            'recall': recall_score(y_test, y_pred, pos_label=1),
            'f1': f1_score(y_test, y_pred, pos_label=1),
            'auc': roc_auc_score(y_test, y_score),
            'avg_precision': average_precision_score(y_test, y_score),
            'pipeline': label,
        }
    return results


def results_to_df(results):
    return pd.DataFrame({
        name: {k: v for k, v in r.items() if k in
               ('accuracy', 'precision', 'recall', 'f1', 'auc', 'avg_precision')}
        for name, r in results.items()
    }).T.round(4)


def plot_confusion_grid(results, y_test, title):
    n = len(results)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(5 * ((n + 1) // 2), 9))
    axes = axes.ravel()
    for ax, (name, r) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, r['y_pred'])
        ConfusionMatrixDisplay(cm, display_labels=['No Fraud', 'Fraud']).plot(
            ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(f"{name}\n(acc={r['accuracy']:.2f})")
    for ax in axes[n:]:
        ax.axis('off')
    fig.suptitle(title, fontsize=15)
    plt.tight_layout()
    return fig


def plot_roc_comparison(results, y_test, title):
    plt.figure(figsize=(7, 6))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r['y_score'])
        plt.plot(fpr, tpr, label=f"{name} (AUC={r['auc']:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(fontsize=9)
    plt.tight_layout()


def plot_pr_comparison(results, y_test, title):
    """Precision-recall curves, the more informative view than ROC when the
    positive class (fraud) is the minority and is what actually matters."""
    plt.figure(figsize=(7, 6))
    baseline = y_test.mean()
    for name, r in results.items():
        prec, rec, _ = precision_recall_curve(y_test, r['y_score'])
        plt.plot(rec, prec, label=f"{name} (AP={r['avg_precision']:.2f})")
    plt.axhline(baseline, color='k', linestyle='--', label=f'No-skill baseline ({baseline:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(fontsize=9)
    plt.tight_layout()


# ===========================================================================
# 7. PIPELINE A: NAIVE (ALL FEATURES)
# ===========================================================================
results_naive = train_and_evaluate(Xtr_naive, Xte_naive, y_train, y_test, 'Naive')
print("\n=== Naive (all features) results ===")
print(results_to_df(results_naive))

fig = plot_confusion_grid(results_naive, y_test, "Confusion Matrices, Naive Pipeline (all features)")
fig.savefig("fig_confusion_naive.png", dpi=110)
plt.close(fig)

plot_roc_comparison(results_naive, y_test, "ROC Curves, Naive Pipeline (all features)")
plt.savefig("fig_roc_naive.png", dpi=110)
plt.close()

plot_pr_comparison(results_naive, y_test, "Precision-Recall Curves, Naive Pipeline (all features)")
plt.savefig("fig_pr_naive.png", dpi=110)
plt.close()

# ===========================================================================
# 8. PIPELINE B: LEAKAGE-AWARE
# ===========================================================================
results_leak = train_and_evaluate(Xtr_leak, Xte_leak, y_train, y_test, 'Leakage-Aware')
print("\n=== Leakage-aware results (Risk_Score, Failed_Transaction_Count_7d removed) ===")
print(results_to_df(results_leak))

fig = plot_confusion_grid(results_leak, y_test, "Confusion Matrices, Leakage-Aware Pipeline")
fig.savefig("fig_confusion_leakage_aware.png", dpi=110)
plt.close(fig)

plot_roc_comparison(results_leak, y_test, "ROC Curves, Leakage-Aware Pipeline")
plt.savefig("fig_roc_leakage_aware.png", dpi=110)
plt.close()

plot_pr_comparison(results_leak, y_test, "Precision-Recall Curves, Leakage-Aware Pipeline")
plt.savefig("fig_pr_leakage_aware.png", dpi=110)
plt.close()

# ===========================================================================
# 9. NAIVE VS LEAKAGE-AWARE COMPARISON
# ===========================================================================
# The single most important chart in this project: how much of each model's
# apparent performance was actually the leak, versus real signal. Both
# ROC-AUC and average precision (PR-AUC) are shown, since PR-AUC is the more
# honest metric here given fraud is the minority class.
models_list = list(get_models().keys())
auc_naive = [results_naive[m]['auc'] for m in models_list]
auc_leak = [results_leak[m]['auc'] for m in models_list]
ap_naive = [results_naive[m]['avg_precision'] for m in models_list]
ap_leak = [results_leak[m]['avg_precision'] for m in models_list]

x = np.arange(len(models_list))
width = 0.35
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

axes[0].bar(x - width / 2, auc_naive, width, label='Naive (with leak)', color='crimson', alpha=0.8)
axes[0].bar(x + width / 2, auc_leak, width, label='Leakage-Aware', color='steelblue', alpha=0.8)
axes[0].set_xticks(x, models_list, rotation=30, ha='right')
axes[0].set_ylabel('ROC-AUC')
axes[0].set_title('ROC-AUC: Naive (Leaky) vs. Leakage-Aware')
axes[0].legend()

axes[1].bar(x - width / 2, ap_naive, width, label='Naive (with leak)', color='crimson', alpha=0.8)
axes[1].bar(x + width / 2, ap_leak, width, label='Leakage-Aware', color='steelblue', alpha=0.8)
axes[1].axhline(y_test.mean(), color='k', linestyle='--', linewidth=1, label='No-skill baseline')
axes[1].set_xticks(x, models_list, rotation=30, ha='right')
axes[1].set_ylabel('Average Precision (PR-AUC)')
axes[1].set_title('PR-AUC: Naive (Leaky) vs. Leakage-Aware')
axes[1].legend()

plt.tight_layout()
plt.savefig("fig_naive_vs_leakage_aware.png", dpi=110)
plt.close()

# Accuracy specifically, one uniquely-colored bar per model, side by side for
# each pipeline. This is a simpler companion to the AUC/PR-AUC view above:
# easier to read at a glance, at the cost of being the metric most distorted
# by class imbalance (see Section 8's discussion of PR-AUC vs. accuracy).
model_colors = plt.cm.tab10(np.linspace(0, 1, len(models_list)))


def plot_accuracy_bars(results, title, ax):
    accs = [results[m]['accuracy'] for m in models_list]
    bars = ax.bar(models_list, accs, color=model_colors)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_xticks(range(len(models_list)))
    ax.set_xticklabels(models_list, rotation=30, ha='right')
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f'{acc:.2f}', ha='center', va='bottom', fontsize=9)


fig, axes = plt.subplots(1, 2, figsize=(16, 6))
plot_accuracy_bars(results_naive, 'Model Accuracy Comparison, Naive Pipeline', axes[0])
plot_accuracy_bars(results_leak, 'Model Accuracy Comparison, Leakage-Aware Pipeline', axes[1])
plt.tight_layout()
plt.savefig("fig_accuracy_comparison_by_model.png", dpi=110)
plt.close()

# ===========================================================================
# 10. COST-SENSITIVE THRESHOLD OPTIMIZATION
# ===========================================================================
# Every leakage-aware model above sits at AUC ~0.50, i.e. no better than
# chance: once Risk_Score and Failed_Transaction_Count_7d are removed, this
# synthetic dataset contains no learnable fraud signal at all. There is
# nothing to threshold-tune on a random model, so that finding is reported
# as-is above rather than dressed up with a fake optimization.
#
# To demonstrate the cost-sensitive thresholding methodology itself on a
# model with genuine graded, learnable signal, this section uses the naive
# pipeline's Logistic Regression. Unlike the tree ensembles (which reach a
# perfect, step-function-like separation and leave nothing to tune),
# Logistic Regression cannot fully represent the two-feature OR-rule and so
# produces smooth, spread-out probabilities, exactly the situation
# threshold-tuning is meant for. On a genuinely leakage-free dataset, this
# same code would be pointed at the leakage-aware pipeline instead.
best_model_name = 'Logistic Regression'
print(f"\nModel used for threshold optimization: {best_model_name} (naive pipeline)")

best_result = results_naive[best_model_name]
y_scores = best_result['y_score']
amounts_test = X_full.loc[idx_test, 'Transaction_Amount'].values
FP_COST = 5.0  # flat operational cost of reviewing/declining a legitimate transaction

thresholds = np.linspace(0.01, 0.99, 99)
total_costs = []
for t in thresholds:
    pred_t = (y_scores >= t).astype(int)
    fn_mask = (y_test.values == 1) & (pred_t == 0)
    fp_mask = (y_test.values == 0) & (pred_t == 1)
    cost = amounts_test[fn_mask].sum() + FP_COST * fp_mask.sum()
    total_costs.append(cost)

optimal_idx = int(np.argmin(total_costs))
optimal_threshold = thresholds[optimal_idx]
cost_at_optimal = total_costs[optimal_idx]
cost_at_default = total_costs[int(np.argmin(np.abs(thresholds - 0.5)))]

print(f"Cost at default threshold (0.50): ${cost_at_default:,.2f}")
print(f"Cost at optimal threshold ({optimal_threshold:.2f}): ${cost_at_optimal:,.2f}")
print(f"Savings from re-tuning the threshold: ${cost_at_default - cost_at_optimal:,.2f}")
print(
    "CAVEAT: this dollar figure demonstrates the thresholding METHOD only. "
    "It is not a trustworthy deployment estimate, since Logistic Regression "
    "here still has access to the two leak-derived features, so part of its "
    "signal is a shortcut that would not exist in a real, non-leaky dataset."
)

plt.figure(figsize=(8, 5))
plt.plot(thresholds, total_costs, color='darkorange')
plt.axvline(0.5, color='gray', linestyle='--', label='Default threshold (0.50)')
plt.axvline(optimal_threshold, color='green', linestyle='--',
            label=f'Optimal threshold ({optimal_threshold:.2f})')
plt.xlabel('Classification Threshold')
plt.ylabel('Total Cost ($)')
plt.title(f'Cost vs. Threshold, {best_model_name} (Naive Pipeline)')
plt.legend()
plt.tight_layout()
plt.savefig("fig_cost_threshold.png", dpi=110)
plt.close()

# The caveat above is worth proving, not just stating: repeat the identical
# sweep on the leakage-aware Logistic Regression (no access to the two leak
# features at all). If the "savings" above were genuine business signal
# rather than partly an artifact of the leak, a similarly shaped curve
# should appear here too, only weaker. It does not.
y_scores_leak = results_leak['Logistic Regression']['y_score']
total_costs_leak = []
for t in thresholds:
    pred_t = (y_scores_leak >= t).astype(int)
    fn_mask = (y_test.values == 1) & (pred_t == 0)
    fp_mask = (y_test.values == 0) & (pred_t == 1)
    cost = amounts_test[fn_mask].sum() + FP_COST * fp_mask.sum()
    total_costs_leak.append(cost)

optimal_idx_leak = int(np.argmin(total_costs_leak))
pred_leak_at_optimal = (y_scores_leak >= thresholds[optimal_idx_leak]).astype(int)
flag_rate_leak = pred_leak_at_optimal.mean()
print(f"\nSame sweep, Logistic Regression (leakage-aware): "
      f"optimal threshold {thresholds[optimal_idx_leak]:.2f}, "
      f"cost ${total_costs_leak[optimal_idx_leak]:,.2f} vs. default "
      f"${total_costs_leak[int(np.argmin(np.abs(thresholds - 0.5)))]:,.2f}")
print(
    f"Note: at that 'optimal' threshold, the leakage-aware model flags "
    f"{flag_rate_leak:.1%} of all transactions as fraud. With a near-chance "
    f"model and false-negative cost (real dollar amounts) far exceeding the "
    f"flat $5 false-positive cost, the cost-minimizing policy degenerates to "
    f"'flag almost everyone' rather than genuine discrimination. This is a "
    f"different failure mode than the naive model's inflated savings, but "
    f"it is still not a usable fraud policy: it is brute-force review, not "
    f"detection."
)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].plot(thresholds, total_costs, color='darkorange')
axes[0].axvline(0.5, color='gray', linestyle='--', label='Default (0.50)')
axes[0].axvline(optimal_threshold, color='green', linestyle='--',
                label=f'Optimal ({optimal_threshold:.2f})')
axes[0].set_title('Naive Logistic Regression\n(has access to the leak)')
axes[0].set_xlabel('Classification Threshold')
axes[0].set_ylabel('Total Cost ($)')
axes[0].legend()

axes[1].plot(thresholds, total_costs_leak, color='steelblue')
axes[1].axvline(0.5, color='gray', linestyle='--', label='Default (0.50)')
axes[1].axvline(thresholds[optimal_idx_leak], color='green', linestyle='--',
                label=f'Optimal ({thresholds[optimal_idx_leak]:.2f})')
axes[1].set_title('Leakage-Aware Logistic Regression\n(no access to the leak)')
axes[1].set_xlabel('Classification Threshold')
axes[1].set_ylabel('Total Cost ($)')
axes[1].legend()

fig.suptitle('The naive-pipeline savings do not survive removing the leak features', fontsize=13)
plt.tight_layout()
plt.savefig("fig_cost_threshold_leak_comparison.png", dpi=110)
plt.close()

# Confusion matrices at default vs optimal threshold, side by side.
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, t, title in [(axes[0], 0.5, 'Default Threshold (0.50)'),
                      (axes[1], optimal_threshold, f'Optimal Threshold ({optimal_threshold:.2f})')]:
    pred_t = (y_scores >= t).astype(int)
    cm = confusion_matrix(y_test, pred_t)
    ConfusionMatrixDisplay(cm, display_labels=['No Fraud', 'Fraud']).plot(ax=ax, cmap='Oranges', colorbar=False)
    ax.set_title(title)
plt.tight_layout()
plt.savefig("fig_cost_confusion_comparison.png", dpi=110)
plt.close()

# ===========================================================================
# 11. EXPLAINABILITY WITH SHAP
# ===========================================================================
# Random Forest is used for SHAP regardless of which model "won" above,
# since TreeExplainer is fast and exact for tree ensembles. Two versions are
# shown side by side: naive (to visually confirm the leak) and
# leakage-aware (to see what actually drives predictions once it's removed).
rf_naive = results_naive['Random Forest']['model']
rf_leak = results_leak['Random Forest']['model']

feature_names_naive = pre_naive.get_feature_names_out()
feature_names_leak = pre_leak.get_feature_names_out()

# TreeExplainer is exact but still scales with the number of rows explained;
# a random 1,000-row sample of the test set is standard practice for a
# summary plot and keeps runtime reasonable without changing the conclusion.
rng = np.random.default_rng(RANDOM_STATE)
sample_idx = rng.choice(Xte_naive.shape[0], size=min(1000, Xte_naive.shape[0]), replace=False)
Xte_naive_sample = Xte_naive[sample_idx]
Xte_leak_sample = Xte_leak[sample_idx]


def compute_shap_values(model, X_sample):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sample, check_additivity=False)
    if isinstance(sv, list):
        sv = sv[1]
    elif sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv


shap_values_naive = compute_shap_values(rf_naive, Xte_naive_sample)
shap_values_leak = compute_shap_values(rf_leak, Xte_leak_sample)

plt.figure()
shap.summary_plot(shap_values_naive, Xte_naive_sample, feature_names=feature_names_naive,
                   show=False, plot_size=(8, 6))
plt.title("SHAP Summary, Naive Random Forest (leak visible)")
plt.tight_layout()
plt.savefig("fig_shap_naive.png", dpi=110)
plt.close()

plt.figure()
shap.summary_plot(shap_values_leak, Xte_leak_sample, feature_names=feature_names_leak,
                   show=False, plot_size=(8, 6))
plt.title("SHAP Summary, Leakage-Aware Random Forest")
plt.tight_layout()
plt.savefig("fig_shap_leakage_aware.png", dpi=110)
plt.close()

# ===========================================================================
# 12. ALL-MODELS, ALL-METRICS COMPARISON
# ===========================================================================
# One consolidated view of every model against every metric, for each
# pipeline: a heatmap is used instead of a 42-bar grouped bar chart (7
# models x 6 metrics) since it stays readable at this size and the color
# scale makes the naive-vs-leakage-aware contrast immediately visible.
metrics_cols = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'avg_precision']
naive_matrix = results_to_df(results_naive)[metrics_cols]
leak_matrix = results_to_df(results_leak)[metrics_cols]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(naive_matrix, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
            linewidths=0.5, cbar_kws={'label': 'Score'}, ax=axes[0])
axes[0].set_title("Naive Pipeline (all features)")

sns.heatmap(leak_matrix, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
            linewidths=0.5, cbar_kws={'label': 'Score'}, ax=axes[1])
axes[1].set_title("Leakage-Aware Pipeline")

fig.suptitle("All Models x All Metrics", fontsize=15)
plt.tight_layout()
plt.savefig("fig_all_models_all_metrics.png", dpi=110)
plt.close()

# ===========================================================================
# 13. SUMMARY
# ===========================================================================
print("\n=== Summary ===")
print(f"Naive pipeline best AUC:          {max(auc_naive):.4f} ({models_list[int(np.argmax(auc_naive))]})")
print(f"Leakage-aware pipeline best AUC:  {max(auc_leak):.4f} ({models_list[int(np.argmax(auc_leak))]}, "
      f"essentially chance level)")
print(f"Cost-optimal threshold (naive Logistic Regression): {optimal_threshold:.2f} (vs. default 0.50)")
print(f"Illustrative cost reduction (methodology demo, not a deployment estimate): "
      f"${cost_at_default - cost_at_optimal:,.2f}")
print(
    "\nHeadline finding: near-perfect naive accuracy is entirely explained by "
    "two leaking features. Once removed, no model beats random guessing, "
    "meaning this synthetic dataset does not contain a realistic fraud "
    "detection problem outside of that leak. The cost-threshold figure above "
    "should be read the same way: it shows how to optimize a threshold given "
    "a cost matrix, not a number to expect in production, since the model it "
    "was computed on still has access to the leaking features."
)
