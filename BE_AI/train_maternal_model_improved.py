import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
from sklearn.preprocessing import LabelBinarizer
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:
    print("[WARNING] imbalanced-learn not compatible, using class_weight='balanced' instead")
    SMOTE = None
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("IMPROVED MATERNAL HEALTH RISK PREDICTION MODEL TRAINING")
print("=" * 70)

# ======================================================
# 1. LOAD AND EXPLORE DATA
# ======================================================
df = pd.read_csv("Maternal Health Risk Data Set.csv")
df.rename(columns={"RiskLevel": "Risk"}, inplace=True)

print("\n[1] Dataset Overview")
print("-" * 70)
print(f"Total samples: {len(df)}")
print(f"Features: {df.columns.tolist()}")
print("\nFirst 5 rows:")
print(df.head())

print("\n[2] Class Distribution (Original)")
print("-" * 70)
class_counts = df["Risk"].value_counts()
print(class_counts)
print("\nClass percentages:")
for risk_class, count in class_counts.items():
    print(f"  {risk_class}: {count/len(df)*100:.2f}%")

# ======================================================
# 2. PREPROCESSING
# ======================================================
print("\n[3] Data Preprocessing")
print("-" * 70)
df = df.dropna()
print(f"Samples after removing NaN: {len(df)}")

X = df[["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]]
y = df["Risk"]

# Split data with stratification
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

print("\nTraining set class distribution:")
print(Y_train.value_counts())

# ======================================================
# 3. APPLY SMOTE FOR BALANCED TRAINING DATA
# ======================================================
print("\n[4] Applying SMOTE for Data Balancing")
print("-" * 70)

if SMOTE is not None:
    try:
        # SMOTE to balance classes
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_balanced, Y_train_balanced = smote.fit_resample(X_train, Y_train)

        print(f"Training samples after SMOTE: {len(X_train_balanced)}")
        print("\nBalanced training set class distribution:")
        print(pd.Series(Y_train_balanced).value_counts())
    except Exception as e:
        print(f"[WARNING] SMOTE failed: {e}")
        print("Using original training data with class_weight='balanced' instead")
        X_train_balanced = X_train
        Y_train_balanced = Y_train
else:
    print("[INFO] Using original training data with class_weight='balanced'")
    X_train_balanced = X_train
    Y_train_balanced = Y_train

# ======================================================
# 4. HYPERPARAMETER TUNING WITH GRID SEARCH
# ======================================================
print("\n[5] Hyperparameter Tuning with GridSearchCV")
print("-" * 70)

# Define parameter grid
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [4, 5, 6, 7],
    'min_samples_split': [8, 10, 12],
    'min_samples_leaf': [4, 5, 6],
    'max_features': ['sqrt', 'log2'],
}

# Base model
base_model = RandomForestClassifier(
    random_state=42,
    bootstrap=True,
    oob_score=True,
    class_weight='balanced',
    n_jobs=-1
)

# Grid search with stratified k-fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    base_model,
    param_grid,
    cv=skf,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)

print("Searching for best hyperparameters...")
print("This may take a few minutes...")
grid_search.fit(X_train_balanced, Y_train_balanced)

print("\n[OK] Best parameters found:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"\nBest CV F1-score (macro): {grid_search.best_score_:.4f}")

# ======================================================
# 5. TRAIN FINAL MODEL WITH BEST PARAMETERS
# ======================================================
print("\n[6] Training Final Model")
print("-" * 70)

model = grid_search.best_estimator_

print(f"Out-of-bag score: {model.oob_score_:.4f}")

# Cross-validation on balanced data
cv_scores = cross_val_score(model, X_train_balanced, Y_train_balanced, cv=5, scoring='accuracy')
print(f"\nCross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ======================================================
# 6. COMPREHENSIVE EVALUATION
# ======================================================
print("\n[7] Model Evaluation")
print("-" * 70)

# Predictions
train_pred = model.predict(X_train_balanced)
test_pred = model.predict(X_test)
test_pred_proba = model.predict_proba(X_test)

# Accuracy
train_accuracy = accuracy_score(Y_train_balanced, train_pred)
test_accuracy = accuracy_score(Y_test, test_pred)

print("\n--- Overfitting Check ---")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Difference: {abs(train_accuracy - test_accuracy):.4f}")

if abs(train_accuracy - test_accuracy) < 0.05:
    print("[OK] Excellent generalization (minimal overfitting)")
elif abs(train_accuracy - test_accuracy) < 0.10:
    print("[OK] Good generalization (low overfitting)")
else:
    print("[WARNING] Moderate overfitting detected")

# Detailed classification report
print("\n--- Classification Report (Test Set) ---")
print(classification_report(Y_test, test_pred, digits=4))

# Confusion Matrix
print("\n--- Confusion Matrix (Test Set) ---")
cm = confusion_matrix(Y_test, test_pred, labels=['high risk', 'low risk', 'mid risk'])
print("                Predicted")
print("                high risk  low risk  mid risk")
for i, label in enumerate(['high risk', 'low risk', 'mid risk']):
    print(f"Actual {label:10s}", end="")
    for j in range(3):
        print(f"{cm[i][j]:10d}", end="")
    print()

# Calculate metrics per class
print("\n--- Per-Class Metrics ---")
precision, recall, f1, support = precision_recall_fscore_support(
    Y_test, test_pred, labels=['high risk', 'low risk', 'mid risk']
)

for i, label in enumerate(['high risk', 'low risk', 'mid risk']):
    print(f"\n{label}:")
    print(f"  Precision: {precision[i]:.4f} (of predicted {label}, {precision[i]*100:.2f}% were correct)")
    print(f"  Recall:    {recall[i]:.4f} (of actual {label}, {recall[i]*100:.2f}% were detected)")
    print(f"  F1-score:  {f1[i]:.4f}")
    print(f"  Support:   {support[i]} samples")

# ROC AUC
lb = LabelBinarizer()
Y_test_binarized = lb.fit_transform(Y_test)
roc_auc = roc_auc_score(Y_test_binarized, test_pred_proba, multi_class="ovr", average="macro")
print(f"\n--- ROC AUC (macro, OvR) ---")
print(f"Score: {roc_auc:.4f}")

# Feature importance
print("\n--- Feature Importance ---")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.iterrows():
    print(f"{row['feature']:15s}: {row['importance']:.4f} ({row['importance']*100:.2f}%)")

# ======================================================
# 7. SAVE MODEL
# ======================================================
print("\n[8] Saving Model")
print("-" * 70)

with open("maternal_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("[OK] Model saved to maternal_model.pkl")

# Save model metadata
metadata = {
    'best_params': grid_search.best_params_,
    'test_accuracy': test_accuracy,
    'roc_auc': roc_auc,
    'feature_importance': feature_importance.to_dict(),
    'train_samples': len(X_train_balanced),
    'test_samples': len(X_test)
}

with open("model_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("[OK] Model metadata saved to model_metadata.pkl")

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"\nFinal Model Performance:")
print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"  ROC AUC Score: {roc_auc:.4f}")
print(f"  Overfitting:   {abs(train_accuracy - test_accuracy):.4f} difference")
print("\nModel is ready for deployment!")
