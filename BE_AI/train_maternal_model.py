import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import pickle
import numpy as np

# ======================================================
# 1. LOAD CSV
# ======================================================
df = pd.read_csv("Maternal Health Risk Data Set.csv")

# Rename kolom label
df.rename(columns={"RiskLevel": "Risk"}, inplace=True)

print("Sample data:")
print(df.head(), "\n")

# ======================================================
# 2. PREPROCESSING
# ======================================================
df = df.dropna()

X = df[["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]]
y = df["Risk"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ======================================================
# 3. TRAIN RANDOM FOREST WITH OVERFITTING PREVENTION
# ======================================================
# Reduced complexity to prevent overfitting on small dataset
model = RandomForestClassifier(
    n_estimators=100,          # Reduced from 300 to prevent overfitting
    max_depth=5,               # Reduced from 10 to limit tree complexity
    min_samples_split=10,      # Require at least 10 samples to split a node
    min_samples_leaf=5,        # Require at least 5 samples in each leaf
    max_features='sqrt',       # Use sqrt of features for each split (reduces variance)
    bootstrap=True,            # Enable bootstrap sampling
    oob_score=True,            # Out-of-bag score for validation
    random_state=42,
    class_weight='balanced'    # Handle class imbalance if present
)

print("Training model with cross-validation...")
# 5-fold cross-validation to assess generalization
cv_scores = cross_val_score(model, X_train, Y_train, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

model.fit(X_train, Y_train)
print(f"Out-of-bag score: {model.oob_score_:.4f}")

# ======================================================
# 4. EVALUASI
# ======================================================
# Evaluate on training set to check for overfitting
train_pred = model.predict(X_train)
train_accuracy = (train_pred == Y_train).mean()

# Evaluate on test set
pred = model.predict(X_test)
test_accuracy = (pred == Y_test).mean()
pred_proba = model.predict_proba(X_test)  # for ROC AUC

print("\n=== Training vs Test Accuracy (Overfitting Check) ===")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Difference: {abs(train_accuracy - test_accuracy):.4f}")
if abs(train_accuracy - test_accuracy) < 0.05:
    print("[OK] Good generalization (low overfitting)")
elif abs(train_accuracy - test_accuracy) < 0.10:
    print("[WARNING] Moderate overfitting")
else:
    print("[ERROR] High overfitting detected")

print("\n=== Classification Report (Test Set) ===")
print(classification_report(Y_test, pred))

# ---------- ROC AUC MULTI-CLASS ----------
lb = LabelBinarizer()
Y_test_binarized = lb.fit_transform(Y_test)

roc_auc = roc_auc_score(
    Y_test_binarized,
    pred_proba,
    multi_class="ovr",
    average="macro"
)

print(f"\nROC AUC (macro, OvR): {roc_auc:.4f}")

# Feature importance
print("\n=== Feature Importance ===")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance)

# ======================================================
# 5. SAVE MODEL
# ======================================================
with open("maternal_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved to maternal_model.pkl")
