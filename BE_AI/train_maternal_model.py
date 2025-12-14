import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import pickle
import numpy as np

df = pd.read_csv("Maternal Health Risk Data Set.csv")

df.rename(columns={"RiskLevel": "Risk"}, inplace=True)

print("Sample data:")
print(df.head(), "\n")

df = df.dropna()

X = df[["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]]
y = df["Risk"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


model = RandomForestClassifier(
    n_estimators=100,        
    max_depth=5,              
    min_samples_split=10,    
    min_samples_leaf=5,       
    max_features='sqrt',     
    bootstrap=True,        
    oob_score=True,     
    random_state=42,
    class_weight='balanced'   
)

print("Training model with cross-validation...")
cv_scores = cross_val_score(model, X_train, Y_train, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

model.fit(X_train, Y_train)
print(f"Out-of-bag score: {model.oob_score_:.4f}")

train_pred = model.predict(X_train)
train_accuracy = (train_pred == Y_train).mean()

pred = model.predict(X_test)
test_accuracy = (pred == Y_test).mean()
pred_proba = model.predict_proba(X_test) 

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

lb = LabelBinarizer()
Y_test_binarized = lb.fit_transform(Y_test)

roc_auc = roc_auc_score(
    Y_test_binarized,
    pred_proba,
    multi_class="ovr",
    average="macro"
)

print(f"\nROC AUC (macro, OvR): {roc_auc:.4f}")

print("\n=== Feature Importance ===")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance)

with open("maternal_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved to maternal_model.pkl")
