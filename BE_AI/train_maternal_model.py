import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import pickle

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
# 3. TRAIN RANDOM FOREST
# ======================================================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

model.fit(X_train, Y_train)

# ======================================================
# 4. EVALUASI
# ======================================================
pred = model.predict(X_test)
pred_proba = model.predict_proba(X_test)  # for ROC AUC

print("=== Classification Report ===")
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

# ======================================================
# 5. SAVE MODEL
# ======================================================
with open("maternal_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved → maternal_model.pkl")
