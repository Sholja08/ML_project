import pandas as pd
import numpy as np
import joblib

df = pd.read_csv("diabetes.csv")
print("\n", df.head())
print("\n", df.shape)
print("\n", df.info())
print("\n", df.describe())
print("\n", df.isnull().sum())

# Check disguised zeros
cols_to_check = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols_to_check:
    zero_count = (df[col] == 0).sum()
    zero_pct = (zero_count / len(df)) * 100
    print(f"{col}: {zero_count} zeros ({zero_pct:.2f}%)")

print(df["Insulin"].mean())
print(df["Insulin"].median())

# Clean: convert disguised zeros to NaN, then fill with median
for col in cols_to_check:
    df[col] = df[col].replace(0, np.nan)
print(df.isnull().sum())

for col in cols_to_check:
    df[col] = df[col].fillna(df[col].median())
print(df.isnull().sum())

for col in cols_to_check:
    print(f"{col} -> min: {df[col].min()}, missing: {df[col].isnull().sum()}")

# Split features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(X_train.shape, y_train.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# ---------- Logistic Regression (final chosen model) ----------
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression(class_weight="balanced", max_iter=1000)
log_model.fit(X_train_scaled, y_train)
log_pred = log_model.predict(X_test_scaled)
log_probs = log_model.predict_proba(X_test_scaled)[:, 1]

print("\n=== Logistic Regression (Final Model) ===")
print("Accuracy:", accuracy_score(y_test, log_pred))
print(confusion_matrix(y_test, log_pred))
print(classification_report(y_test, log_pred))
print("ROC-AUC:", roc_auc_score(y_test, log_probs))

# Note: Random Forest and XGBoost were also tested during model selection.
# Logistic Regression was chosen for better interpretability and simpler
# deployment, with comparable performance on this dataset size (768 rows).

joblib.dump(
    {"model": log_model, "scaler": scaler, "columns": list(X.columns)},
    "diabetes_model.pkl"
)
print("\nModel saved as diabetes_model.pkl")
