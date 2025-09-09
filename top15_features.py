# ------------------------------
# Isolation Forest + SHAP (Top 5 Features)
# ------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest

# ------------------------------
# 0. Config / reproducibility
# ------------------------------
np.random.seed(42)
OUT_DIR = "./outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------
# 1. Load dataset
# ------------------------------
df = pd.read_csv("compressed_data.csv")
df['Attack Type'] = df['Attack Type'].str.lower()
if 'target' in df.columns:
    df.drop(['target'], axis=1, inplace=True)

# ------------------------------
# 2. Encode categorical features
# ------------------------------
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    if col != 'Attack Type':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# ------------------------------
# 3. Feature / target split
# ------------------------------
X = df.drop('Attack Type', axis=1)
y = df['Attack Type']

# ------------------------------
# 4. Train-test split + scaling
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# 5. Isolation Forest (unsupervised anomaly detection)
# ------------------------------
iso = IsolationForest(contamination=0.05, random_state=42)
iso.fit(X_train_scaled)

# ------------------------------
# 6. SHAP Explainability for Isolation Forest
# ------------------------------
print("\nGenerating SHAP explanations for Isolation Forest...")
explainer_if = shap.TreeExplainer(iso)
shap_vals_if = explainer_if.shap_values(X_test_scaled[:500])  # limit for speed

# ------------------------------
# 7. Top 5 SHAP Features
# ------------------------------
print("\nExtracting Top 5 SHAP Features...")

# Mean absolute SHAP values across samples
importances = np.mean(np.abs(shap_vals_if), axis=0)
feature_names = X.columns

# Create dataframe
shap_importance_df = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": importances
}).sort_values(by="mean_abs_shap", ascending=False)

# Select top 5
top5 = shap_importance_df.head(5)

# Plot horizontal bar chart
plt.figure(figsize=(8, 5))
plt.barh(top5["feature"], top5["mean_abs_shap"], color="purple")
plt.gca().invert_yaxis()  # Highest at top
plt.xlabel("Mean |SHAP| Value")
plt.title("Top 5 Features - Isolation Forest (SHAP Analysis)")
plt.tight_layout()

# Save + Show
plt.savefig(os.path.join(OUT_DIR, "top5_shap_isolationforest.png"))
plt.show()

print("\nTop 5 SHAP Features (Isolation Forest):")
print(top5)
