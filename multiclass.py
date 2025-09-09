import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import shap

# 1. Load the dataset
df = pd.read_csv("compressed_data.csv")

# 2. Use Attack Type for multi-class classification
# (Categories: normal, dos, probe, r2l, u2r)
df['Attack Type'] = df['Attack Type'].str.lower()

# 3. Drop the 'target' column (fine-grained label)
df.drop(['target'], axis=1, inplace=True)

# 4. Encode categorical features
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    if col != 'Attack Type':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# 5. Define features and target
X = df.drop('Attack Type', axis=1)
y = df['Attack Type']

# Encode attack categories to numerical labels
le_attack = LabelEncoder()
y_encoded = le_attack.fit_transform(y)

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# 7. Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 8. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 9. Predictions
y_pred = model.predict(X_test)

# 10. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(
    y_test, y_pred, target_names=le_attack.classes_))

# 11. Confusion Matrix - Counts + Normalized
conf_mat = confusion_matrix(y_test, y_pred)
conf_mat_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

# Heatmap with counts
plt.figure(figsize=(10,8))
sns.heatmap(conf_mat_norm, annot=conf_mat, fmt='d', cmap="Blues",
            xticklabels=le_attack.classes_, yticklabels=le_attack.classes_,
            cbar_kws={'label': 'Proportion'})
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix: Multi-Class Attack Classification\n(Counts with Row-wise Normalization)")
plt.show()

# Heatmap with percentages
plt.figure(figsize=(10,8))
sns.heatmap(conf_mat_norm, annot=True, fmt=".2f", cmap="YlGnBu",
            xticklabels=le_attack.classes_, yticklabels=le_attack.classes_,
            cbar_kws={'label': 'Proportion'})
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Normalized %)")
plt.show()

# === Correlation Heatmap ===
# === Correlation Heatmap ===
# Select only numeric columns
# === Proper Correlation Heatmap ===
import numpy as np

# Keep only numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])

# Compute correlation matrix
corr_matrix = numeric_df.corr()

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Plot the heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, mask=mask, cmap="coolwarm", annot=True, fmt=".2f",
            cbar_kws={'label': 'Correlation Coefficient'},
            annot_kws={"size": 8})
plt.title("Correlation Heatmap of Numeric Features", fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

