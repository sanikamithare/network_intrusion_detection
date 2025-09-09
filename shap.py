import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import lime
import lime.lime_tabular
import shap
import matplotlib.pyplot as plt

# ----------------------------
# 1. Load and Preprocess Data
# ----------------------------
file_path = 'compressed_data.csv'
df = pd.read_csv(file_path)

# Separate features and target
X = df.drop('target', axis=1)  # Change 'target' to your actual target column
y = df['target']

# Encode categorical columns
label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Scale numerical data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# 2. Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ----------------------------
# 3. Train Random Forest Model
# ----------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# ----------------------------
# 4. LIME Explanation
# ----------------------------
feature_names = list(X.columns)
class_names = target_encoder.classes_

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=feature_names,
    class_names=class_names,
    mode='classification'
)

sample_index = 5
exp = explainer.explain_instance(X_test[sample_index], rf_model.predict_proba, num_features=10)

# Show LIME output (graph)
fig = exp.as_pyplot_figure()
plt.title(f"LIME Explanation for Instance {sample_index}", fontsize=14)
plt.tight_layout()
plt.show()

# Print LIME output (text)
print("\nLIME Explanation (text format):")
for feature, weight in exp.as_list():
    print(f"{feature}: {weight:.4f}")

# ----------------------------
# 5. SHAP Explanation
# ----------------------------
explainer_shap = shap.TreeExplainer(rf_model)
shap_values = explainer_shap.shap_values(X_test)

# Set bigger figure size and DPI for clarity
plt.figure(figsize=(14, 8), dpi=150)
shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)

plt.title("SHAP Feature Importance", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()
