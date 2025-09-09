import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import lime
import lime.lime_tabular
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
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ----------------------------
# 3. Train Random Forest Model
# ----------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# ----------------------------
# 4. Model Evaluation
# ----------------------------
y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print("\nModel Performance Metrics:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

# ----------------------------
# 5. Visualization of Metrics
# ----------------------------
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
values = [accuracy, precision, recall, f1]

plt.figure(figsize=(8, 6))
bars = plt.bar(metrics, values, color=['#4CAF50', '#2196F3', '#FFC107', '#E91E63'])
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Model Performance Metrics (Random Forest)")

# Annotate bars with values
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{height:.2f}",
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.show()

# ----------------------------
# 6. LIME Explanation
# ----------------------------
feature_names = list(X.columns)
class_names = target_encoder.classes_

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=feature_names,
    class_names=class_names,
    mode='classification'
)

# Pick a sample instance to explain
sample_index = 5
exp = explainer.explain_instance(X_test[sample_index], rf_model.predict_proba, num_features=10)

# ----------------------------
# 7. Show Graph Output for LIME
# ----------------------------
fig = exp.as_pyplot_figure()
plt.title(f"LIME Explanation for Instance {sample_index}")
plt.show()

# Print LIME explanation in terminal
print("\nLIME Explanation (text format):")
for feature, weight in exp.as_list():
    print(f"{feature}: {weight:.4f}")
