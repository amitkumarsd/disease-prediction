import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os

print("--- [Diabetes Phase 1] Loading Data ---")
try:
    df = pd.read_csv('diabetes.csv')
except FileNotFoundError:
    print("Error: 'diabetes.csv' not found.")
    exit()

X = df.drop('Outcome', axis=1)
y = df['Outcome']
print("Diabetes data loaded.")

# Save feature names with a specific prefix
feature_names = list(X.columns)
joblib.dump(feature_names, 'diabetes_features.joblib')
print(f"Diabetes features saved: {feature_names}")

print("\n--- [Diabetes Phase 2] Preprocessing ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- [Diabetes Phase 3] Training ---")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
print("Diabetes model trained.")

print("\n--- Diabetes Model Evaluation ---")
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

print("\n--- [Diabetes Phase 4] Saving Artifacts ---")
# Notice the 'diabetes_' prefix so we don't overwrite the heart model
joblib.dump(model, 'diabetes_model.joblib')
joblib.dump(scaler, 'diabetes_scaler.joblib')
print("Success! Diabetes artifacts saved.")