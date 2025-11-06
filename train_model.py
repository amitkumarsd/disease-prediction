import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

print("--- [Phase 1] Loading Data ---")
try:
    df = pd.read_csv('heart.csv')
except FileNotFoundError:
    print("Error: 'heart.csv' not found. Ensure it is in this folder.")
    exit()

X = df.drop('target', axis=1)
y = df['target']
print("Data loaded.")

# Save feature names to ensure correct input order later
feature_names = list(X.columns)
joblib.dump(feature_names, 'feature_names.joblib')

print("\n--- [Phase 2] Preprocessing ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- [Phase 3] Training ---")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
print("Model trained.")

print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred, target_names=['No Disease', 'Disease']))

print("\n--- [Phase 4] Saving ---")
joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')
print("Success! Artifacts saved.")