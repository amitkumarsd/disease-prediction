from flask import Flask, request, render_template, session, redirect
import joblib
import numpy as np
import shap
import os
import time
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_this_project'

# --- Get Base Directory ---
base_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(base_dir, 'static')
os.makedirs(static_dir, exist_ok=True)

# --- Load Heart Disease Artifacts (Using their specific names) ---
try:
    heart_model = joblib.load(os.path.join(base_dir, 'model.joblib'))
    heart_scaler = joblib.load(os.path.join(base_dir, 'scaler.joblib'))
    heart_feats = joblib.load(os.path.join(base_dir, 'feature_names.joblib')) # <-- This is the fix
    heart_explainer = shap.TreeExplainer(heart_model)
    print("✅ Heart Disease artifacts loaded successfully.")
except FileNotFoundError:
    print("⚠️ Warning: Heart Disease artifacts not found. Run 'train_model.py' first.")
    heart_model, heart_scaler, heart_feats, heart_explainer = None, None, None, None

# --- Load Diabetes Artifacts (Using their specific names) ---
try:
    diab_model = joblib.load(os.path.join(base_dir, 'diabetes_model.joblib'))
    diab_scaler = joblib.load(os.path.join(base_dir, 'diabetes_scaler.joblib'))
    diab_feats = joblib.load(os.path.join(base_dir, 'diabetes_features.joblib'))
    diab_explainer = shap.TreeExplainer(diab_model)
    print("✅ Diabetes artifacts loaded successfully.")
except FileNotFoundError:
    print("⚠️ Warning: Diabetes artifacts not found. Run 'train_diabetes.py' first.")
    diab_model, diab_scaler, diab_feats, diab_explainer = None, None, None, None

# ===========================
# PAGE ROUTES
# ===========================
@app.route('/')
def home(): return render_template('index.html')

@app.route('/personal_info')
def personal_info():
    return render_template('personal_info.html', target=request.args.get('next', 'home'))

@app.route('/save_personal_info', methods=['POST'])
def save_personal_info():
    session['user_name'] = request.form.get('full_name')
    target = request.form.get('target_disease')
    if target == 'heart': return redirect('/heart')
    if target == 'diabetes': return redirect('/diabetes')
    return redirect('/')

@app.route('/heart')
def heart_page(): return render_template('heart_wizard.html')

@app.route('/diabetes')
def diabetes_page(): return render_template('diabetes_wizard.html')

# ===========================
# GENERIC PREDICTION HELPER
# ===========================
def make_prediction(model, scaler, feats, explainer, form_data, disease_name):
    # This function now checks if the model exists before trying to use it
    if model is None or scaler is None or feats is None or explainer is None:
        return f"Error: The model for {disease_name} is not loaded. Please check server logs."

    try:
        features = [float(form_data[name]) for name in feats]
        features_scaled = scaler.transform(np.array([features]))
        prediction = int(model.predict(features_scaled)[0])
        confidence = round(model.predict_proba(features_scaled)[0][1] * 100, 2)

        if prediction == 1:
            res_class, pred_text = "high-risk", f"HIGH RISK OF {disease_name.upper()}"
            rec = "Strongly recommend consulting a specialist immediately."
        else:
            res_class, pred_text = "low-risk", f"LOW RISK OF {disease_name.upper()}"
            rec = "Maintain a healthy lifestyle and regular check-ups."

        # SHAP Plot
        shap_values = explainer.shap_values(features_scaled)
        base_val = explainer.expected_value[1] if hasattr(explainer.expected_value, '__len__') and len(explainer.expected_value) > 1 else explainer.expected_value
        if isinstance(shap_values, list): patient_shap = shap_values[1][0]
        elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3: patient_shap = shap_values[0, :, 1]
        else: patient_shap = shap_values[0]

        force_plot = shap.force_plot(base_val, patient_shap, np.array(features), feature_names=feats, matplotlib=False, show=False)
        plot_file = f'shap_{disease_name.replace(" ", "_")}_{int(time.time())}.html'
        shap.save_html(os.path.join(static_dir, plot_file), force_plot)

        return render_template('result.html', result_class=res_class, prediction_text=pred_text,
                             confidence=f"{confidence}%", recommendation=rec, shap_plot_url=f'/static/{plot_file}')
    except Exception as e: return f"Error during prediction: {e}"

# ===========================
# PREDICTION ROUTES
# ===========================
@app.route('/predict', methods=['POST'])
def predict_heart():
    return make_prediction(heart_model, heart_scaler, heart_feats, heart_explainer, request.form, "Heart Disease")

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    return make_prediction(diab_model, diab_scaler, diab_feats, diab_explainer, request.form, "Diabetes")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)