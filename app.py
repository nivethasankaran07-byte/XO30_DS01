"""
PS01 Interactive Maintenance & Explainability Dashboard
======================================================
Run with: python -m streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from full_pipeline import load_and_preprocess, smote_resample, ExplainabilityEngine
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

st.set_page_config(page_title="Predictive Equipment Health", layout="wide")

st.title("⚙️ Explainable Predictive Equipment Health System")
st.markdown("Early risk detection, probability calibration, and automated maintenance prioritization.")

@st.cache_resource
def train_and_cache_model():
    csv_file = "equipment_data.csv"
    if not os.path.exists(csv_file):
        for candidate in ["data/equipment_data.csv", "../data/equipment_data.csv"]:
            if os.path.exists(candidate):
                csv_file = candidate
                break

    if not os.path.exists(csv_file):
        return None, None, None, None

    X, y, ids = load_and_preprocess(csv_file)
    X_res, y_res = smote_resample(X.values, y.values, random_state=42)

    base = HistGradientBoostingClassifier(random_state=42)
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=3)
    calibrated.fit(X_res, y_res)

    explainer = ExplainabilityEngine(X.columns.tolist())
    explainer.fit(X, y)

    return calibrated, explainer, X.columns.tolist(), X.mean()

model, explainer, feature_names, defaults = train_and_cache_model()

if model is None:
    st.error("Please ensure 'equipment_data.csv' is placed in the project folder or 'data/' folder.")
    st.stop()

# Sidebar: Interactive Machine Testing
st.sidebar.header("🔬 Live Equipment Simulator")

air_temp = st.sidebar.slider("Air Temperature [K]", 280.0, 320.0, float(defaults["Air_temperature_K"]))
proc_temp = st.sidebar.slider("Process Temperature [K]", 290.0, 330.0, float(defaults["Process_temperature_K"]))
rpm = st.sidebar.slider("Rotational Speed [rpm]", 1000, 3000, int(defaults["Rotational_speed_rpm"]))
torque = st.sidebar.slider("Torque [Nm]", 0.0, 100.0, float(defaults["Torque_Nm"]))
wear = st.sidebar.slider("Tool Wear [min]", 0, 300, int(defaults["Tool_wear_min"]))
m_type = st.sidebar.selectbox("Machine Type", ["L", "M", "H"])

# Engineer inputs
temp_diff = proc_temp - air_temp
power = (2 * np.pi * rpm * torque) / 60000.0
strain = wear * torque

# Dynamically construct input sample matching exact model features
sample_series = pd.Series(0.0, index=feature_names)
sample_series["Air_temperature_K"] = air_temp
sample_series["Process_temperature_K"] = proc_temp
sample_series["Rotational_speed_rpm"] = rpm
sample_series["Torque_Nm"] = torque
sample_series["Tool_wear_min"] = wear
sample_series["Temp_Difference"] = temp_diff
sample_series["Power_kW"] = power
sample_series["Tool_Wear_Strain"] = strain

# Activate the matching machine type dummy variable if present
for col in feature_names:
    if col.startswith("Type_"):
        sample_series[col] = 1.0 if col == f"Type_{m_type}" else 0.0

sample_df = pd.DataFrame([sample_series])
risk_prob = model.predict_proba(sample_df.values)[0, 1]

# Dynamic display metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Failure Probability", value=f"{risk_prob * 100:.1f}%")

with col2:
    if risk_prob >= 0.65:
        tier, color = "CRITICAL", "🔴"
    elif risk_prob >= 0.40:
        tier, color = "HIGH", "🟠"
    elif risk_prob >= 0.20:
        tier, color = "MEDIUM", "🟡"
    else:
        tier, color = "LOW", "🟢"
    st.metric(label="Assigned Risk Tier", value=f"{color} {tier}")

with col3:
    margin = abs(risk_prob - 0.40)
    conf = "High Confidence" if margin > 0.15 else "Borderline (Manual Review)"
    st.metric(label="Decision Confidence", value=conf)

st.markdown("---")

col_left, col_right = st.columns(2)
with col_left:
    st.subheader("📋 Root-Cause Factors (Explainability)")
    reasons = explainer.explain_instance(sample_series, top_k=3)
    for r in reasons:
        st.write(f"- **{r}**")

with col_right:
    st.subheader("🛠️ Recommended Action")
    if tier == "CRITICAL":
        st.error("**P1: Emergency Halt.** Critical failure imminent; halt spindle and overhaul tool immediately.")
    elif tier == "HIGH":
        st.warning("**P2: Priority Dispatch.** High risk detected; dispatch maintenance crew within 8 hours.")
    elif tier == "MEDIUM":
        st.info("**P3: Scheduled Check.** Moderate degradation; flag for sensor and wear inspection on next shift.")
    else:
        st.success("**P4: Normal Operation.** Telemetry within normal limits; automated logging continues.")
