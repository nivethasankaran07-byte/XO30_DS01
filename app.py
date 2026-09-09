"""
PS01 Interactive Maintenance, Explainability & Cost Optimization Dashboard
==========================================================================
Run with: python -m streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from full_pipeline import clean_and_prep, smote_resample, ExplainabilityEngine

st.set_page_config(page_title="Predictive Equipment Health", layout="wide")

st.title("⚙️ Explainable Predictive Equipment Health System")
st.markdown("Cost-Optimal Triage (FN=50x FP), Probability Calibration, and Root-Cause Diagnostics.")

@st.cache_resource
def train_and_cache_model():
    train_file = "equipment_data.csv"
    if not os.path.exists(train_file):
        for candidate in ["data/equipment_data.csv", "../data/equipment_data.csv"]:
            if os.path.exists(candidate):
                train_file = candidate
                break

    if not os.path.exists(train_file):
        return None, None, None, None, 0.047, 400.70, 502.10

    df_train = pd.read_csv(train_file)
    X, y, ids = clean_and_prep(df_train)
    X_res, y_res = smote_resample(X.values, y.values, random_state=42)

    base = HistGradientBoostingClassifier(random_state=42)
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=3)
    calibrated.fit(X_res, y_res)

    explainer = ExplainabilityEngine(X.columns.tolist())
    explainer.fit(X, y)

    # Calculate optimal threshold under 50:1 cost ratio
    probas = calibrated.predict_proba(X.values)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 400)
    costs = []
    N = len(y)
    for t in thresholds:
        pred = (probas >= t).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        cost_1k = ((fn * 50.0 + fp * 1.0) / N) * 1000.0
        costs.append(cost_1k)
    costs = np.array(costs)
    opt_idx = np.argmin(costs)
    opt_t = thresholds[opt_idx]
    opt_cost = costs[opt_idx]
    cost_05 = costs[np.argmin(np.abs(thresholds - 0.50))]

    return calibrated, explainer, X.columns.tolist(), X.mean(), opt_t, opt_cost, cost_05

model, explainer, feature_names, defaults, opt_thresh, opt_cost_1k, cost_05_1k = train_and_cache_model()

if model is None:
    st.error("Please ensure 'equipment_data.csv' is in your project folder.")
    st.stop()

# Top Metrics: Cost Optimization Highlight
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Cost-Optimal Threshold", f"τ = {opt_thresh:.3f}", "Tuned for 50:1 Ratio")
with m2:
    st.metric("Cost / 1,000 @ τ*", f"{opt_cost_1k:.1f} units", f"-{((cost_05_1k - opt_cost_1k)/cost_05_1k)*100:.1f}% vs 0.5")
with m3:
    st.metric("Cost / 1,000 @ 0.50", f"{cost_05_1k:.1f} units", "Default Threshold Penalty", delta_color="inverse")
with m4:
    st.metric("Missed Failure Weight", "50× False Alarm", "Industrial Downtime Risk")

st.markdown("---")

tab_sim, tab_curve, tab_surprise = st.tabs(["🔬 Live Machine Simulator", "📈 Cost-vs-Threshold Curve", "📋 Surprise Challenge 1 Results"])

with tab_sim:
    c_left, c_right = st.columns([1, 2])
    with c_left:
        st.subheader("Sensor Input Controls")
        air_temp = st.slider("Air Temperature [K]", 280.0, 320.0, float(defaults["Air_temperature_K"]))
        proc_temp = st.slider("Process Temperature [K]", 290.0, 330.0, float(defaults["Process_temperature_K"]))
        rpm = st.slider("Rotational Speed [rpm]", 1000, 3000, int(defaults["Rotational_speed_rpm"]))
        torque = st.slider("Torque [Nm]", 0.0, 100.0, float(defaults["Torque_Nm"]))
        wear = st.slider("Tool Wear [min]", 0, 300, int(defaults["Tool_wear_min"]))
        m_type = st.selectbox("Machine Type", ["L", "M", "H"])

        temp_diff = proc_temp - air_temp
        power = (2 * np.pi * rpm * torque) / 60000.0
        strain = wear * torque

        sample_series = pd.Series(0.0, index=feature_names)
        sample_series["Air_temperature_K"] = air_temp
        sample_series["Process_temperature_K"] = proc_temp
        sample_series["Rotational_speed_rpm"] = rpm
        sample_series["Torque_Nm"] = torque
        sample_series["Tool_wear_min"] = wear
        sample_series["Temp_Difference"] = temp_diff
        sample_series["Power_kW"] = power
        sample_series["Tool_Wear_Strain"] = strain

        for col in feature_names:
            if col.startswith("Type_"):
                sample_series[col] = 1.0 if col == f"Type_{m_type}" else 0.0

        risk_prob = model.predict_proba(pd.DataFrame([sample_series]).values)[0, 1]

    with c_right:
        st.subheader("Automated Operational Assessment")
        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("Calibrated Failure Risk", f"{risk_prob * 100:.1f}%")
        with k2:
            if risk_prob >= 0.70:
                tier, color = "CRITICAL", "🔴"
            elif risk_prob >= opt_thresh:
                tier, color = "HIGH", "🟠"
            elif risk_prob >= 0.02:
                tier, color = "MEDIUM", "🟡"
            else:
                tier, color = "LOW", "🟢"
            st.metric("Operational Risk Tier", f"{color} {tier}")
        with k3:
            margin = abs(risk_prob - opt_thresh)
            conf = "High Confidence" if margin > 0.05 else "Borderline (Manual Review)"
            st.metric("Decision Confidence", conf)

        st.markdown("#### Root-Cause Factors (Explainability)")
        reasons = explainer.explain_instance(sample_series, top_k=3)
        for r in reasons:
            st.markdown(f"- **{r}**")

        st.markdown("#### Recommended Maintenance Action")
        if tier == "CRITICAL":
            st.error("**P1: Emergency Spindle Halt.** Catastrophic failure imminent; immediate overhaul required.")
        elif tier == "HIGH":
            st.warning(f"**P2: Priority Dispatch.** Risk exceeds optimal threshold ({opt_thresh:.3f}); dispatch crew within 8 hours.")
        elif tier == "MEDIUM":
            st.info("**P3: Scheduled Check.** Moderate degradation; flag for wear check on next shift.")
        else:
            st.success("**P4: Nominal Operation.** Continuous telemetry monitoring.")

with tab_curve:
    st.subheader("Cost Optimization Curve (FN Cost = 50× FP Cost)")
    if os.path.exists("cost_vs_threshold_curve.png"):
        st.image("cost_vs_threshold_curve.png", use_container_width=True)
    else:
        st.info("Run 'python full_pipeline.py' in terminal to generate the high-resolution plot.")

with tab_surprise:
    st.subheader("Surprise Challenge 1: 750 Equipment Predictions")
    pred_file = "surprise_challenge_predictions.csv"
    if os.path.exists(pred_file):
        df_pred = pd.read_csv(pred_file)
        st.dataframe(df_pred, use_container_width=True)
        st.download_button(
            label="📥 Download Surprise Challenge Predictions CSV",
            data=df_pred.to_csv(index=False),
            file_name="surprise_challenge_predictions.csv",
            mime="text/csv",
        )
    else:
        st.info("Run 'python full_pipeline.py' to generate 'surprise_challenge_predictions.csv'.")
