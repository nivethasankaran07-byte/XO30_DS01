# PS01 — Explainable Predictive Equipment Health & Early Risk Triage System

## 📌 Executive Summary
An end-to-end industrial IoT predictive maintenance system designed to estimate continuous equipment failure risk, classify machines into actionable risk tiers, explain root-cause sensor anomalies, and automatically generate a prioritized maintenance triage schedule[span_0](start_span)[span_0](end_span)[span_1](start_span)[span_1](end_span).

---

## 🎯 Key Differentiators (Why This Solves PS01)

1. **Continuous Risk Scoring Over Naive Binary Labels:** Instead of an arbitrary 0/1 prediction, the system outputs calibrated probabilities representing real-world failure likelihood[span_2](start_span)[span_2](end_span)[span_3](start_span)[span_3](end_span).
2. **Physics-Informed Feature Engineering:** Models thermodynamic and mechanical degradation drivers:
   - **Thermal Gradient ($\Delta T$):** $\text{Process Temperature} - \text{Air Temperature}$
   - **Mechanical Power Output ($kW$):** $\frac{2\pi \times \text{RPM} \times \text{Torque}}{60,000}$
   - **Dynamic Wear Strain:** $\text{Tool Wear} \times \text{Torque}$
3. **Leak-Free Imbalance Handling:** SMOTE oversampling is applied **strictly to the training fold**[span_4](start_span)[span_4](end_span)[span_5](start_span)[span_5](end_span). The test evaluation preserves the true operational failure rarity (~3.7%) to prevent synthetic metric inflation[span_6](start_span)[span_6](end_span)[span_7](start_span)[span_7](end_span).
4. **Asymmetric Cost-Optimal Decision Threshold:** In industrial settings, an unplanned catastrophic breakdown costs significantly more than an inspection[span_8](start_span)[span_8](end_span)[span_9](start_span)[span_9](end_span). We tune the operational threshold across the Precision-Recall curve using an asymmetric cost ratio ($\text{Cost}(FN) : \text{Cost}(FP) = 6.67 : 1$) rather than defaulting to 0.5[span_10](start_span)[span_10](end_span).
5. **Probability Calibration:** Raw gradient-boosted trees output uncalibrated ranking scores[span_11](start_span)[span_11](end_span)[span_12](start_span)[span_12](end_span). We apply **Isotonic Regression** (`CalibratedClassifierCV`) so a 75% score reflects an empirical 75% breakdown likelihood[span_13](start_span)[span_13](end_span)[span_14](start_span)[span_14](end_span).
6. **Instance-Level Root-Cause Explainability:** Each high-risk prediction is explained by measuring standard-deviation ($\sigma$) divergence from normal equipment operating profiles, telling operators *why* a machine is failing[span_15](start_span)[span_15](end_span)[span_16](start_span)[span_16](end_span).
7. **Interactive Technician Dashboard:** Includes a live Streamlit interface for manual simulation, telemetry inspection, and instant priority triage.

---

## 📂 Repository Structure

```text
equipment_health_project/
├── data/
│   └── equipment_data.csv               # Industrial telemetry dataset
├── full_pipeline.py                     # Complete end-to-end ML & triage pipeline
├── app.py                               # Interactive Streamlit dashboard
├── maintenance_triage_action_plan.csv   # Exported prioritized maintenance queue
└── README.md                            # System documentation

⚙️ Operational Architecture
1. Data Cleaning & Physics Preprocessing
 * Ingests all sensor telemetry (Air Temperature, Process Temperature, Rotational Speed, Torque, Tool Wear, Type).
 * Filters OCR noise and corrupt readings outside physical tolerances without discarding true anomalies leading to failure.
2. Multi-Tier Risk Protocol & Priority Dispatch
 * 🔴 Critical (\ge 70\%): P1 — Emergency halt; immediate spindle and component overhaul.
 * 🟠 High (\ge \text{Cost-Tuned Threshold}): P2 — Priority dispatch; inspect tooling and cooling within 8 hours.
 * 🟡 Medium (\ge 20\%): P3 — Scheduled check; verify lubrication and alignment on next shift.
 * 🟢 Low (< 20\%): P4 — Nominal operation; continue automated telemetry streaming.
3. Prediction Uncertainty
Confidence is evaluated via distance from the operational decision boundary:


Predictions with a margin < 0.15 are tagged as Borderline / Low Confidence for secondary manual review.
4. Metric Justification
Accuracy is deliberately omitted as a primary metric because predicting all zeros yields ~96.3% accuracy while missing 100% of breakdowns. The pipeline evaluates on:
 * Failure Recall (Minimizing critical False Negatives)
 * ROC-AUC & PR-AUC (Threshold-agnostic discrimination quality)
 * Brier Score Loss (Calibrated probability reliability)
 * F_2-Score (Harmonic mean weighting recall over precision)
🚀 How to Run
Step 1: Install Dependencies
python -m pip install pandas numpy scikit-learn streamlit

Step 2: Run Full ML Pipeline & Generate Triage Report
python full_pipeline.py

Executes stratified cross-validation, probability calibration, cost-threshold tuning, and writes maintenance_triage_action_plan.csv.
Step 3: Launch the Interactive Web Dashboard
python -m streamlit run app.py

Opens http://localhost:8501 to simulate sensor values, view calibrated risk scores, and inspect root causes live.
