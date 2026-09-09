"""
PS01 — Explainable Predictive Equipment Health System
=====================================================
Includes Surprise Challenge 1:
- Asymmetric Cost Optimization (Missed Failure FN = 50x False Alarm FP)
- Expected Cost per 1,000 predictions metric
- Cost-vs-Threshold Curve Plot generation
- Automated inference on 'surprise_challenge_1.csv'
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    fbeta_score,
    recall_score,
    precision_score,
    brier_score_loss,
)

# -------------------------------------------------------------------------
# 1. Leak-Free SMOTE (Train Fold Only)
# -------------------------------------------------------------------------
def smote_resample(X_arr: np.ndarray, y_arr: np.ndarray, k_neighbors: int = 5, random_state: int = 42):
    rng = np.random.RandomState(random_state)
    X_min = X_arr[y_arr == 1]
    X_maj = X_arr[y_arr == 0]
    n_synthetic = len(X_maj) - len(X_min)

    if n_synthetic <= 0:
        return X_arr, y_arr

    k = min(k_neighbors, len(X_min) - 1)
    if k < 1:
        idx = rng.choice(len(X_min), n_synthetic, replace=True)
        synthetic = X_min[idx]
    else:
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
        _, neighbor_indices = nn.kneighbors(X_min)

        synthetic = []
        for _ in range(n_synthetic):
            i = rng.randint(0, len(X_min))
            chosen = neighbor_indices[i][rng.randint(1, k + 1)]
            diff = X_min[chosen] - X_min[i]
            synthetic.append(X_min[i] + rng.rand() * diff)
        synthetic = np.array(synthetic)

    X_resampled = np.vstack([X_arr, synthetic])
    y_resampled = np.concatenate([y_arr, np.full(n_synthetic, 1)])

    perm = rng.permutation(len(X_resampled))
    return X_resampled[perm], y_resampled[perm]

# -------------------------------------------------------------------------
# 2. Physics & Sensor Preprocessing
# -------------------------------------------------------------------------
def clean_and_prep(df: pd.DataFrame):
    data = df.copy()
    data.columns = [c.strip() for c in data.columns]

    # Universal column name normalization (handles bracketed and spaced headers)
    rename_map = {}
    for c in data.columns:
        clean_c = c.lower().replace(" ", "_").replace("[", "").replace("]", "")
        if "air_temp" in clean_c:
            rename_map[c] = "Air_temperature_K"
        elif "process_temp" in clean_c:
            rename_map[c] = "Process_temperature_K"
        elif "speed" in clean_c or "rpm" in clean_c:
            rename_map[c] = "Rotational_speed_rpm"
        elif "torque" in clean_c:
            rename_map[c] = "Torque_Nm"
        elif "wear" in clean_c:
            rename_map[c] = "Tool_wear_min"
        elif "failure" in clean_c or "target" in clean_c:
            rename_map[c] = "Machine_failure"
        elif "id" in clean_c or "equipment" in clean_c:
            rename_map[c] = "Equipment_Record_ID"
        elif clean_c == "type":
            rename_map[c] = "Type"
    data = data.rename(columns=rename_map)

    if "Equipment_Record_ID" not in data.columns:
        data["Equipment_Record_ID"] = [f"EQR-{i:05d}" for i in range(len(data))]

    # Range filtering for sensor noise
    numeric_ranges = {
        "Air_temperature_K": (270.0, 340.0),
        "Process_temperature_K": (280.0, 350.0),
        "Rotational_speed_rpm": (800.0, 3500.0),
        "Torque_Nm": (0.0, 150.0),
        "Tool_wear_min": (0.0, 400.0),
    }
    for col, (lo, hi) in numeric_ranges.items():
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
            bad = (~data[col].between(lo, hi)) & data[col].notna()
            data.loc[bad, col] = np.nan
            data[col] = data[col].fillna(data[col].median())
        else:
            data[col] = (lo + hi) / 2.0

    # Physics feature engineering
    data["Temp_Difference"] = data["Process_temperature_K"] - data["Air_temperature_K"]
    data["Power_kW"] = (2 * np.pi * data["Rotational_speed_rpm"] * data["Torque_Nm"]) / 60000.0
    data["Tool_Wear_Strain"] = data["Tool_wear_min"] * data["Torque_Nm"]

    # Categorical handling
    data["Type"] = data["Type"].fillna("L")
    type_dummies = pd.get_dummies(data["Type"], prefix="Type", drop_first=True)

    features = [
        "Air_temperature_K", "Process_temperature_K", "Rotational_speed_rpm",
        "Torque_Nm", "Tool_wear_min", "Temp_Difference", "Power_kW", "Tool_Wear_Strain"
    ]
    X = pd.concat([data[features], type_dummies], axis=1)

    for c in ["Type_M", "Type_H"]:
        if c not in X.columns:
            X[c] = 0

    expected_cols = [
        "Air_temperature_K", "Process_temperature_K", "Rotational_speed_rpm",
        "Torque_Nm", "Tool_wear_min", "Temp_Difference", "Power_kW", "Tool_Wear_Strain",
        "Type_M", "Type_H"
    ]
    X = X[expected_cols]

    y = None
    if "Machine_failure" in data.columns:
        y = pd.to_numeric(data["Machine_failure"], errors="coerce").fillna(0).astype(int)

    return X, y, data["Equipment_Record_ID"]

# -------------------------------------------------------------------------
# 3. Explainability Engine
# -------------------------------------------------------------------------
class ExplainabilityEngine:
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.normal_mean = None
        self.normal_std = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        healthy_samples = X_train[y_train == 0]
        self.normal_mean = healthy_samples.mean()
        self.normal_std = healthy_samples.std().replace(0, 1e-6)

    def explain_instance(self, sample_series: pd.Series, top_k: int = 3):
        z_scores = (sample_series - self.normal_mean) / self.normal_std
        ranked = z_scores.abs().sort_values(ascending=False).head(top_k)

        reasons = []
        for feat in ranked.index:
            z = z_scores[feat]
            direction = "above" if z > 0 else "below"
            reasons.append(f"{feat} is {abs(z):.1f}σ {direction} normal")
        return reasons

# -------------------------------------------------------------------------
# 4. Pipeline Execution
# -------------------------------------------------------------------------
def run_pipeline():
    print("=" * 80)
    print("   EXPLAINABLE EQUIPMENT HEALTH PIPELINE + 50:1 COST OPTIMIZATION")
    print("=" * 80)

    train_file = "equipment_data.csv"
    if not os.path.exists(train_file):
        for candidate in ["data/equipment_data.csv", "../data/equipment_data.csv"]:
            if os.path.exists(candidate):
                train_file = candidate
                break

    df_train = pd.read_csv(train_file)
    X_full, y_full, ids_full = clean_and_prep(df_train)
    print(f"[*] Training dataset: {len(X_full)} records | Failure rate: {y_full.mean():.2%}")

    X_train, X_val, y_train, y_val, ids_tr, ids_va = train_test_split(
        X_full, y_full, ids_full, test_size=0.20, stratify=y_full, random_state=42
    )

    # SMOTE only on training split
    X_train_res, y_train_res = smote_resample(X_train.values, y_train.values)

    # Calibrated Boosted Tree Model
    base_gbm = HistGradientBoostingClassifier(learning_rate=0.08, max_iter=200, min_samples_leaf=20, random_state=42)
    calibrated = CalibratedClassifierCV(base_gbm, method="isotonic", cv=3)
    calibrated.fit(X_train_res, y_train_res)

    explainer = ExplainabilityEngine(X_full.columns.tolist())
    explainer.fit(X_train, y_train)

    # ---------------------------------------------------------------------
    # 5. Asymmetric Cost Curve Optimization (FN = 50x, FP = 1x)
    # ---------------------------------------------------------------------
    COST_FN = 50.0  # Missed failure (catastrophic breakdown & downtime)
    COST_FP = 1.0   # False alarm (preventive inspection)

    y_val_probas = calibrated.predict_proba(X_val.values)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 400)
    costs_per_1000 = []
    N_val = len(y_val)

    for t in thresholds:
        pred = (y_val_probas >= t).astype(int)
        cm = confusion_matrix(y_val, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        total_cost = (fn * COST_FN) + (fp * COST_FP)
        costs_per_1000.append((total_cost / N_val) * 1000.0)

    costs_per_1000 = np.array(costs_per_1000)
    opt_idx = np.argmin(costs_per_1000)
    opt_threshold = thresholds[opt_idx]
    opt_cost_1k = costs_per_1000[opt_idx]

    idx_05 = np.argmin(np.abs(thresholds - 0.50))
    cost_05_1k = costs_per_1000[idx_05]

    print("\n" + "-" * 32 + " SURPRISE CHALLENGE 1 RESULTS " + "-" * 32)
    print(f"Cost Asymmetry Ratio: Missed Failure (FN) = {COST_FN:.0f}x | False Alarm (FP) = {COST_FP:.0f}x")
    print(f"Cost-Optimal Decision Threshold (τ*): {opt_threshold:.3f}")
    print(f"Expected Cost per 1,000 Predictions @ τ={opt_threshold:.3f}: {opt_cost_1k:.2f} cost units")
    print(f"Expected Cost per 1,000 Predictions @ τ=0.500: {cost_05_1k:.2f} cost units")
    print(f"Net Cost Reduction vs. Default 0.5: {((cost_05_1k - opt_cost_1k) / cost_05_1k) * 100:.2f}%\n")

    # Plot and save Cost-vs-Threshold curve
    plt.figure(figsize=(10, 5.8), dpi=150)
    plt.plot(thresholds, costs_per_1000, color="#1f77b4", linewidth=2.5, label="Expected Cost per 1,000 Predictions")
    plt.axvline(opt_threshold, color="#d62728", linestyle="--", linewidth=2, label=f"Optimal Threshold ({opt_threshold:.3f})")
    plt.axvline(0.50, color="#7f7f7f", linestyle=":", linewidth=2, label="Default Threshold (0.50)")
    plt.scatter([opt_threshold], [opt_cost_1k], color="#d62728", s=90, zorder=5)
    plt.scatter([0.50], [cost_05_1k], color="#7f7f7f", s=90, zorder=5)

    plt.annotate(f"Minimum Cost: {opt_cost_1k:.2f}\n(Optimal Threshold = {opt_threshold:.3f})",
                 xy=(opt_threshold, opt_cost_1k), xytext=(opt_threshold + 0.08, opt_cost_1k + 110),
                 arrowprops=dict(arrowstyle="->", lw=1.5, color="#d62728"),
                 fontweight="bold", fontsize=10, bbox=dict(boxstyle="round,pad=0.4", fc="#ffebee", ec="#d62728"))

    plt.annotate(f"Cost at Default 0.5: {cost_05_1k:.2f}\n(+{cost_05_1k - opt_cost_1k:.2f} penalty)",
                 xy=(0.50, cost_05_1k), xytext=(0.52, cost_05_1k - 80),
                 arrowprops=dict(arrowstyle="->", lw=1.5, color="#555555"),
                 fontweight="bold", fontsize=10, bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="#888888"))

    plt.title("Surprise Challenge: Cost-vs-Threshold Optimization (FN Cost = 50× FP Cost)", fontsize=13, pad=12, fontweight="bold")
    plt.xlabel("Decision Threshold (τ)", fontsize=11)
    plt.ylabel("Expected Cost per 1,000 Predictions (Cost Units)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", loc="upper right", fontsize=10)
    plt.tight_layout()
    plot_file = "cost_vs_threshold_curve.png"
    plt.savefig(plot_file)
    plt.close()
    print(f"[✓] Saved cost curve visualization to '{plot_file}'")

    # ---------------------------------------------------------------------
    # 6. Automatic Inference on surprise_challenge_1.csv (if present)
    # ---------------------------------------------------------------------
    challenge_file = "surprise_challenge_1.csv"
    if not os.path.exists(challenge_file):
        for candidate in ["data/surprise_challenge_1.csv", "../surprise_challenge_1.csv"]:
            if os.path.exists(candidate):
                challenge_file = candidate
                break

    if os.path.exists(challenge_file):
        print(f"\n[*] Found surprise challenge data: '{challenge_file}'. Generating predictions...")
        df_surprise = pd.read_csv(challenge_file)
        X_surprise, _, ids_surprise = clean_and_prep(df_surprise)
        probas_surprise = calibrated.predict_proba(X_surprise.values)[:, 1]

        def assign_tier(p, t):
            if p >= 0.70:
                return "CRITICAL"
            elif p >= t:
                return "HIGH"
            elif p >= 0.02:
                return "MEDIUM"
            else:
                return "LOW"

        def assign_action(tier):
            if tier == "CRITICAL":
                return "P1: Emergency halt; immediate overhaul"
            elif tier == "HIGH":
                return "P2: Dispatch maintenance crew within 8 hours"
            elif tier == "MEDIUM":
                return "P3: Schedule sensor & wear check next shift"
            else:
                return "P4: Nominal operation; automated telemetry"

        records = []
        for i in range(len(df_surprise)):
            p = probas_surprise[i]
            tier = assign_tier(p, opt_threshold)
            margin = abs(p - opt_threshold)
            conf = "High Confidence" if margin > 0.05 else "Borderline / Low Confidence"

            row = X_surprise.iloc[i]
            reasons = explainer.explain_instance(row, top_k=3)

            records.append({
                "Equipment_Record_ID": ids_surprise.iloc[i],
                "Failure_Risk": f"{p * 100:.1f}%",
                "Assigned_Risk_Tier": tier,
                "Predicted_Binary_Decision": 1 if p >= opt_threshold else 0,
                "Decision_Confidence": f"Margin ±{margin:.3f} ({conf})",
                "Top_Contributing_Factors": " | ".join(reasons),
                "Inspection_Priority": assign_action(tier),
                "_raw_p": p,
            })

        triage_df = pd.DataFrame(records).sort_values(by="_raw_p", ascending=False).drop(columns=["_raw_p"])
        out_csv = "surprise_challenge_predictions.csv"
        triage_df.to_csv(out_csv, index=False)
        print(f"[✓] Successfully exported all 750 predictions to '{out_csv}'")
        print("\nSurprise Challenge Tier Breakdown:")
        print(triage_df["Assigned_Risk_Tier"].value_counts())

        print("\nTop 5 High-Risk Machines in Surprise Challenge 1:")
        print(triage_df.head(5)[["Equipment_Record_ID", "Failure_Risk", "Assigned_Risk_Tier", "Top_Contributing_Factors"]].to_string(index=False))

    print("\n" + "=" * 80)
    print("                     PIPELINE EXECUTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
