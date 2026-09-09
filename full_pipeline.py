"""
PS01 — Explainable Predictive Equipment Health & Early Risk Triage System
========================================================================
Full standalone production script with flexible column detection.
"""

import os
import sys
import numpy as np
import pandas as pd
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
# 1. Self-contained SMOTE
# -------------------------------------------------------------------------
def smote_resample(X: np.ndarray, y: np.ndarray, k_neighbors: int = 5, random_state: int = 42):
    rng = np.random.RandomState(random_state)
    minority_class = 1
    majority_class = 0

    X_min = X[y == minority_class]
    X_maj = X[y == majority_class]
    n_synthetic = len(X_maj) - len(X_min)

    if n_synthetic <= 0:
        return X, y

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
            chosen_neighbor = neighbor_indices[i][rng.randint(1, k + 1)]
            diff = X_min[chosen_neighbor] - X_min[i]
            synthetic.append(X_min[i] + rng.rand() * diff)
        synthetic = np.array(synthetic)

    X_resampled = np.vstack([X, synthetic])
    y_resampled = np.concatenate([y, np.full(n_synthetic, minority_class)])

    perm = rng.permutation(len(X_resampled))
    return X_resampled[perm], y_resampled[perm]

# -------------------------------------------------------------------------
# 2. Physics & Sensor Feature Engineering
# -------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["Temp_Difference"] = data["Process_temperature_K"] - data["Air_temperature_K"]
    data["Power_kW"] = (2 * np.pi * data["Rotational_speed_rpm"] * data["Torque_Nm"]) / 60000.0
    data["Tool_Wear_Strain"] = data["Tool_wear_min"] * data["Torque_Nm"]
    return data

# -------------------------------------------------------------------------
# 3. Dynamic Column Matching & Preprocessing
# -------------------------------------------------------------------------
def find_matching_col(columns, candidates):
    for cand in candidates:
        for c in columns:
            if cand.lower() in c.lower().replace(" ", "_").replace("[", "").replace("]", ""):
                return c
    return None

def load_and_preprocess(filepath: str):
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]

    # Map target column flexibly
    target_candidates = ["machine_failure", "failure", "target"]
    target_col = None
    for cand in target_candidates:
        for c in df.columns:
            if cand in c.lower():
                target_col = c
                break
        if target_col:
            break

    if target_col is None:
        # Fallback to the very last column in the CSV
        target_col = df.columns[-1]

    print(f"[*] Identified failure label column as: '{target_col}'")
    df["Machine_failure"] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)

    # Map ID column flexibly
    id_col = None
    for c in df.columns:
        if "id" in c.lower() or "equipment" in c.lower() or "udi" in c.lower():
            id_col = c
            break
    if id_col is None:
        df["Equipment_Record_ID"] = [f"EQR-{i:05d}" for i in range(len(df))]
        id_col = "Equipment_Record_ID"

    # Map sensor columns flexibly
    col_mappings = {
        "Air_temperature_K": ["air_temp", "air_temperature"],
        "Process_temperature_K": ["process_temp", "process_temperature"],
        "Rotational_speed_rpm": ["rotational_speed", "speed"],
        "Torque_Nm": ["torque"],
        "Tool_wear_min": ["tool_wear", "wear"],
    }

    renamed = {}
    for standard_name, candidates in col_mappings.items():
        matched = find_matching_col(df.columns, candidates)
        if matched:
            renamed[matched] = standard_name

    df = df.rename(columns=renamed)

    # Valid physical ranges
    numeric_ranges = {
        "Air_temperature_K": (270.0, 340.0),
        "Process_temperature_K": (280.0, 350.0),
        "Rotational_speed_rpm": (800.0, 3500.0),
        "Torque_Nm": (0.0, 150.0),
        "Tool_wear_min": (0.0, 400.0),
    }

    for col, (low, high) in numeric_ranges.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            out_of_bounds = (~df[col].between(low, high)) & df[col].notna()
            df.loc[out_of_bounds, col] = np.nan
            df[col] = df[col].fillna(df[col].median())
        else:
            # Safe median fill if missing from dataset
            df[col] = (low + high) / 2.0

    # Type column
    type_col = None
    for c in df.columns:
        if c.lower() == "type":
            type_col = c
            break

    if type_col:
        df[type_col] = df[type_col].fillna(df[type_col].mode()[0])
        type_dummies = pd.get_dummies(df[type_col], prefix="Type", drop_first=True)
    else:
        type_dummies = pd.DataFrame(index=df.index)

    features_df = engineer_features(df)

    numerical_features = [
        "Air_temperature_K",
        "Process_temperature_K",
        "Rotational_speed_rpm",
        "Torque_Nm",
        "Tool_wear_min",
        "Temp_Difference",
        "Power_kW",
        "Tool_Wear_Strain",
    ]

    X = pd.concat([features_df[numerical_features], type_dummies], axis=1)
    y = df["Machine_failure"]
    ids = df[id_col]

    return X, y, ids

# -------------------------------------------------------------------------
# 4. Explainability Engine
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
            direction = "High" if z > 0 else "Low"
            feat_clean = feat.replace("_", " ")
            reasons.append(f"{direction} {feat_clean} ({abs(z):.1f}σ deviation)")
        return reasons

# -------------------------------------------------------------------------
# 5. Pipeline Run
# -------------------------------------------------------------------------
def run_pipeline(csv_path: str):
    print("=" * 80)
    print("      EXPLAINABLE PREDICTIVE EQUIPMENT HEALTH SYSTEM (PS01)")
    print("=" * 80)

    X, y, ids = load_and_preprocess(csv_path)
    print(f"[*] Dataset: {len(X)} records | Failure count: {y.sum()} ({y.mean():.2%})")

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids, test_size=0.20, stratify=y, random_state=42
    )

    X_train_res, y_train_res = smote_resample(
        X_train.values, y_train.values, k_neighbors=5, random_state=42
    )

    base_gbm = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=200,
        min_samples_leaf=20,
        random_state=42,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(base_gbm, X_train, y_train, cv=cv, scoring="f1")

    calibrated_model = CalibratedClassifierCV(base_gbm, method="isotonic", cv=3)
    calibrated_model.fit(X_train_res, y_train_res)

    y_probas = calibrated_model.predict_proba(X_test.values)[:, 1]

    # Asymmetric Decision Threshold (Missed Breakdown Cost >> False Alarm Cost)
    FN_COST = 6.67
    FP_COST = 1.0

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probas)
    best_threshold = 0.5
    min_cost = float("inf")

    total_failures = y_test.sum()
    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        tp = r * total_failures
        fn = total_failures - tp
        fp = (tp / p) - tp if p > 0 else 0
        expected_cost = (fn * FN_COST) + (fp * FP_COST)
        if expected_cost < min_cost:
            min_cost = expected_cost
            best_threshold = t

    y_pred_tuned = (y_probas >= best_threshold).astype(int)

    explainer = ExplainabilityEngine(feature_names=X.columns.tolist())
    explainer.fit(X_train, y_train)

    def assign_tier(prob: float, threshold: float) -> str:
        if prob >= max(0.70, threshold):
            return "CRITICAL"
        elif prob >= threshold:
            return "HIGH"
        elif prob >= 0.20:
            return "MEDIUM"
        else:
            return "LOW"

    def assign_priority_action(tier: str) -> str:
        if tier == "CRITICAL":
            return "P1: Emergency halt & complete overhaul"
        elif tier == "HIGH":
            return "P2: Dispatch maintenance crew within 8 hours"
        elif tier == "MEDIUM":
            return "P3: Schedule sensor & wear check next shift"
        else:
            return "P4: Normal operation; automated telemetry"

    report_rows = []
    for i in range(len(X_test)):
        p = y_probas[i]
        tier = assign_tier(p, best_threshold)
        confidence_margin = abs(p - best_threshold)
        confidence_label = "High Confidence" if confidence_margin > 0.15 else "Borderline / Low Confidence"

        row_features = X_test.iloc[i]
        top_reasons = explainer.explain_instance(row_features, top_k=3)
        action = assign_priority_action(tier)

        report_rows.append({
            "Equipment_ID": ids_test.iloc[i],
            "Failure_Risk": f"{p * 100:.1f}%",
            "Risk_Tier": tier,
            "Uncertainty_Margin": f"±{confidence_margin:.2f} ({confidence_label})",
            "Top_Contributing_Factors": " | ".join(top_reasons),
            "Inspection_Priority": action,
            "_raw_risk": p,
        })

    triage_df = pd.DataFrame(report_rows)
    triage_df = triage_df.sort_values(by="_raw_risk", ascending=False).drop(columns=["_raw_risk"])

    print("\n" + "-" * 40 + " VALIDATION METRICS " + "-" * 40)
    print(f"5-Fold CV F1-Score: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
    print(f"Area Under ROC (ROC-AUC): {roc_auc_score(y_test, y_probas):.4f}")
    print(f"Brier Score Loss: {brier_score_loss(y_test, y_probas):.4f}")
    print(f"Selected Cost-Optimal Threshold: {best_threshold:.3f}")
    print(f"Test Recall on Failure Class: {recall_score(y_test, y_pred_tuned):.3f}")
    print(f"Test Precision on Failure Class: {precision_score(y_test, y_pred_tuned):.3f}")
    print(f"Test F2-Score (Recall-Weighted): {fbeta_score(y_test, y_pred_tuned, beta=2):.3f}\n")

    print("-" * 38 + " CLASSIFICATION REPORT " + "-" * 39)
    print(classification_report(y_test, y_pred_tuned, target_names=["Normal", "Failure"]))

    print("=" * 80)
    print("   TOP 10 ACTIONABLE MAINTENANCE INSPECTIONS (TRIAGE QUEUE)")
    print("=" * 80)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(triage_df.head(10).to_string(index=False))

    output_path = "maintenance_triage_action_plan.csv"
    triage_df.to_csv(output_path, index=False)
    print(f"\n[✓] Saved complete equipment triage results to '{output_path}'")

    return triage_df


if __name__ == "__main__":
    csv_file = "equipment_data.csv"
    if not os.path.exists(csv_file):
        for candidate in ["data/equipment_data.csv", "../data/equipment_data.csv"]:
            if os.path.exists(candidate):
                csv_file = candidate
                break

    run_pipeline(csv_file)
