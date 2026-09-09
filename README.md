# PS01 — Predictive Equipment Health: An Explainable Risk Assessment System

## Status
🚧 **Work in progress — second commit.** Baseline (Phase 1) plus full
imbalance handling, boosted-tree model with calibration, cost-justified
threshold, risk tiers, per-alert explanations, and uncertainty margin
(Phases 2–8) are implemented in `notebooks/full_pipeline.py`.

**A note on tooling:** this was built in a sandboxed environment with
no internet access, so `xgboost`, `imbalanced-learn`, and `shap`
couldn't be pip-installed. Functionally equivalent substitutes were
built instead, with clear "drop-in replacement" instructions in each
file's docstring:

| Wanted | Used instead | File |
|---|---|---|
| SMOTE | Manual KNN-based SMOTE | `notebooks/smote_utils.py` |
| XGBoost | `HistGradientBoostingClassifier` (also boosted trees) | `notebooks/full_pipeline.py` |
| SHAP | Permutation importance (global) + z-score per-instance explanation (local) | `notebooks/explain_utils.py` |

If you have internet access, swap these one-for-one — the pipeline
structure (train/test split → resample train only → calibrate →
threshold-tune → explain) stays identical.

## Problem
Estimate equipment failure risk from historical operating measurements,
in a way that's interpretable enough to support maintenance decisions —
not just a failure/no-failure label. See `PS01 problem statement` for
full requirements.

## Dataset
`data/equipment_data.csv` — equipment operating records with:
- `Equipment_Record_ID`, `Type` (L/M/H)
- `Air_temperature_K`, `Process_temperature_K`
- `Rotational_speed_rpm`, `Torque_Nm`, `Tool_wear_min`
- `Machine_failure` (target, 0/1)

Note: source was a scanned/exported PDF table; a small number of rows
have parsing noise, handled via range-based flagging in the cleaning
step (see script). This will be revisited with a more robust
extraction pass.

## Approach

### Phase 1 — Baseline (`notebooks/eda_and_baseline.py`)
EDA, range-based cleaning, Random Forest with `class_weight="balanced"`,
basic feature importances. Result: ROC-AUC 0.95, but recall on the
failure class was only 0.29 (default 0.5 threshold, no resampling).

### Phase 2 — Imbalance handling (`notebooks/smote_utils.py`)
Failure rate is ~3.7% of records. SMOTE is applied to the **training
split only** (never the test set, to avoid leaking synthetic signal
into evaluation) to bring the training classes to parity.

### Phase 3 — Model (`notebooks/full_pipeline.py`)
- `HistGradientBoostingClassifier` (boosted trees, xgboost-equivalent
  for this sandbox)
- 5-fold stratified cross-validation on the *original* (non-resampled)
  data, since only ~250 failure examples exist total — a single split
  would be too noisy to trust
- `CalibratedClassifierCV` (isotonic) so output probabilities are
  genuine risk estimates, not just ranking scores

### Phase 4 — Risk tiers
Probability → Low (<0.2) / Medium (0.2–threshold) / High (≥threshold),
where the threshold comes from Phase 5, not an arbitrary cutoff.

### Phase 5 — Decision threshold
Default 0.5 is not used. Instead: **assumption stated explicitly** —
a missed failure (false negative) is 5x costlier than a false alarm
(false positive), reflecting unplanned downtime/safety risk vs. an
unnecessary inspection. The threshold that minimizes this weighted
cost over the precision-recall curve is selected (~0.42 in the current
run — recall jumps from 0.29 → 0.75 vs. the Phase-1 baseline).

### Phase 6 — Explainability (`notebooks/explain_utils.py`)
- Global: permutation importance (F1-scoring) — model-agnostic ranking
- Per-alert: each high-risk prediction is explained by which features
  deviate most (in std units) from the normal-operation reference
  distribution, e.g. *"Torque_Nm is 2.4 std above normal"*

### Phase 7 — Uncertainty
Calibrated probability's distance from the decision threshold is used
as a confidence proxy — predictions close to the threshold are flagged
as low-confidence and may warrant manual review.

### Phase 8 — Evaluation
Accuracy is never the headline metric (a "never fails" model would
score ~96% accuracy with 0 recall — misleading given the imbalance).
Reported instead: precision/recall/F1 on the failure class, ROC-AUC,
confusion matrix, and 5-fold CV variance.

## Roadmap (remaining)
- [ ] Swap manual substitutes for xgboost/imbalanced-learn/shap once
      running with internet access
- [ ] Hyperparameter tuning (grid/random search) on the boosted model
- [ ] More rigorous missing-data imputation (e.g. `sklearn.KNNImputer`)
      in place of median fill
- [ ] Package inference as a small script/API that takes new readings
      and returns risk tier + explanation
- [ ] Formal writeup / slides summarizing results for submission

## How to run
```bash
pip install pandas numpy scikit-learn
cd notebooks
python eda_and_baseline.py
```

## Team notes
Committing early with a working baseline so there's something concrete
to build on — cleaning, explainability, and threshold work continue
next session.
