"""
Final LightGBM bias correction with station-grouped five-fold OOF.

Uses the 63/16 development–test split and the hyperparameters selected in
the search stage. The independent test set is evaluated once after retraining
on all development stations.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


# -------------------- Configuration --------------------
SEED, N_SPLITS = 42, 5
DATA = Path("Extracted_features_dataset.csv")
OUT = Path("results")
OUT.mkdir(parents=True, exist_ok=True)

STATION, TIME = "station", "datetime"
MUSICA, OBS, TARGET = "PM25", "PM2.5 (Hourly measured)", "Bias"
SPLIT_FILE, FOLD_FILE = OUT / "station_split.csv", OUT / "development_fold_split.csv"

# Optimal LightGBM hyperparameters from the completed search
PARAMS = {
    "objective": "regression_l1", 
    "metric": "mae", 
    "num_leaves": 400,
    "min_child_samples": 50, 
    "learning_rate": 0.1, 
    "n_estimators": 3000,
    "random_state": SEED, 
    "n_jobs": 12, 
    "verbose": -1,
}


def metrics(obs, raw, corrected):
    """Return before/after correction metrics."""
    def basic(pred):
        return (r2_score(obs, pred), mean_absolute_error(obs, pred),
                mean_squared_error(obs, pred, squared=False), np.mean(pred - obs))

    rb, mb, qb, bb = basic(raw)
    ra, ma, qa, ba = basic(corrected)
    pct = lambda drop, base: np.nan if np.isclose(base, 0) else 100 * drop / base
    return {
        "R2_Before": rb, "R2_After": ra, "R2_Increase": ra - rb,
        "MAE_Before": mb, "MAE_After": ma, "MAE_Drop": mb - ma,
        "MAE_Improvement_Pct": pct(mb - ma, mb),
        "RMSE_Before": qb, "RMSE_After": qa, "RMSE_Drop": qb - qa,
        "RMSE_Improvement_Pct": pct(qb - qa, qb),
        "BIAS_Before": bb, "BIAS_After": ba,
        "BIAS_Abs_Reduction": abs(bb) - abs(ba),
        "BIAS_Improvement_Pct": pct(abs(bb) - abs(ba), abs(bb)),
    }


def prediction_table(data, predicted_bias):
    """Build aligned prediction output using stable row_id."""
    out = data[["row_id", STATION, TIME, MUSICA, OBS, TARGET]].copy()
    out = out.rename(columns={OBS: "AURN_Observation", TARGET: "True_Bias"})
    out["Predicted_Bias"] = predicted_bias
    out["PM2.5_Corrected"] = out[MUSICA] - out["Predicted_Bias"]
    out["Residual_Before"] = out[MUSICA] - out["AURN_Observation"]
    out["Residual_After"] = out["PM2.5_Corrected"] - out["AURN_Observation"]
    out["Absolute_Error_Before"] = out["Residual_Before"].abs()
    out["Absolute_Error_After"] = out["Residual_After"].abs()
    return out


# -------------------- Load data and split 63/16 stations --------------------
df = pd.read_csv(DATA)
df[STATION] = df[STATION].astype(str)
df[TIME] = pd.to_datetime(df[TIME])
df["row_id"] = df[STATION] + "__" + df[TIME].dt.strftime("%Y-%m-%dT%H:%M:%S")
if df["row_id"].duplicated().any() or df[STATION].nunique() != 79:
    raise ValueError("Expected 79 stations and unique station-datetime rows.")
if not np.allclose(df[TARGET], df[MUSICA] - df[OBS]):
    raise ValueError("Bias must equal MUSICA PM25 minus AURN PM2.5.")

if SPLIT_FILE.exists():
    split = pd.read_csv(SPLIT_FILE, dtype={STATION: str})
else:
    stations = np.array(sorted(df[STATION].unique()))
    shuffled = np.random.RandomState(SEED).permutation(stations)
    split = pd.DataFrame({STATION: shuffled,
                          "dataset": ["development"] * 63 + ["test"] * 16})
    split.to_csv(SPLIT_FILE, index=False)

dev_stations = split.loc[split["dataset"] == "development", STATION]
test_stations = split.loc[split["dataset"] == "test", STATION]
if len(dev_stations) != 63 or len(test_stations) != 16:
    raise ValueError("station_split.csv must contain 63 development and 16 test stations.")

dev = df[df[STATION].isin(dev_stations)].copy()
test = df[df[STATION].isin(test_stations)].copy()
excluded = {"Unnamed: 0", "row_id", STATION, TIME, OBS, TARGET}
features = [c for c in df.columns if c not in excluded]
if len(features) != 34:
    raise ValueError(f"Expected 34 features, found {len(features)}: {features}")


# -------------------- Create/read common five-fold station split --------------------
if FOLD_FILE.exists():
    folds = pd.read_csv(FOLD_FILE, dtype={"row_id": str})
    dev = dev.merge(folds[["row_id", "fold_id"]], on="row_id", validate="one_to_one")
else:
    dev["fold_id"] = -1
    splitter = GroupKFold(n_splits=N_SPLITS)
    for fold, (_, val_idx) in enumerate(
            splitter.split(dev[features], dev[TARGET], groups=dev[STATION]), 1):
        dev.iloc[val_idx, dev.columns.get_loc("fold_id")] = fold
    folds = dev[["row_id", STATION, TIME, "fold_id"]]
    folds.to_csv(FOLD_FILE, index=False)

print(f"Development: {dev.shape[0]} rows, 63 stations, {len(features)} features")
print(f"Final test: {test.shape[0]} rows, 16 stations, {len(features)} features")
print(dev.groupby("fold_id").agg(rows=("row_id", "size"), stations=(STATION, "nunique")))


# -------------------- Start five-fold OOF training --------------------
oof_parts, fold_rows, histories = [], [], []
for fold in range(1, N_SPLITS + 1):
    train_fold, val_fold = dev[dev.fold_id != fold], dev[dev.fold_id == fold]
    evals_result = {}
    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(train_fold[features], train_fold[TARGET],
              eval_set=[(val_fold[features], val_fold[TARGET])],
              eval_metric="mae",
              callbacks=[lgb.record_evaluation(evals_result), lgb.log_evaluation(50)])
    pred = prediction_table(val_fold, model.predict(val_fold[features]))
    pred["Fold_ID"], pred["Model"] = fold, "LightGBM"
    oof_parts.append(pred)
    row = metrics(pred.AURN_Observation, pred[MUSICA], pred["PM2.5_Corrected"])
    row.update({"Fold_ID": fold, "Train_Rows": len(train_fold),
                "Validation_Rows": len(val_fold),
                "Train_Stations": train_fold[STATION].nunique(),
                "Validation_Stations": val_fold[STATION].nunique()})
    fold_rows.append(row)
    mae_history = next(iter(next(iter(evals_result.values())).values()))
    histories.append(pd.DataFrame({"Fold_ID": fold,
                                   "Boosting_Round": np.arange(1, len(mae_history) + 1),
                                   "Validation_MAE": mae_history}))

oof = pd.concat(oof_parts).sort_values("row_id")
fold_metrics = pd.DataFrame(fold_rows)
history = pd.concat(histories, ignore_index=True)
overall = pd.DataFrame([metrics(oof.AURN_Observation, oof[MUSICA], oof["PM2.5_Corrected"])])
station_metrics = pd.DataFrame([
    {STATION: station, **metrics(g.AURN_Observation, g[MUSICA], g["PM2.5_Corrected"])}
    for station, g in oof.groupby(STATION)
])


# -------------------- Train final model on all 63 development stations --------------------
final_model = lgb.LGBMRegressor(**PARAMS)
final_model.fit(dev[features], dev[TARGET])


# -------------------- Final evaluation on 16 untouched test stations --------------------
test_pred = prediction_table(test, final_model.predict(test[features])).sort_values("row_id")
test_pred["Model"] = "LightGBM"
test_metrics = pd.DataFrame([
    metrics(test_pred.AURN_Observation, test_pred[MUSICA], test_pred["PM2.5_Corrected"])
])
params_out = pd.DataFrame([{**PARAMS, "feature_count": len(features)}])


# -------------------- Save outputs --------------------
fold_metrics.to_csv(OUT / "LGBM_fold_metrics.csv", index=False)
oof.to_csv(OUT / "LGBM_OOF_predictions.csv", index=False)
overall.to_csv(OUT / "LGBM_OOF_overall_metrics.csv", index=False)
station_metrics.to_csv(OUT / "LGBM_OOF_station_metrics.csv", index=False)
history.to_csv(OUT / "LGBM_training_history.csv", index=False)
params_out.to_csv(OUT / "LGBM_best_parameters.csv", index=False)
test_pred.to_csv(OUT / "LGBM_final_test_predictions.csv", index=False)
test_metrics.to_csv(OUT / "LGBM_final_test_metrics.csv", index=False)
print(f"Finished. Results saved to: {OUT}")