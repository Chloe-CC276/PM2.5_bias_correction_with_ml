"""
Equal-weight bias-correction ensemble:
    DeepKriging / 3 + feature-axis CNN / 3 + LightGBM / 3

Protocol
--------
1. Keep the fixed 63-development/16-test station split.
2. Produce leakage-free OOF predictions with the same five station folds.
3. Retrain every component on all 63 development stations.
4. Evaluate once on the untouched test stations.

The nominal user weights (0.333, 0.333, 0.333) sum to 0.999. They are
normalised below, so each effective weight is exactly 1/3.
"""

import gc
import json
import os
import random
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, regularizers


# ============================== Configuration ==============================
SEED, N_SPLITS, N_JOBS = 42, 5, 12

DATA = Path("Extracted_features_dataset_withgeo.csv")
OUT = Path("Ensemble_DK_CNN_LGBM")
COMMON_OUT = DATA.parent.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

STATION, TIME = "station", "datetime"
LAT, LON = "grid_lat", "grid_lon"
MUSICA, OBS, TARGET = "PM25", "PM2.5 (Hourly measured)", "Bias"
SPLIT_FILE = Path("station_split.csv")
FOLD_FILE = Path("development_fold_split.csv")

# DeepKriging optimum
DK_UNITS, DK_LAYERS, DK_EPOCHS = 1024, 3, 100
DK_BATCH, DK_LR, DK_L2, DK_DROPOUT = 256, 5e-4, 1e-6, 0.2
DK_KNOT_COUNTS = (8, 16, 32)

# CNN optimum
CNN_FILTERS, CNN_KERNEL, CNN_LAYERS = 128, 6, 3
CNN_DENSE, CNN_EPOCHS = 32, 90
CNN_BATCH, CNN_LR, CNN_L2, CNN_DROPOUT = 256, 5e-4, 1e-6, 0.2

# LightGBM optimum. Fixed 3,000 trees reproduces the selected configuration;
# no test observations are used for early stopping or model selection.
LGBM_PARAMS = {
    "objective": "regression_l1",
    "metric": "mae",
    "num_leaves": 400,
    "min_child_samples": 50,
    "learning_rate": 0.1,
    "n_estimators": 3000,
    "random_state": SEED,
    "n_jobs": N_JOBS,
    "verbosity": -1,
}

NOMINAL_WEIGHTS = {
    "DeepKriging": 0.33,
    "CNN": 0.33,
    "LightGBM": 0.33,
}
_weight_sum = sum(NOMINAL_WEIGHTS.values())
WEIGHTS = {name: value / _weight_sum for name, value in NOMINAL_WEIGHTS.items()}


def set_seed(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def rmse(y_true, y_pred):
    # Compatible with both older and newer scikit-learn versions.
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def four_metrics(observation, prediction):
    observation = np.asarray(observation, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    return {
        "R2": float(r2_score(observation, prediction)),
        "MAE": float(mean_absolute_error(observation, prediction)),
        "RMSE": rmse(observation, prediction),
        "BIAS": float(np.mean(prediction - observation)),
    }


def aligned_metrics(observation, raw_pm25, corrected_pm25):
    """Four requested metrics plus aligned before/after improvements."""
    before = four_metrics(observation, raw_pm25)
    after = four_metrics(observation, corrected_pm25)

    def percent(drop, base):
        return np.nan if np.isclose(base, 0) else 100.0 * drop / abs(base)

    return {
        "R2": after["R2"],
        "MAE": after["MAE"],
        "RMSE": after["RMSE"],
        "BIAS": after["BIAS"],
        "R2_Before": before["R2"],
        "R2_After": after["R2"],
        "R2_Increase": after["R2"] - before["R2"],
        "MAE_Before": before["MAE"],
        "MAE_After": after["MAE"],
        "MAE_Drop": before["MAE"] - after["MAE"],
        "MAE_Improvement_Pct": percent(
            before["MAE"] - after["MAE"], before["MAE"]
        ),
        "RMSE_Before": before["RMSE"],
        "RMSE_After": after["RMSE"],
        "RMSE_Drop": before["RMSE"] - after["RMSE"],
        "RMSE_Improvement_Pct": percent(
            before["RMSE"] - after["RMSE"], before["RMSE"]
        ),
        "BIAS_Before": before["BIAS"],
        "BIAS_After": after["BIAS"],
        "BIAS_Abs_Reduction": abs(before["BIAS"]) - abs(after["BIAS"]),
        "BIAS_Improvement_Pct": percent(
            abs(before["BIAS"]) - abs(after["BIAS"]), before["BIAS"]
        ),
    }


def prediction_table(data, dk_bias, cnn_bias, lgbm_bias, fold_id=None):
    """Store component and ensemble predictions for every observation."""
    columns = ["row_id", STATION, TIME, LAT, LON, MUSICA, OBS, TARGET]
    out = data[columns].copy()
    out = out.rename(columns={OBS: "AURN_Observation", TARGET: "True_Bias"})

    out["DeepKriging_Predicted_Bias"] = np.asarray(dk_bias).ravel()
    out["CNN_Predicted_Bias"] = np.asarray(cnn_bias).ravel()
    out["LightGBM_Predicted_Bias"] = np.asarray(lgbm_bias).ravel()
    out["DeepKriging_PM2.5_Corrected"] = (
        out[MUSICA] - out["DeepKriging_Predicted_Bias"]
    )
    out["CNN_PM2.5_Corrected"] = out[MUSICA] - out["CNN_Predicted_Bias"]
    out["LightGBM_PM2.5_Corrected"] = (
        out[MUSICA] - out["LightGBM_Predicted_Bias"]
    )

    out["Ensemble_Predicted_Bias"] = (
        WEIGHTS["DeepKriging"] * out["DeepKriging_Predicted_Bias"]
        + WEIGHTS["CNN"] * out["CNN_Predicted_Bias"]
        + WEIGHTS["LightGBM"] * out["LightGBM_Predicted_Bias"]
    )
    out["PM2.5_Corrected"] = out[MUSICA] - out["Ensemble_Predicted_Bias"]
    out["Residual_Before"] = out[MUSICA] - out["AURN_Observation"]
    out["Residual_After"] = out["PM2.5_Corrected"] - out["AURN_Observation"]
    out["Absolute_Error_Before"] = out["Residual_Before"].abs()
    out["Absolute_Error_After"] = out["Residual_After"].abs()
    out["Weight_DeepKriging"] = WEIGHTS["DeepKriging"]
    out["Weight_CNN"] = WEIGHTS["CNN"]
    out["Weight_LightGBM"] = WEIGHTS["LightGBM"]
    out["Model"] = "DeepKriging_CNN_LightGBM_equal_weight"
    if fold_id is not None:
        out["Fold_ID"] = int(fold_id)
    return out


# ============================ DeepKriging model ============================
class SpatialBasis:
    """Training-fold-only, multi-resolution Gaussian radial basis functions."""

    def __init__(self, knot_counts=DK_KNOT_COUNTS, seed=SEED):
        self.knot_counts = knot_counts
        self.seed = seed
        self.coord_scaler = StandardScaler()
        self.centres = []
        self.bandwidths = []

    def fit(self, station_coordinates):
        coordinates = (
            station_coordinates[[LAT, LON]].drop_duplicates().to_numpy(dtype=float)
        )
        coordinates = self.coord_scaler.fit_transform(coordinates)
        self.centres, self.bandwidths = [], []
        for requested_k in self.knot_counts:
            k = min(requested_k, len(coordinates))
            km = KMeans(n_clusters=k, random_state=self.seed + k, n_init=10)
            centres = km.fit(coordinates).cluster_centers_
            if len(centres) > 1:
                distances = np.sqrt(
                    ((centres[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
                )
                distances[distances == 0] = np.nan
                bandwidth = 1.5 * np.nanmedian(np.nanmin(distances, axis=1))
            else:
                bandwidth = 1.0
            self.centres.append(centres)
            self.bandwidths.append(max(float(bandwidth), 1e-3))
        return self

    def transform(self, data):
        coordinates = self.coord_scaler.transform(
            data[[LAT, LON]].to_numpy(dtype=float)
        )
        parts = []
        for centres, bandwidth in zip(self.centres, self.bandwidths):
            distance2 = ((coordinates[:, None, :] - centres[None, :, :]) ** 2).sum(
                axis=2
            )
            parts.append(np.exp(-distance2 / (2.0 * bandwidth**2)))
        return np.concatenate(parts, axis=1).astype(np.float32)

    @property
    def n_basis(self):
        return int(sum(len(centres) for centres in self.centres))


def make_dk_inputs(train_data, other_data, features):
    station_coordinates = train_data.groupby(STATION, as_index=False)[[LAT, LON]].median()
    basis = SpatialBasis().fit(station_coordinates)
    train_array = np.column_stack(
        [
            train_data[features].to_numpy(dtype=np.float32),
            basis.transform(train_data),
        ]
    )
    other_array = np.column_stack(
        [
            other_data[features].to_numpy(dtype=np.float32),
            basis.transform(other_data),
        ]
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_array).astype(np.float32)
    x_other = scaler.transform(other_array).astype(np.float32)
    return x_train, x_other, scaler, basis


def build_deepkriging(n_inputs):
    inputs = keras.Input(shape=(n_inputs,), name="features_and_spatial_basis")
    x = inputs
    for layer_id in range(DK_LAYERS):
        units = max(DK_UNITS // (2**layer_id), 32)
        x = layers.Dense(
            units,
            use_bias=False,
            kernel_regularizer=regularizers.l2(DK_L2),
            name=f"dk_dense_{layer_id + 1}",
        )(x)
        x = layers.BatchNormalization(name=f"dk_bn_{layer_id + 1}")(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(DK_DROPOUT, name=f"dk_dropout_{layer_id + 1}")(x)
    outputs = layers.Dense(1, name="Predicted_Bias")(x)
    model = keras.Model(inputs, outputs, name="DeepKriging_bias_correction")
    model.compile(
        optimizer=keras.optimizers.Adam(DK_LR, clipnorm=1.0),
        loss=keras.losses.Huber(delta=5.0),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


# ================================ CNN model ================================
def make_cnn_inputs(train_data, other_data, features):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_data[features]).astype(np.float32)
    x_other = scaler.transform(other_data[features]).astype(np.float32)
    return x_train[..., None], x_other[..., None], scaler


def build_cnn(n_features):
    inputs = keras.Input(shape=(n_features, 1), name="ordered_feature_axis")
    x = inputs
    for layer_id in range(CNN_LAYERS):
        filters = min(CNN_FILTERS * (2**layer_id), 256)
        x = layers.Conv1D(
            filters=filters,
            kernel_size=CNN_KERNEL,
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(CNN_L2),
            name=f"cnn_conv_{layer_id + 1}",
        )(x)
        x = layers.Dropout(CNN_DROPOUT, name=f"cnn_dropout_{layer_id + 1}")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(
        CNN_DENSE,
        activation="relu",
        kernel_regularizer=regularizers.l2(CNN_L2),
        name="cnn_dense",
    )(x)
    x = layers.Dropout(CNN_DROPOUT)(x)
    outputs = layers.Dense(1, name="Predicted_Bias")(x)
    model = keras.Model(inputs, outputs, name="CNN_bias_correction")
    model.compile(
        optimizer=keras.optimizers.Adam(CNN_LR),
        loss="mae",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def history_frame(history, fold_id, component, final_training=False):
    n_epochs = len(history.history["loss"])
    result = pd.DataFrame(
        {
            "Training_Phase": "final_63_stations" if final_training else "OOF_CV",
            "Component": component,
            "Fold_ID": "Final" if final_training else fold_id,
            "Epoch": np.arange(1, n_epochs + 1),
            "Train_Loss": history.history["loss"],
        }
    )
    for source, destination in (
        ("mae", "Train_MAE"),
        ("val_loss", "Validation_Loss"),
        ("val_mae", "Validation_MAE"),
    ):
        if source in history.history:
            result[destination] = history.history[source]
    return result


def metric_row(predictions):
    return aligned_metrics(
        predictions["AURN_Observation"],
        predictions[MUSICA],
        predictions["PM2.5_Corrected"],
    )


# ================================ Load data ================================
set_seed()
df = pd.read_csv(DATA)
df[STATION] = df[STATION].astype(str)
df[TIME] = pd.to_datetime(df[TIME])
df["row_id"] = df[STATION] + "__" + df[TIME].dt.strftime("%Y-%m-%dT%H:%M:%S")

required = {STATION, TIME, LAT, LON, MUSICA, OBS, TARGET}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")
if df[STATION].nunique() != 79:
    raise ValueError(f"Expected 79 stations, found {df[STATION].nunique()}.")
if df["row_id"].duplicated().any():
    raise ValueError("row_id is not unique; station-datetime pairs must be unique.")
if not np.allclose(df[TARGET], df[MUSICA] - df[OBS], rtol=1e-5, atol=1e-6):
    raise ValueError("Bias must equal MUSICA PM25 minus AURN PM2.5.")

# Use the exact common station split, creating it only if it does not yet exist.
if SPLIT_FILE.exists():
    split = pd.read_csv(SPLIT_FILE, dtype={STATION: str})
else:
    shuffled = np.random.RandomState(SEED).permutation(
        np.asarray(sorted(df[STATION].unique()))
    )
    split = pd.DataFrame(
        {STATION: shuffled, "dataset": ["development"] * 63 + ["test"] * 16}
    )
    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(SPLIT_FILE, index=False)

if split[STATION].duplicated().any() or set(split[STATION]) != set(df[STATION].unique()):
    raise ValueError("station_split.csv does not match the 79 stations in DATA.")
dev_stations = split.loc[split["dataset"] == "development", STATION]
test_stations = split.loc[split["dataset"] == "test", STATION]
if len(dev_stations) != 63 or len(test_stations) != 16:
    raise ValueError("station_split.csv must contain 63 development and 16 test stations.")
if set(dev_stations) & set(test_stations):
    raise ValueError("Development/test station leakage detected.")

dev = df[df[STATION].isin(dev_stations)].copy()
test = df[df[STATION].isin(test_stations)].copy()

# LAT/LON are reserved for the DeepKriging basis. The 34 original predictors
# are identical for all three base learners.
excluded = {"Unnamed: 0", "row_id", STATION, TIME, LAT, LON, OBS, TARGET}
features = [column for column in df.columns if column not in excluded]
if len(features) != 34:
    raise ValueError(f"Expected 34 original features, found {len(features)}: {features}")

# Read/create a single shared fold assignment. GroupKFold keeps entire stations
# together, so rows from one station cannot occur in both train and validation.
if FOLD_FILE.exists():
    fold_map = pd.read_csv(FOLD_FILE, dtype={"row_id": str})
    if fold_map["row_id"].duplicated().any():
        raise ValueError("Duplicate row_id in development_fold_split.csv.")
    dev = dev.merge(
        fold_map[["row_id", "fold_id"]], on="row_id", how="left", validate="one_to_one"
    )
    if dev["fold_id"].isna().any():
        raise ValueError("The common fold file does not cover every development row.")
    dev["fold_id"] = dev["fold_id"].astype(int)
else:
    dev["fold_id"] = -1
    splitter = GroupKFold(n_splits=N_SPLITS)
    for fold_id, (_, validation_index) in enumerate(
        splitter.split(dev[features], dev[TARGET], groups=dev[STATION]), start=1
    ):
        dev.iloc[validation_index, dev.columns.get_loc("fold_id")] = fold_id
    dev[["row_id", STATION, TIME, "fold_id"]].to_csv(FOLD_FILE, index=False)

if set(dev["fold_id"].unique()) != set(range(1, N_SPLITS + 1)):
    raise ValueError("Fold IDs must be exactly 1, 2, 3, 4, 5.")

print(f">>> Development: {len(dev)} rows, {dev[STATION].nunique()} stations")
print(f">>> Test:        {len(test)} rows, {test[STATION].nunique()} stations")
print(f">>> Predictors:  {len(features)}")
print(f">>> Effective weights: {WEIGHTS}")
print(dev.groupby("fold_id").agg(rows=("row_id", "size"), stations=(STATION, "nunique")))


# ========================= Five-fold OOF training ==========================
oof_parts, fold_rows, history_parts = [], [], []

for fold_id in range(1, N_SPLITS + 1):
    train_fold = dev[dev["fold_id"] != fold_id].copy()
    val_fold = dev[dev["fold_id"] == fold_id].copy()
    overlap = set(train_fold[STATION]) & set(val_fold[STATION])
    if overlap:
        raise RuntimeError(f"Fold {fold_id} station leakage: {sorted(overlap)}")

    print(f"\n{'=' * 72}\n>>> OOF fold {fold_id}/{N_SPLITS}")
    y_train = train_fold[TARGET].to_numpy(dtype=np.float32)

    # ----- DeepKriging -----
    x_dk_train, x_dk_val, _, dk_basis = make_dk_inputs(
        train_fold, val_fold, features
    )
    keras.backend.clear_session()
    set_seed(SEED + fold_id)
    dk_model = build_deepkriging(x_dk_train.shape[1])
    dk_history = dk_model.fit(
        x_dk_train,
        y_train,
        validation_data=(x_dk_val, val_fold[TARGET].to_numpy(dtype=np.float32)),
        epochs=DK_EPOCHS,
        batch_size=DK_BATCH,
        shuffle=True,
        verbose=2,
    )
    dk_pred = dk_model.predict(x_dk_val, batch_size=DK_BATCH, verbose=0).ravel()
    history_parts.append(history_frame(dk_history, fold_id, "DeepKriging"))
    del dk_model, x_dk_train, x_dk_val
    keras.backend.clear_session()
    gc.collect()

    # ----- CNN -----
    x_cnn_train, x_cnn_val, _ = make_cnn_inputs(train_fold, val_fold, features)
    set_seed(SEED + 100 + fold_id)
    cnn_model = build_cnn(len(features))
    cnn_history = cnn_model.fit(
        x_cnn_train,
        y_train,
        validation_data=(x_cnn_val, val_fold[TARGET].to_numpy(dtype=np.float32)),
        epochs=CNN_EPOCHS,
        batch_size=CNN_BATCH,
        shuffle=True,
        verbose=2,
    )
    cnn_pred = cnn_model.predict(x_cnn_val, batch_size=CNN_BATCH, verbose=0).ravel()
    history_parts.append(history_frame(cnn_history, fold_id, "CNN"))
    del cnn_model, x_cnn_train, x_cnn_val
    keras.backend.clear_session()
    gc.collect()

    # ----- LightGBM -----
    lgbm_model = lgb.LGBMRegressor(**LGBM_PARAMS)
    lgbm_model.fit(
        train_fold[features],
        train_fold[TARGET],
        eval_set=[(val_fold[features], val_fold[TARGET])],
        callbacks=[lgb.log_evaluation(period=100)],
    )
    lgbm_pred = lgbm_model.predict(val_fold[features])
    evaluation = lgbm_model.evals_result_.get("valid_0", {}).get("l1", [])
    if evaluation:
        history_parts.append(
            pd.DataFrame(
                {
                    "Training_Phase": "OOF_CV",
                    "Component": "LightGBM",
                    "Fold_ID": fold_id,
                    "Epoch": np.arange(1, len(evaluation) + 1),
                    "Validation_MAE": evaluation,
                }
            )
        )

    fold_prediction = prediction_table(
        val_fold, dk_pred, cnn_pred, lgbm_pred, fold_id=fold_id
    )
    oof_parts.append(fold_prediction)
    row = metric_row(fold_prediction)
    row.update(
        {
            "Fold_ID": fold_id,
            "Train_Rows": len(train_fold),
            "Validation_Rows": len(val_fold),
            "Train_Stations": train_fold[STATION].nunique(),
            "Validation_Stations": val_fold[STATION].nunique(),
            "Original_Features": len(features),
            "DeepKriging_Spatial_Basis_Features": dk_basis.n_basis,
        }
    )
    fold_rows.append(row)
    print(
        f">>> Fold {fold_id} ensemble: R2={row['R2']:.4f}, "
        f"MAE={row['MAE']:.4f}, RMSE={row['RMSE']:.4f}, BIAS={row['BIAS']:.4f}"
    )
    del lgbm_model
    gc.collect()


oof = pd.concat(oof_parts, ignore_index=True).sort_values("row_id").reset_index(drop=True)
if len(oof) != len(dev) or oof["row_id"].duplicated().any():
    raise RuntimeError("Invalid OOF coverage: every development row must appear exactly once.")
if set(oof["row_id"]) != set(dev["row_id"]):
    raise RuntimeError("OOF row IDs do not match the development dataset.")

fold_metrics = pd.DataFrame(fold_rows)
oof_overall = pd.DataFrame([metric_row(oof)])
oof_station = pd.DataFrame(
    [
        {STATION: station, **metric_row(group)}
        for station, group in oof.groupby(STATION, sort=True)
    ]
)
training_history = pd.concat(history_parts, ignore_index=True, sort=False)


# ================== Retrain all components on 63 stations ==================
print(f"\n{'=' * 72}\n>>> Retraining all components on 63 development stations")
y_dev = dev[TARGET].to_numpy(dtype=np.float32)

# Final DeepKriging
x_dk_dev, x_dk_test, dk_scaler, final_basis = make_dk_inputs(dev, test, features)
keras.backend.clear_session()
set_seed(SEED)
final_dk = build_deepkriging(x_dk_dev.shape[1])
final_dk_history = final_dk.fit(
    x_dk_dev,
    y_dev,
    epochs=DK_EPOCHS,
    batch_size=DK_BATCH,
    shuffle=True,
    verbose=2,
)
test_dk_pred = final_dk.predict(x_dk_test, batch_size=DK_BATCH, verbose=0).ravel()
history_parts_final = [
    history_frame(final_dk_history, "Final", "DeepKriging", final_training=True)
]
final_dk.save(OUT / "Ensemble_final_DeepKriging.keras")
joblib.dump(dk_scaler, OUT / "Ensemble_final_DeepKriging_scaler.joblib")
joblib.dump(final_basis, OUT / "Ensemble_final_spatial_basis.joblib")
del x_dk_dev, x_dk_test
gc.collect()

# Final CNN
x_cnn_dev, x_cnn_test, cnn_scaler = make_cnn_inputs(dev, test, features)
keras.backend.clear_session()
set_seed(SEED + 100)
final_cnn = build_cnn(len(features))
final_cnn_history = final_cnn.fit(
    x_cnn_dev,
    y_dev,
    epochs=CNN_EPOCHS,
    batch_size=CNN_BATCH,
    shuffle=True,
    verbose=2,
)
test_cnn_pred = final_cnn.predict(x_cnn_test, batch_size=CNN_BATCH, verbose=0).ravel()
history_parts_final.append(
    history_frame(final_cnn_history, "Final", "CNN", final_training=True)
)
final_cnn.save(OUT / "Ensemble_final_CNN.keras")
joblib.dump(cnn_scaler, OUT / "Ensemble_final_CNN_scaler.joblib")
del x_cnn_dev, x_cnn_test
keras.backend.clear_session()
gc.collect()

# Final LightGBM
final_lgbm = lgb.LGBMRegressor(**LGBM_PARAMS)
final_lgbm.fit(dev[features], dev[TARGET])
test_lgbm_pred = final_lgbm.predict(test[features])
joblib.dump(final_lgbm, OUT / "Ensemble_final_LightGBM.joblib")


# ======================= Independent test evaluation =======================
test_prediction = prediction_table(
    test, test_dk_pred, test_cnn_pred, test_lgbm_pred
).sort_values("row_id").reset_index(drop=True)
test_metrics = pd.DataFrame([metric_row(test_prediction)])

final_history = pd.concat(history_parts_final, ignore_index=True, sort=False)
training_history = pd.concat(
    [training_history, final_history], ignore_index=True, sort=False
)

parameters = pd.DataFrame(
    [
        {
            "Ensemble": "DeepKriging_CNN_LightGBM_equal_weight",
            "Seed": SEED,
            "CV": "5-fold GroupKFold by station",
            "Development_Stations": dev[STATION].nunique(),
            "Test_Stations": test[STATION].nunique(),
            "Original_Features": len(features),
            "Weight_DeepKriging": WEIGHTS["DeepKriging"],
            "Weight_CNN": WEIGHTS["CNN"],
            "Weight_LightGBM": WEIGHTS["LightGBM"],
            "DK_units": DK_UNITS,
            "DK_n_layers": DK_LAYERS,
            "DK_epochs": DK_EPOCHS,
            "DK_batch_size": DK_BATCH,
            "DK_learning_rate": DK_LR,
            "DK_l2": DK_L2,
            "DK_dropout": DK_DROPOUT,
            "DK_knot_counts": "|".join(map(str, DK_KNOT_COUNTS)),
            "CNN_filters": CNN_FILTERS,
            "CNN_kernel_size": CNN_KERNEL,
            "CNN_n_conv_layers": CNN_LAYERS,
            "CNN_dense_units": CNN_DENSE,
            "CNN_epochs": CNN_EPOCHS,
            "CNN_batch_size": CNN_BATCH,
            "CNN_learning_rate": CNN_LR,
            "CNN_l2": CNN_L2,
            "CNN_dropout": CNN_DROPOUT,
            **{f"LGBM_{key}": value for key, value in LGBM_PARAMS.items()},
        }
    ]
)


# ================================ Save CSVs ================================
fold_metrics.to_csv(OUT / "Ensemble_fold_metrics.csv", index=False)
oof.to_csv(OUT / "Ensemble_OOF_predictions.csv", index=False)
oof_overall.to_csv(OUT / "Ensemble_OOF_overall_metrics.csv", index=False)
oof_station.to_csv(OUT / "Ensemble_OOF_station_metrics.csv", index=False)
training_history.to_csv(OUT / "Ensemble_training_history.csv", index=False)
parameters.to_csv(OUT / "Ensemble_best_parameters.csv", index=False)
test_prediction.to_csv(OUT / "Ensemble_final_test_predictions.csv", index=False)
test_metrics.to_csv(OUT / "Ensemble_final_test_metrics.csv", index=False)
with open(OUT / "Ensemble_feature_columns.json", "w", encoding="utf-8") as file:
    json.dump(features, file, ensure_ascii=False, indent=2)

print("\n>>> OOF overall metrics (model-selection evidence):")
print(oof_overall[["R2", "MAE", "RMSE", "BIAS"]].to_string(index=False))
print("\n>>> Independent-test metrics:")
print(test_metrics[["R2", "MAE", "RMSE", "BIAS"]].to_string(index=False))
print(f"\n>>> All outputs saved to: {OUT}")
