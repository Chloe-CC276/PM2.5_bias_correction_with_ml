"""
Final DeepKriging bias correction with station-grouped five-fold OOF.

Coordinates are encoded with multi-resolution spatial basis functions and
passed to an MLP. Basis functions are fitted on training stations only.
"""

import gc
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, regularizers


# -------------------- Configuration --------------------
SEED, N_SPLITS = 42, 5
DATA = Path("Extracted_features_dataset_withgeo.csv")
RESULTS_ROOT = DATA.parent / "results"
OUT = RESULTS_ROOT / "DeepKriging_U1024_L3"
OUT.mkdir(parents=True, exist_ok=True)

SPLIT_FILE = Path("station_split.csv")
FOLD_FILE = Path("development_fold_split.csv")

STATION, TIME = "station", "datetime"
LAT, LON = "grid_lat", "grid_lon"
MUSICA, OBS, TARGET = "PM25", "PM2.5 (Hourly measured)", "Bias"

PARAMS = {
    "Config": "DeepKriging_U1024_L3",
    "units": 1024,
    "n_layers": 3,
    "activation": "relu",
    "dropout": 0.2,
    "batch_size": 256,
    "learning_rate": 5e-4,
    "l2": 1e-6,
    "fold_best_epochs": "56|87|84|111|56",
    "median_best_epoch": 84,
    "selection_metric": "validation_MAE",
}
FOLD_EPOCHS = {1: 56, 2: 87, 3: 84, 4: 111, 5: 56}
KNOT_COUNTS = (8, 16, 32)


def set_seed(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def configure_gpu():
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


class SpatialBasis:
    """Multi-resolution Gaussian spatial basis fitted on training stations."""

    def __init__(self, knot_counts=KNOT_COUNTS, seed=SEED):
        self.knot_counts = knot_counts
        self.seed = seed
        self.coord_scaler = StandardScaler()
        self.centres = []
        self.bandwidths = []

    def fit(self, station_coordinates):
        coords = station_coordinates[[LAT, LON]].drop_duplicates().to_numpy(float)
        coords = self.coord_scaler.fit_transform(coords)
        self.centres, self.bandwidths = [], []

        for requested_k in self.knot_counts:
            k = min(requested_k, len(coords))
            centres = KMeans(
                n_clusters=k, random_state=self.seed + k, n_init=10
            ).fit(coords).cluster_centers_

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
        coords = self.coord_scaler.transform(data[[LAT, LON]].to_numpy(float))
        basis_parts = []
        for centres, bandwidth in zip(self.centres, self.bandwidths):
            squared_distance = (
                (coords[:, None, :] - centres[None, :, :]) ** 2
            ).sum(axis=2)
            basis_parts.append(np.exp(-squared_distance / (2.0 * bandwidth**2)))
        return np.concatenate(basis_parts, axis=1).astype(np.float32)

    @property
    def n_basis(self):
        return int(sum(len(centres) for centres in self.centres))


def make_inputs(train_data, other_data, feature_columns):
    """Fit spatial basis and scaler using training stations only."""
    station_coords = train_data.groupby(STATION, as_index=False)[[LAT, LON]].median()
    basis = SpatialBasis().fit(station_coords)
    train_basis = basis.transform(train_data)
    other_basis = basis.transform(other_data)
    train_base = train_data[feature_columns].to_numpy(dtype=np.float32)
    other_base = other_data[feature_columns].to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(np.column_stack([train_base, train_basis]))
    x_other = scaler.transform(np.column_stack([other_base, other_basis]))
    return x_train.astype(np.float32), x_other.astype(np.float32), basis


def build_model(n_inputs):
    inputs = keras.Input(shape=(n_inputs,), name="features_and_spatial_basis")
    x = inputs
    for layer_id in range(PARAMS["n_layers"]):
        layer_units = max(PARAMS["units"] // (2**layer_id), 32)
        x = layers.Dense(
            layer_units,
            use_bias=False,
            kernel_regularizer=regularizers.l2(PARAMS["l2"]),
            name=f"dense_{layer_id + 1}",
        )(x)
        x = layers.BatchNormalization(name=f"batch_norm_{layer_id + 1}")(x)
        x = layers.Activation(PARAMS["activation"])(x)
        x = layers.Dropout(PARAMS["dropout"], name=f"dropout_{layer_id + 1}")(x)

    output = layers.Dense(1, name="Predicted_Bias")(x)
    model = keras.Model(inputs, output, name="DeepKriging_bias_correction")
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=PARAMS["learning_rate"], clipnorm=1.0
        ),
        loss=keras.losses.Huber(delta=5.0),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def metrics(obs, raw, corrected):
    """Use the same metric definitions and field names as XGBoost."""

    def basic(pred):
        return (
            r2_score(obs, pred),
            mean_absolute_error(obs, pred),
            mean_squared_error(obs, pred, squared=False),
            np.mean(pred - obs),
        )

    rb, mb, qb, bb = basic(raw)
    ra, ma, qa, ba = basic(corrected)
    pct = lambda drop, base: np.nan if np.isclose(base, 0) else 100 * drop / base
    return {
        "R2_Before": rb,
        "R2_After": ra,
        "R2_Increase": ra - rb,
        "MAE_Before": mb,
        "MAE_After": ma,
        "MAE_Drop": mb - ma,
        "MAE_Improvement_Pct": pct(mb - ma, mb),
        "RMSE_Before": qb,
        "RMSE_After": qa,
        "RMSE_Drop": qb - qa,
        "RMSE_Improvement_Pct": pct(qb - qa, qb),
        "BIAS_Before": bb,
        "BIAS_After": ba,
        "BIAS_Abs_Reduction": abs(bb) - abs(ba),
        "BIAS_Improvement_Pct": pct(abs(bb) - abs(ba), abs(bb)),
    }


def prediction_table(data, predicted_bias, fold_id=None):
    out = data[["row_id", STATION, TIME, LAT, LON, MUSICA, OBS, TARGET]].copy()
    out = out.rename(columns={OBS: "AURN_Observation", TARGET: "True_Bias"})
    out["Predicted_Bias"] = np.asarray(predicted_bias).ravel()
    out["PM2.5_Corrected"] = out[MUSICA] - out["Predicted_Bias"]
    out["Residual_Before"] = out[MUSICA] - out["AURN_Observation"]
    out["Residual_After"] = out["PM2.5_Corrected"] - out["AURN_Observation"]
    out["Absolute_Error_Before"] = out["Residual_Before"].abs()
    out["Absolute_Error_After"] = out["Residual_After"].abs()
    out["Model"], out["Config"] = "DeepKriging", PARAMS["Config"]
    if fold_id is not None:
        out["Fold_ID"] = fold_id
    return out


def load_aligned_data():
    df = pd.read_csv(DATA)
    df[STATION] = df[STATION].astype(str)
    df[TIME] = pd.to_datetime(df[TIME])
    df["row_id"] = df[STATION] + "__" + df[TIME].dt.strftime("%Y-%m-%dT%H:%M:%S")

    required = {STATION, TIME, LAT, LON, MUSICA, OBS, TARGET}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(df.columns))}")
    if df["row_id"].duplicated().any() or df[STATION].nunique() != 79:
        raise ValueError("Expected 79 stations and unique station-datetime rows.")
    if not np.allclose(df[TARGET], df[MUSICA] - df[OBS]):
        raise ValueError("Bias must equal MUSICA PM25 minus AURN PM2.5.")

    if SPLIT_FILE.exists():
        split = pd.read_csv(SPLIT_FILE, dtype={STATION: str})
    else:
        stations = np.random.RandomState(SEED).permutation(
            np.array(sorted(df[STATION].unique()))
        )
        split = pd.DataFrame(
            {STATION: stations, "dataset": ["development"] * 63 + ["test"] * 16}
        )
        split.to_csv(SPLIT_FILE, index=False)

    dev_stations = split.loc[split["dataset"] == "development", STATION]
    test_stations = split.loc[split["dataset"] == "test", STATION]
    if len(dev_stations) != 63 or len(test_stations) != 16:
        raise ValueError("station_split.csv must contain 63 development and 16 test stations.")

    dev = df[df[STATION].isin(dev_stations)].copy()
    test = df[df[STATION].isin(test_stations)].copy()

    # Coordinates are represented by spatial basis functions, not duplicated as covariates.
    excluded = {"Unnamed: 0", "row_id", STATION, TIME, LAT, LON, OBS, TARGET}
    features = [column for column in df.columns if column not in excluded]
    if len(features) != 34:
        raise ValueError(
            f"Expected 34 non-spatial features after excluding coordinates: {features}"
        )

    if FOLD_FILE.exists():
        folds = pd.read_csv(FOLD_FILE, dtype={"row_id": str})
        if set(dev["row_id"]) != set(folds["row_id"]):
            raise ValueError("DeepKriging rows do not exactly match the XGBoost folds.")
        dev = dev.merge(
            folds[["row_id", "fold_id"]], on="row_id", validate="one_to_one"
        )
    else:
        dev["fold_id"] = -1
        splitter = GroupKFold(n_splits=N_SPLITS)
        for fold, (_, val_idx) in enumerate(
            splitter.split(dev[features], dev[TARGET], groups=dev[STATION]), 1
        ):
            dev.iloc[val_idx, dev.columns.get_loc("fold_id")] = fold
        dev[["row_id", STATION, TIME, "fold_id"]].to_csv(FOLD_FILE, index=False)

    if set(dev["fold_id"].astype(int)) != set(range(1, N_SPLITS + 1)):
        raise ValueError("Development fold IDs must be 1-5.")
    return dev, test, features


def main():
    configure_gpu()
    set_seed()
    dev, test, features = load_aligned_data()
    print(f">>> Parameters: {PARAMS}")
    print(f">>> Development: {len(dev)} rows, 63 stations, 34 base features")
    print(f">>> Final test: {len(test)} rows, 16 stations")
    print(dev.groupby("fold_id").agg(rows=("row_id", "size"), stations=(STATION, "nunique")))

    # -------------------- Five-fold station-level OOF training --------------------
    oof_parts, fold_rows, histories = [], [], []
    for fold in range(1, N_SPLITS + 1):
        train_fold = dev[dev.fold_id != fold]
        val_fold = dev[dev.fold_id == fold]
        x_train, x_val, basis = make_inputs(train_fold, val_fold, features)
        y_train = train_fold[TARGET].to_numpy(dtype=np.float32)
        y_val = val_fold[TARGET].to_numpy(dtype=np.float32)
        fold_epoch = FOLD_EPOCHS[fold]

        keras.backend.clear_session()
        set_seed(SEED + fold)
        model = build_model(x_train.shape[1])
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=fold_epoch,
            batch_size=PARAMS["batch_size"],
            verbose=2,
            shuffle=True,
        )

        predicted_bias = model.predict(
            x_val, batch_size=PARAMS["batch_size"], verbose=0
        ).ravel()
        pred = prediction_table(val_fold, predicted_bias, fold)
        oof_parts.append(pred)

        row = metrics(
            pred["AURN_Observation"], pred[MUSICA], pred["PM2.5_Corrected"]
        )
        row.update(
            {
                "Fold_ID": fold,
                "Train_Rows": len(train_fold),
                "Validation_Rows": len(val_fold),
                "Train_Stations": train_fold[STATION].nunique(),
                "Validation_Stations": val_fold[STATION].nunique(),
                "Best_Epoch": fold_epoch,
                "Original_Features": len(features),
                "Spatial_Basis_Features": basis.n_basis,
                "Config": PARAMS["Config"],
            }
        )
        fold_rows.append(row)
        histories.append(
            pd.DataFrame(
                {
                    "Fold_ID": fold,
                    "Epoch": np.arange(1, len(history.history["loss"]) + 1),
                    "Train_MAE": history.history["mae"],
                    "Validation_MAE": history.history["val_mae"],
                    "Train_Loss": history.history["loss"],
                    "Validation_Loss": history.history["val_loss"],
                    "Config": PARAMS["Config"],
                }
            )
        )
        print(
            f">>> Fold {fold}: epoch={fold_epoch}, R2={row['R2_After']:.4f}, "
            f"MAE={row['MAE_After']:.4f}, RMSE={row['RMSE_After']:.4f}, "
            f"BIAS={row['BIAS_After']:.4f}"
        )
        del model, x_train, x_val, y_train, y_val
        gc.collect()

    # -------------------- Overall and station OOF metrics --------------------
    oof = pd.concat(oof_parts, ignore_index=True).sort_values("row_id")
    if len(oof) != len(dev) or oof["row_id"].duplicated().any():
        raise ValueError("OOF predictions must cover each development row exactly once.")

    overall = pd.DataFrame(
        [metrics(oof["AURN_Observation"], oof[MUSICA], oof["PM2.5_Corrected"])]
    )
    overall.insert(0, "Config", PARAMS["Config"])
    station_metrics = pd.DataFrame(
        [
            {
                STATION: station,
                "Config": PARAMS["Config"],
                **metrics(
                    group["AURN_Observation"],
                    group[MUSICA],
                    group["PM2.5_Corrected"],
                ),
            }
            for station, group in oof.groupby(STATION)
        ]
    )

    # -------------------- Final training on all 63 development stations --------------------
    x_dev, x_test, final_basis = make_inputs(dev, test, features)
    keras.backend.clear_session()
    set_seed(SEED)
    final_model = build_model(x_dev.shape[1])
    final_model.fit(
        x_dev,
        dev[TARGET].to_numpy(dtype=np.float32),
        epochs=PARAMS["median_best_epoch"],
        batch_size=PARAMS["batch_size"],
        verbose=2,
        shuffle=True,
    )

    # -------------------- Preview only: 16-station test evaluation --------------------
    test_bias = final_model.predict(
        x_test, batch_size=PARAMS["batch_size"], verbose=0
    ).ravel()
    test_pred = prediction_table(test, test_bias).sort_values("row_id")
    test_metrics = pd.DataFrame(
        [
            metrics(
                test_pred["AURN_Observation"],
                test_pred[MUSICA],
                test_pred["PM2.5_Corrected"],
            )
        ]
    )
    test_metrics.insert(0, "Config", PARAMS["Config"])
    params_out = pd.DataFrame(
        [
            {
                **PARAMS,
                "original_features": len(features),
                "spatial_basis_features": final_basis.n_basis,
                "total_input_features": len(features) + final_basis.n_basis,
                "development_stations": 63,
                "test_stations": 16,
                "model_selection_basis": "63_station_5fold_OOF_MAE",
                "test_usage": "preview_only_not_for_model_selection",
            }
        ]
    )

    # -------------------- Save eight XGBoost-aligned CSV outputs --------------------
    pd.DataFrame(fold_rows).to_csv(OUT / "DeepKriging_fold_metrics.csv", index=False)
    oof.to_csv(OUT / "DeepKriging_OOF_predictions.csv", index=False)
    overall.to_csv(OUT / "DeepKriging_OOF_overall_metrics.csv", index=False)
    station_metrics.to_csv(OUT / "DeepKriging_OOF_station_metrics.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(
        OUT / "DeepKriging_training_history.csv", index=False
    )
    params_out.to_csv(OUT / "DeepKriging_best_parameters.csv", index=False)
    test_pred.to_csv(OUT / "DeepKriging_final_test_predictions.csv", index=False)
    test_metrics.to_csv(OUT / "DeepKriging_final_test_metrics.csv", index=False)

    print("\n>>> OOF overall metrics (use for model selection):")
    print(overall.to_string(index=False))
    print("\n>>> Test metrics (preview only; do not use for model selection):")
    print(test_metrics.to_string(index=False))
    print(f">>> Finished. Results saved to: {OUT}")


if __name__ == "__main__":
    main()