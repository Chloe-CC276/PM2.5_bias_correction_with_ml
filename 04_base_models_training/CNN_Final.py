"""
Final 1-D CNN bias correction with station-grouped five-fold OOF.

Convolution is applied along the feature axis. Training follows the shared
63/16 station split and fold assignment used by the other base learners.
"""

import gc
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


# -------------------- Configuration --------------------
SEED, N_SPLITS = 42, 5
DATA = Path("Extracted_features_dataset.csv")
RESULTS_ROOT = DATA.parent / "results"
SPLIT_FILE = Path("station_split.csv")
FOLD_FILE = Path("development_fold_split.csv")

STATION, TIME = "station", "datetime"
MUSICA, OBS, TARGET = "PM25", "PM2.5 (Hourly measured)", "Bias"

CONFIGS = [
    {
        "Config": "CNN_F96_K8",
        "filters": 96,
        "kernel_size": 8,
        "n_conv_layers": 3,
        "dense_units": 32,
        "dropout": 0.2,
        "batch_size": 256,
        "learning_rate": 5e-4,
        "l2": 1e-6,
        "best_epoch": 74,
        "selection_metric": "validation_MAE",
    },
    {
        "Config": "CNN_F128_K6",
        "filters": 128,
        "kernel_size": 6,
        "n_conv_layers": 3,
        "dense_units": 32,
        "dropout": 0.2,
        "batch_size": 256,
        "learning_rate": 5e-4,
        "l2": 1e-6,
        "best_epoch": 74,
        "selection_metric": "validation_MAE",
    },
]


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


def build_model(n_features, params):
    """Reproduce the exact feature-axis CNN used during tuning."""
    model = Sequential([Input(shape=(n_features, 1), name="feature_axis")])
    for layer_id in range(params["n_conv_layers"]):
        layer_filters = min(int(params["filters"] * (2**layer_id)), 256)
        model.add(
            Conv1D(
                filters=layer_filters,
                kernel_size=params["kernel_size"],
                padding="same",
                activation="relu",
                kernel_regularizer=l2(params["l2"]),
                name=f"conv1d_{layer_id + 1}",
            )
        )
        model.add(Dropout(params["dropout"], name=f"dropout_{layer_id + 1}"))

    model.add(GlobalAveragePooling1D(name="global_average_pooling"))
    model.add(
        Dense(
            params["dense_units"],
            activation="relu",
            kernel_regularizer=l2(params["l2"]),
            name="dense",
        )
    )
    model.add(Dropout(params["dropout"], name="dense_dropout"))
    model.add(Dense(1, name="Predicted_Bias"))
    model.compile(
        optimizer=Adam(learning_rate=params["learning_rate"]),
        loss="mae",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
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


def prediction_table(data, predicted_bias, config, fold_id=None):
    """Create XGBoost-aligned row-level predictions."""
    out = data[["row_id", STATION, TIME, MUSICA, OBS, TARGET]].copy()
    out = out.rename(columns={OBS: "AURN_Observation", TARGET: "True_Bias"})
    out["Predicted_Bias"] = np.asarray(predicted_bias).ravel()
    out["PM2.5_Corrected"] = out[MUSICA] - out["Predicted_Bias"]
    out["Residual_Before"] = out[MUSICA] - out["AURN_Observation"]
    out["Residual_After"] = out["PM2.5_Corrected"] - out["AURN_Observation"]
    out["Absolute_Error_Before"] = out["Residual_Before"].abs()
    out["Absolute_Error_After"] = out["Residual_After"].abs()
    out["Model"], out["Config"] = "Feature-axis 1D-CNN", config
    if fold_id is not None:
        out["Fold_ID"] = fold_id
    return out


def load_aligned_data():
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
    excluded = {"Unnamed: 0", "row_id", STATION, TIME, OBS, TARGET}
    features = [column for column in df.columns if column not in excluded]
    if len(features) != 34:
        raise ValueError(f"Expected 34 features, found {len(features)}: {features}")

    if FOLD_FILE.exists():
        folds = pd.read_csv(FOLD_FILE, dtype={"row_id": str})
        if set(dev["row_id"]) != set(folds["row_id"]):
            raise ValueError("CNN rows do not exactly match the XGBoost fold file.")
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


def run_configuration(dev, test, features, params):
    config = params["Config"]
    out = RESULTS_ROOT / config
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 72}\n>>> Running {config}: {params}\n{'=' * 72}")

    # -------------------- Five-fold station-level OOF training --------------------
    oof_parts, fold_rows, histories = [], [], []
    for fold in range(1, N_SPLITS + 1):
        train_fold = dev[dev.fold_id != fold]
        val_fold = dev[dev.fold_id == fold]

        scaler = StandardScaler()
        x_train = scaler.fit_transform(train_fold[features])
        x_val = scaler.transform(val_fold[features])
        x_train = x_train.reshape(-1, len(features), 1).astype(np.float32)
        x_val = x_val.reshape(-1, len(features), 1).astype(np.float32)
        y_train = train_fold[TARGET].to_numpy(dtype=np.float32)
        y_val = val_fold[TARGET].to_numpy(dtype=np.float32)

        tf.keras.backend.clear_session()
        set_seed(SEED + fold)
        model = build_model(len(features), params)
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=params["best_epoch"],
            batch_size=params["batch_size"],
            verbose=2,
            shuffle=True,
        )

        predicted_bias = model.predict(
            x_val, batch_size=params["batch_size"], verbose=0
        ).ravel()
        pred = prediction_table(val_fold, predicted_bias, config, fold)
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
                "Best_Epoch": params["best_epoch"],
                "Config": config,
            }
        )
        fold_rows.append(row)
        histories.append(
            pd.DataFrame(
                {
                    "Fold_ID": fold,
                    "Epoch": np.arange(1, len(history.history["loss"]) + 1),
                    "Train_MAE": history.history["loss"],
                    "Validation_MAE": history.history["val_loss"],
                    "Config": config,
                }
            )
        )
        print(
            f">>> Fold {fold}: R2={row['R2_After']:.4f}, "
            f"MAE={row['MAE_After']:.4f}, RMSE={row['RMSE_After']:.4f}, "
            f"BIAS={row['BIAS_After']:.4f}"
        )
        del model, scaler, x_train, x_val, y_train, y_val
        gc.collect()

    # -------------------- Overall and station OOF metrics --------------------
    oof = pd.concat(oof_parts, ignore_index=True).sort_values("row_id")
    if len(oof) != len(dev) or oof["row_id"].duplicated().any():
        raise ValueError("OOF predictions must cover each development row exactly once.")

    fold_metrics = pd.DataFrame(fold_rows)
    overall = pd.DataFrame(
        [metrics(oof["AURN_Observation"], oof[MUSICA], oof["PM2.5_Corrected"])]
    )
    overall.insert(0, "Config", config)
    station_metrics = pd.DataFrame(
        [
            {
                STATION: station,
                "Config": config,
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
    final_scaler = StandardScaler()
    x_dev = final_scaler.fit_transform(dev[features])
    x_test = final_scaler.transform(test[features])
    x_dev = x_dev.reshape(-1, len(features), 1).astype(np.float32)
    x_test = x_test.reshape(-1, len(features), 1).astype(np.float32)

    tf.keras.backend.clear_session()
    set_seed(SEED)
    final_model = build_model(len(features), params)
    final_model.fit(
        x_dev,
        dev[TARGET].to_numpy(dtype=np.float32),
        epochs=params["best_epoch"],
        batch_size=params["batch_size"],
        verbose=2,
        shuffle=True,
    )

    # -------------------- Preview only: 16-station test evaluation --------------------
    test_bias = final_model.predict(
        x_test, batch_size=params["batch_size"], verbose=0
    ).ravel()
    test_pred = prediction_table(test, test_bias, config).sort_values("row_id")
    test_metrics = pd.DataFrame(
        [
            metrics(
                test_pred["AURN_Observation"],
                test_pred[MUSICA],
                test_pred["PM2.5_Corrected"],
            )
        ]
    )
    test_metrics.insert(0, "Config", config)
    params_out = pd.DataFrame(
        [
            {
                **params,
                "n_features": len(features),
                "development_stations": 63,
                "test_stations": 16,
                "model_selection_basis": "63_station_5fold_OOF_MAE",
                "test_usage": "preview_only_not_for_model_selection",
            }
        ]
    )

    # -------------------- Save eight XGBoost-aligned CSV outputs --------------------
    pd.DataFrame(fold_rows).to_csv(out / "CNN_fold_metrics.csv", index=False)
    oof.to_csv(out / "CNN_OOF_predictions.csv", index=False)
    overall.to_csv(out / "CNN_OOF_overall_metrics.csv", index=False)
    station_metrics.to_csv(out / "CNN_OOF_station_metrics.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(
        out / "CNN_training_history.csv", index=False
    )
    params_out.to_csv(out / "CNN_best_parameters.csv", index=False)
    test_pred.to_csv(out / "CNN_final_test_predictions.csv", index=False)
    test_metrics.to_csv(out / "CNN_final_test_metrics.csv", index=False)

    print("\n>>> OOF overall metrics (use for model selection):")
    print(overall.to_string(index=False))
    print("\n>>> Test metrics (preview only; do not use for model selection):")
    print(test_metrics.to_string(index=False))
    print(f">>> Results saved to: {out}")

    del final_model, final_scaler, x_dev, x_test
    gc.collect()


def main():
    configure_gpu()
    set_seed()
    dev, test, features = load_aligned_data()
    print(f">>> Development: {len(dev)} rows, 63 stations, {len(features)} features")
    print(f">>> Final test: {len(test)} rows, 16 stations")
    print(dev.groupby("fold_id").agg(rows=("row_id", "size"), stations=(STATION, "nunique")))
    print(
        ">>> IMPORTANT: select between configurations using OOF MAE only; "
        "test metrics are preview outputs."
    )

    for params in CONFIGS:
        run_configuration(dev, test, features, params)

    print("\n>>> Both CNN configurations completed.")


if __name__ == "__main__":
    main()