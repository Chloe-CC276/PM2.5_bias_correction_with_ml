"""
Two-stage hyperparameter search for a 1-D CNN bias-correction model.

Stage 1 is a broad random search; stage 2 is a local grid over filters and
kernel_size. Search uses an 80,000-row subsample from development stations
and a station-grouped validation split.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Conv1D, Dense, Dropout,
                                     GlobalAveragePooling1D, Input)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


# -------------------- Configuration --------------------
SEED, SAMPLE_SIZE = 42, 80000
N_RANDOM_TRIALS, MAX_EPOCHS, PATIENCE = 24, 100, 12
DATA = Path("Extracted_features_dataset.csv")
OUT = Path("CNN_tuning")
FIG = Path("Figure")
SPLIT_FILE = OUT.parent / "station_split.csv"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

STATION, TIME = "station", "datetime"
MUSICA, OBS, TARGET = "PM25", "PM2.5 (Hourly measured)", "Bias"
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


def build_model(n_features, filters, kernel_size, n_conv_layers, dense_units,
                dropout, learning_rate, l2_value):
    """Build a 1D CNN that extracts local patterns across 34 input features."""
    model = Sequential([Input(shape=(n_features, 1))])
    for layer in range(n_conv_layers):
        layer_filters = min(int(filters * (2 ** layer)), 256)
        model.add(Conv1D(
            filters=layer_filters, kernel_size=kernel_size, padding="same",
            activation="relu", kernel_regularizer=l2(l2_value)
        ))
        model.add(Dropout(dropout))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(dense_units, activation="relu", kernel_regularizer=l2(l2_value)))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mae")
    return model


def metrics(obs, musica, predicted_bias):
    corrected = musica - predicted_bias
    return {
        "R2": r2_score(obs, corrected),
        "MAE": mean_absolute_error(obs, corrected),
        "RMSE": np.sqrt(mean_squared_error(obs, corrected)),
        "BIAS": np.mean(corrected - obs),
    }


def before_after_metrics(obs, musica, corrected):
    """Return before/after metrics, absolute changes and percentage changes."""
    def basic(pred):
        return {
            "R2": r2_score(obs, pred),
            "MAE": mean_absolute_error(obs, pred),
            "RMSE": np.sqrt(mean_squared_error(obs, pred)),
            "BIAS": np.mean(pred - obs),
        }

    before, after = basic(musica), basic(corrected)
    safe_pct = lambda change, base: (
        np.nan if np.isclose(base, 0) else 100.0 * change / abs(base)
    )
    return {
        "R2_Before": before["R2"], "R2_After": after["R2"],
        "R2_Increase": after["R2"] - before["R2"],
        "R2_Improvement_Pct": safe_pct(after["R2"] - before["R2"], before["R2"]),
        "MAE_Before": before["MAE"], "MAE_After": after["MAE"],
        "MAE_Drop": before["MAE"] - after["MAE"],
        "MAE_Improvement_Pct": safe_pct(before["MAE"] - after["MAE"], before["MAE"]),
        "RMSE_Before": before["RMSE"], "RMSE_After": after["RMSE"],
        "RMSE_Drop": before["RMSE"] - after["RMSE"],
        "RMSE_Improvement_Pct": safe_pct(before["RMSE"] - after["RMSE"], before["RMSE"]),
        "BIAS_Before": before["BIAS"], "BIAS_After": after["BIAS"],
        "BIAS_Abs_Reduction": abs(before["BIAS"]) - abs(after["BIAS"]),
        "BIAS_Improvement_Pct": safe_pct(
            abs(before["BIAS"]) - abs(after["BIAS"]), abs(before["BIAS"])
        ),
    }


def train_and_score(params, X_train, y_train, X_val, val_current):
    tf.keras.backend.clear_session()
    model = build_model(
        X_train.shape[1], filters=params["filters"],
        kernel_size=params["kernel_size"],
        n_conv_layers=params["n_conv_layers"],
        dense_units=params["dense_units"], dropout=params["dropout"],
        learning_rate=params["learning_rate"], l2_value=params["l2_value"]
    )
    early_stop = EarlyStopping(
        monitor="val_loss", patience=PATIENCE, min_delta=1e-4,
        restore_best_weights=True, verbose=0
    )
    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, val_current[TARGET].to_numpy()),
        epochs=MAX_EPOCHS, batch_size=params["batch_size"],
        callbacks=[early_stop], verbose=0, shuffle=True
    )
    pred = model.predict(X_val, batch_size=params["batch_size"], verbose=0).ravel()
    obs = val_current[OBS].to_numpy()
    musica = val_current[MUSICA].to_numpy()
    corrected = musica - pred
    result = metrics(obs, musica, pred)
    result.update(before_after_metrics(obs, musica, corrected))
    result.update({
        "Best_Epoch": int(np.argmin(history.history["val_loss"]) + 1),
        "Epochs_Run": len(history.history["loss"]),
        "Time_Seconds": time.time() - start,
    })
    return result


def plot_results(results):
    metric_names = ["R2", "MAE", "RMSE", "BIAS"]
    cmaps = ["YlGnBu", "YlOrRd", "YlOrRd", "RdBu_r"]
    fig, axes = plt.subplots(1, 4, figsize=(25, 6), dpi=200)
    for ax, name, cmap_name in zip(axes, metric_names, cmaps):
        table = results.pivot(index="kernel_size", columns="filters", values=name)
        sns.heatmap(table, annot=True, fmt=".4f", cmap=cmap_name,
                    linewidths=0, ax=ax, cbar_kws={"label": name})
        ax.set_title(f"CNN Hyperparameter Tuning ({name})", fontweight="bold")
        ax.set_xlabel("Filters")
        ax.set_ylabel("Kernel size")
    plt.tight_layout()
    plt.savefig(FIG / "CNN_filters_kernel_size_heatmap.png", dpi=300)
    plt.close()

    filters_values = np.sort(results["filters"].unique())
    kernel_values = np.sort(results["kernel_size"].unique())
    x, y = np.meshgrid(filters_values, kernel_values)
    fig = plt.figure(figsize=(25, 7), dpi=200)
    for i, name in enumerate(metric_names, 1):
        z = (results.pivot(index="kernel_size", columns="filters", values=name)
             .reindex(index=kernel_values, columns=filters_values).to_numpy())
        ax = fig.add_subplot(1, 4, i, projection="3d")
        surf = ax.plot_surface(x, y, z, cmap=cm.viridis, edgecolor="black",
                               linewidth=0.35, antialiased=True)
        ax.set_title(f"CNN Response ({name})", fontweight="bold")
        ax.set_xlabel("Filters")
        ax.set_ylabel("Kernel size")
        ax.set_zlabel(name)
        ax.view_init(elev=24, azim=-58)
        fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.1)
    plt.tight_layout()
    plt.savefig(FIG / "CNN_filters_kernel_size_3D.png", dpi=300)
    plt.close()


# -------------------- Data and leakage-free station split --------------------
df = pd.read_csv(DATA)
df[STATION] = df[STATION].astype(str)
df[TIME] = pd.to_datetime(df[TIME])
if df[STATION].nunique() != 79:
    raise ValueError(f"Expected 79 stations, found {df[STATION].nunique()}.")
if not np.allclose(df[TARGET], df[MUSICA] - df[OBS], equal_nan=False):
    raise ValueError("Bias must equal MUSICA PM25 minus AURN PM2.5.")

if SPLIT_FILE.exists():
    split = pd.read_csv(SPLIT_FILE, dtype={STATION: str})
else:
    stations = np.random.RandomState(SEED).permutation(
        np.array(sorted(df[STATION].unique()))
    )
    split = pd.DataFrame({STATION: stations,
                          "dataset": ["development"] * 63 + ["test"] * 16})
    split.to_csv(SPLIT_FILE, index=False)

dev_stations = split.loc[split["dataset"] == "development", STATION]
test_stations = split.loc[split["dataset"] == "test", STATION]
if len(dev_stations) != 63 or len(test_stations) != 16:
    raise ValueError("station_split.csv must contain 63 development and 16 test stations.")
dev = df[df[STATION].isin(dev_stations)].copy()

excluded = {"Unnamed: 0", STATION, TIME, OBS, TARGET}
features = [column for column in df.columns if column not in excluded]
if len(features) != 34:
    raise ValueError(f"Expected 34 features, found {len(features)}: {features}")

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, val_idx = next(splitter.split(dev, groups=dev[STATION]))
tune_train, tune_val = dev.iloc[train_idx].copy(), dev.iloc[val_idx].copy()

# Match the ML tuning dataset: sample 80,000 independent training records.
if len(tune_train) < SAMPLE_SIZE:
    raise ValueError(f"Only {len(tune_train)} tuning rows; need {SAMPLE_SIZE}.")
sample = tune_train.sample(n=SAMPLE_SIZE, random_state=SEED).copy()
val_current = tune_val.reset_index(drop=True).copy()
scaler = StandardScaler()
X_train = scaler.fit_transform(
    sample[features]
).reshape(-1, len(features), 1).astype(np.float32)
X_val = scaler.transform(
    val_current[features]
).reshape(-1, len(features), 1).astype(np.float32)
y_train = sample[TARGET].to_numpy(dtype=np.float32)

print(f"Development: {len(dev)} rows, {dev[STATION].nunique()} stations")
print(f"Tuning train sample: {X_train.shape}")
print(f"Tuning validation records: {X_val.shape}")
print("The 16 final-test stations are not used for tuning.")


# -------------------- Stage 1: broad random search --------------------
space = {
    "filters": [16, 32, 64, 96, 128, 192],
    "kernel_size": [2, 3, 4, 5, 6, 8],
    "n_conv_layers": [1, 2, 3],
    "dense_units": [32, 64, 128, 256],
    "dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "batch_size": [128, 256, 512, 1024],
    "learning_rate": [1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3],
    "l2_value": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
}
rng = np.random.default_rng(SEED)
seen, random_rows = set(), []
while len(random_rows) < N_RANDOM_TRIALS:
    params = {name: rng.choice(values).item() for name, values in space.items()}
    key = tuple(params.items())
    if key in seen:
        continue
    seen.add(key)
    trial = len(random_rows) + 1
    print(f"\n[Random {trial:02d}/{N_RANDOM_TRIALS}] {params}")
    scores = train_and_score(params, X_train, y_train, X_val, val_current)
    random_rows.append({"Trial": trial, **params, **scores})
    print(f"R2={scores['R2']:.4f} | MAE={scores['MAE']:.4f} | "
          f"RMSE={scores['RMSE']:.4f} | BIAS={scores['BIAS']:.4f}")
    pd.DataFrame(random_rows).to_csv(OUT / "CNN_random_search_results.csv", index=False)

random_results = pd.DataFrame(random_rows).sort_values("MAE").reset_index(drop=True)
best = random_results.iloc[0]


# -------------------- Stage 2: fine filters x kernel_size grid --------------------
fixed = {
    "n_conv_layers": int(best["n_conv_layers"]),
    "dense_units": int(best["dense_units"]),
    "dropout": float(best["dropout"]),
    "batch_size": int(best["batch_size"]),
    "learning_rate": float(best["learning_rate"]),
    "l2_value": float(best["l2_value"]),
}
filters_values = [16, 32, 64, 96, 128, 192]
kernel_values = [2, 3, 4, 5, 6, 8]
fine_rows = []
for filters in filters_values:
    for kernel_size in kernel_values:
        params = {**fixed, "filters": filters, "kernel_size": kernel_size}
        print(f"\n[Fine grid] filters={filters} | kernel_size={kernel_size}")
        scores = train_and_score(params, X_train, y_train, X_val, val_current)
        fine_rows.append({**params, **scores})
        print(f"R2={scores['R2']:.4f} | MAE={scores['MAE']:.4f} | "
              f"RMSE={scores['RMSE']:.4f} | BIAS={scores['BIAS']:.4f}")
        pd.DataFrame(fine_rows).to_csv(OUT / "CNN_fine_grid_results.csv", index=False)

fine_results = pd.DataFrame(fine_rows).sort_values("MAE").reset_index(drop=True)
fine_results.to_csv(
    OUT / "CNN_best_validation_before_after_metrics.csv", index=False
)
best_fine = fine_results.iloc[0]
best_params = {
    "filters": int(best_fine["filters"]),
    "kernel_size": int(best_fine["kernel_size"]),
    "n_conv_layers": int(best_fine["n_conv_layers"]),
    "dense_units": int(best_fine["dense_units"]),
    "dropout": float(best_fine["dropout"]),
    "batch_size": int(best_fine["batch_size"]),
    "learning_rate": float(best_fine["learning_rate"]),
    "l2": float(best_fine["l2_value"]),
    "best_epoch": int(best_fine["Best_Epoch"]),
    "selection_metric": "validation_MAE",
}
with open(OUT / "CNN_best_parameters.json", "w", encoding="utf-8") as file:
    json.dump(best_params, file, indent=2)

# -------------------- Retrain best CNN and save validation predictions --------------------
tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(SEED)
final_model = build_model(
    len(features), filters=best_params["filters"],
    kernel_size=best_params["kernel_size"],
    n_conv_layers=best_params["n_conv_layers"],
    dense_units=best_params["dense_units"], dropout=best_params["dropout"],
    learning_rate=best_params["learning_rate"], l2_value=best_params["l2"]
)
final_model.fit(
    X_train, y_train, epochs=best_params["best_epoch"],
    batch_size=best_params["batch_size"], verbose=1, shuffle=True
)
predicted_bias = final_model.predict(
    X_val, batch_size=best_params["batch_size"], verbose=0
).ravel()

prediction_output = val_current[[STATION, TIME, MUSICA, OBS, TARGET]].copy()
prediction_output = prediction_output.rename(columns={
    MUSICA: "PM25_Model_Simulation",
    OBS: "PM25_Observation",
    TARGET: "True_Bias",
})
prediction_output["Predicted_Bias"] = predicted_bias
prediction_output["PM25_Corrected"] = (
    prediction_output["PM25_Model_Simulation"] - prediction_output["Predicted_Bias"]
)
prediction_output["Residual_Before"] = (
    prediction_output["PM25_Model_Simulation"] - prediction_output["PM25_Observation"]
)
prediction_output["Residual_After"] = (
    prediction_output["PM25_Corrected"] - prediction_output["PM25_Observation"]
)
prediction_output["Absolute_Error_Before"] = prediction_output["Residual_Before"].abs()
prediction_output["Absolute_Error_After"] = prediction_output["Residual_After"].abs()
prediction_output.to_csv(OUT / "CNN_best_validation_predictions.csv", index=False)

best_summary = before_after_metrics(
    prediction_output["PM25_Observation"].to_numpy(),
    prediction_output["PM25_Model_Simulation"].to_numpy(),
    prediction_output["PM25_Corrected"].to_numpy(),
)
pd.DataFrame([{**best_params, **best_summary}]).to_csv(
    OUT / "CNN_best_validation_summary.csv", index=False
)

plot_results(pd.DataFrame(fine_rows))
print("\nBest CNN parameters:")
print(json.dumps(best_params, indent=2))
print(f"Results saved to: {OUT}")
print(f"Figures saved to: {FIG}")