"""
Dynamic Mixture-of-Experts (MoE) gating network for PM2.5 bias correction

Experts (already trained; frozen here):
    1. DeepKriging
    2. CNN
    3. LightGBM

The gating network learns sample-specific weights:
    [w_DK(x), w_CNN(x), w_LGBM(x)] = softmax(GatingNetwork(context, expert_predictions))

and produces:
    Predicted_Bias =
        w_DK * DK_Predicted_Bias
      + w_CNN * CNN_Predicted_Bias
      + w_LGBM * LGBM_Predicted_Bias

Final PM2.5 correction:
    PM2.5_Corrected = MUSICA_PM25 - Predicted_Bias

IMPORTANT UNIT FIX
------------------
Historical expert prediction files were produced when MUSICA PM25 was still
stored in kg/m^3 (approximately 1e-9 to 1e-8), while AURN observations were in
ug/m^3. This corrected version converts PM25 to ug/m^3 using x1e9 and shifts
both True_Bias and each expert Predicted_Bias by the exact row-wise unit offset.

For every row:
    PM25_true = PM25_raw * 1e9
    unit_offset = PM25_true - PM25_raw
    True_Bias_true = True_Bias_old + unit_offset
    Predicted_Bias_true = Predicted_Bias_old + unit_offset

This preserves the concentration correction originally implied by each expert:
    PM25_true - Predicted_Bias_true
    == PM25_raw - Predicted_Bias_old

Validation protocol
-------------------
A. The three experts are NOT retrained here.
B. Development input uses their leakage-free 63-station OOF predictions.
C. The gating network itself is evaluated with an OUTER 5-fold GroupKFold by station.
D. Within each outer training fold, a station-grouped inner validation split is used
   only for early stopping; the outer validation stations remain untouched.
E. After meta-level OOF evaluation, a final gate is trained on all 63 development
   stations using an epoch count selected without using the 16 final-test stations.
F. The 16 test stations are evaluated exactly once at the end.

Important
---------
The OOF prediction files must correspond to the SAME development rows and same
63/16 station split. Alignment is checked strictly using row_id.
"""

# ================================ Imports ==================================
import os
import random
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "12")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, regularizers


# ============================== Configuration ==============================
SEED = 42
N_OUTER = 5
PM25_UNIT_FACTOR = 1e9

ROOT = Path(
    "/mnt/iusers01/msc-stu/hum-msc-data-sci-2025-2026/"
    "g05844jc/scratch/ERP_project"
)
RESULTS = ROOT / "results"

# Separate output folder for dynamic MoE so it does not overwrite static stacking.
OUT = RESULTS / "MoE_Gating_DK_CNN_LGBM_unit_fixed"
OUT.mkdir(parents=True, exist_ok=True)

# Existing single-model prediction files.
DK_OOF = RESULTS / "single_model_results" / "DeepKriging_OOF_predictions.csv"
DK_TEST = RESULTS / "single_model_results" / "DeepKriging_final_test_predictions.csv"

CNN_OOF = RESULTS / "single_model_results" / "CNN_OOF_predictions.csv"
CNN_TEST = RESULTS / "single_model_results" / "CNN_final_test_predictions.csv"

LGBM_OOF = RESULTS / "single_model_results" / "LGBM_OOF_predictions.csv"
LGBM_TEST = RESULTS / "single_model_results" / "LGBM_final_test_predictions.csv"

# Original feature table is used only to attach contextual variables to each row.
# The target is NOT reconstructed from this file for model training; True_Bias still
# comes from the aligned expert prediction tables.
DATA = ROOT / "data" / "Extracted_features_dataset_withgeo.csv"

STATION = "station"
TIME = "datetime"
LAT = "grid_lat"
LON = "grid_lon"
MUSICA = "PM25"
OBS = "AURN_Observation"
TRUE_BIAS = "True_Bias"

EXPERT_COLS = [
    "DeepKriging_Predicted_Bias",
    "CNN_Predicted_Bias",
    "LightGBM_Predicted_Bias",
]

# Context variables used by the gate.
# Only variables present in DATA will be retained; required variables are checked.
#
# We deliberately use a compact physically interpretable set instead of all 34
# predictors so that the gate learns "which expert to trust under which condition"
# rather than becoming an unrestricted fourth bias-prediction model.
CONTEXT_CANDIDATES = [
    "PM25",          # MUSICA PM2.5 level
    "PBLH",          # boundary-layer condition
    "CO",
    "NOX",
    "BURDENBCdn",
    "WS",
    "U",
    "V",
    "Hour",
    "Month",
    LAT,
    LON,
]

# Minimum context required for the requested dynamic spatial/temporal gate.
REQUIRED_CONTEXT = ["PM25", "PBLH", "CO", "NOX", "Hour", "Month", LAT, LON]

# Gating network hyperparameters.
GATE_HIDDEN_1 = 64
GATE_HIDDEN_2 = 32
GATE_DROPOUT = 0.15
GATE_L2 = 1e-5
LEARNING_RATE = 5e-4
BATCH_SIZE = 512
MAX_EPOCHS = 150
PATIENCE = 15
MIN_DELTA = 1e-4

# If True, expert predictions are also included as gate inputs.
# This is recommended: the gate can use both environmental context and
# disagreement among experts to decide weights.
USE_EXPERT_PREDS_AS_GATE_INPUT = True


# ============================== Reproducibility =============================
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


# ================================ Metrics ==================================
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics(obs, raw_pm25, corrected_pm25):
    """Aligned before/after PM2.5 metrics."""
    obs = np.asarray(obs, dtype=float)
    raw_pm25 = np.asarray(raw_pm25, dtype=float)
    corrected_pm25 = np.asarray(corrected_pm25, dtype=float)

    def basic(pred):
        return {
            "R2": float(r2_score(obs, pred)),
            "MAE": float(mean_absolute_error(obs, pred)),
            "RMSE": rmse(obs, pred),
            "BIAS": float(np.mean(pred - obs)),
        }

    before = basic(raw_pm25)
    after = basic(corrected_pm25)

    def pct(drop, base):
        return np.nan if np.isclose(base, 0) else 100.0 * drop / abs(base)

    return {
        "R2_Before": before["R2"],
        "R2_After": after["R2"],
        "R2_Increase": after["R2"] - before["R2"],

        "MAE_Before": before["MAE"],
        "MAE_After": after["MAE"],
        "MAE_Drop": before["MAE"] - after["MAE"],
        "MAE_Improvement_Pct": pct(
            before["MAE"] - after["MAE"], before["MAE"]
        ),

        "RMSE_Before": before["RMSE"],
        "RMSE_After": after["RMSE"],
        "RMSE_Drop": before["RMSE"] - after["RMSE"],
        "RMSE_Improvement_Pct": pct(
            before["RMSE"] - after["RMSE"], before["RMSE"]
        ),

        "BIAS_Before": before["BIAS"],
        "BIAS_After": after["BIAS"],
        "BIAS_Abs_Reduction": abs(before["BIAS"]) - abs(after["BIAS"]),
        "BIAS_Improvement_Pct": pct(
            abs(before["BIAS"]) - abs(after["BIAS"]), abs(before["BIAS"])
        ),
    }


# =========================== Prediction-file loading =======================
def validate_and_fix_prediction_units(df, model_name):
    """
    Convert one historical expert prediction table to the correct physical scale.

    Historical table:
        PM25           : kg/m^3
        AURN           : ug/m^3
        True_Bias      : PM25_raw - AURN
        Predicted_Bias : expert prediction of historical wrong-unit target

    Corrected table:
        PM25_correct = PM25_raw * 1e9
        unit_offset = PM25_correct - PM25_raw
        True_Bias_correct = PM25_correct - AURN
        Predicted_Bias_correct = Predicted_Bias_old + unit_offset
    """
    df = df.copy()

    raw_pm25 = df[MUSICA].to_numpy(dtype=float)
    obs = df[OBS].to_numpy(dtype=float)
    old_true_bias = df[TRUE_BIAS].to_numpy(dtype=float)
    old_pred_bias = df["Predicted_Bias"].to_numpy(dtype=float)

    # Confirm this really is one of the historical wrong-unit files.
    expected_old_bias = raw_pm25 - obs
    if not np.allclose(
        old_true_bias,
        expected_old_bias,
        rtol=1e-5,
        atol=1e-5,
    ):
        max_diff = float(np.max(np.abs(old_true_bias - expected_old_bias)))
        raise ValueError(
            f"{model_name}: historical True_Bias is not PM25_raw - AURN. "
            f"Maximum difference={max_diff:.6g}. "
            "Do not apply this automatic unit fix until the input file is checked."
        )

    corrected_pm25 = raw_pm25 * PM25_UNIT_FACTOR
    unit_offset = corrected_pm25 - raw_pm25
    corrected_true_bias = corrected_pm25 - obs
    corrected_pred_bias = old_pred_bias + unit_offset

    # Keep audit columns.
    df["PM25_Raw_Before_Unit_Fix"] = raw_pm25
    df["Historical_True_Bias"] = old_true_bias
    df["Historical_Predicted_Bias"] = old_pred_bias
    df["PM25_Unit_Offset"] = unit_offset

    df[MUSICA] = corrected_pm25
    df[TRUE_BIAS] = corrected_true_bias
    df["Predicted_Bias"] = corrected_pred_bias

    # Algebraic consistency check: corrected PM2.5 concentration is preserved.
    old_corrected = raw_pm25 - old_pred_bias
    new_corrected = corrected_pm25 - corrected_pred_bias
    if not np.allclose(
        old_corrected,
        new_corrected,
        rtol=1e-6,
        atol=1e-6,
    ):
        raise RuntimeError(
            f"{model_name}: unit conversion failed corrected-concentration check."
        )

    return df


def read_prediction(path, model_name):
    """Read one expert prediction file, apply unit correction, and standardise columns."""
    if not path.exists():
        raise FileNotFoundError(f"{model_name} file not found: {path}")

    df = pd.read_csv(path, dtype={"row_id": str, STATION: str})
    if TIME in df.columns:
        df[TIME] = pd.to_datetime(df[TIME])

    required = {
        "row_id", STATION, MUSICA, OBS, TRUE_BIAS, "Predicted_Bias"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{model_name} missing required columns: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    if df["row_id"].duplicated().any():
        raise ValueError(f"{model_name}: duplicated row_id detected.")

    df = validate_and_fix_prediction_units(df, model_name)

    keep = ["row_id", STATION]
    if TIME in df.columns:
        keep.append(TIME)
    keep += [
        MUSICA,
        OBS,
        TRUE_BIAS,
        "PM25_Raw_Before_Unit_Fix",
        "Historical_True_Bias",
        "Historical_Predicted_Bias",
        "PM25_Unit_Offset",
        "Predicted_Bias",
    ]

    df = df[keep].copy()
    df = df.rename(
        columns={"Predicted_Bias": f"{model_name}_Predicted_Bias"}
    )
    return df

def merge_three_predictions(dk_path, cnn_path, lgbm_path):
    """Strictly align DeepKriging/CNN/LightGBM predictions by row_id."""
    dk = read_prediction(dk_path, "DeepKriging")
    cnn = read_prediction(cnn_path, "CNN")
    lgbm = read_prediction(lgbm_path, "LightGBM")

    # Validate the common reference values before discarding duplicates.
    for name, other in [("CNN", cnn), ("LightGBM", lgbm)]:
        check = dk[["row_id", STATION, MUSICA, OBS, TRUE_BIAS]].merge(
            other[["row_id", STATION, MUSICA, OBS, TRUE_BIAS]],
            on="row_id",
            suffixes=("_dk", f"_{name.lower()}"),
            how="inner",
            validate="one_to_one",
        )
        if len(check) != len(dk) or len(check) != len(other):
            raise ValueError(
                f"DeepKriging and {name} do not have exactly the same row_id set."
            )

        for col in [MUSICA, OBS, TRUE_BIAS]:
            a = check[f"{col}_dk"].to_numpy(float)
            b = check[f"{col}_{name.lower()}"].to_numpy(float)
            if not np.allclose(a, b, rtol=1e-6, atol=1e-6):
                raise ValueError(
                    f"Reference column {col} differs between DeepKriging and {name}."
                )

    merged = dk.copy()
    merged = merged.merge(
        cnn[["row_id", "CNN_Predicted_Bias"]],
        on="row_id", how="inner", validate="one_to_one"
    )
    merged = merged.merge(
        lgbm[["row_id", "LightGBM_Predicted_Bias"]],
        on="row_id", how="inner", validate="one_to_one"
    )

    if len(merged) != len(dk):
        raise ValueError("Merged prediction table lost rows.")
    if merged[EXPERT_COLS].isna().any().any():
        raise ValueError("Missing expert predictions after merge.")

    return merged.sort_values("row_id").reset_index(drop=True)


# ============================ Context attachment ============================
def load_context():
    """Load contextual environmental/time/space variables keyed by row_id."""
    if not DATA.exists():
        raise FileNotFoundError(f"Context data file not found: {DATA}")

    data = pd.read_csv(DATA)
    data[STATION] = data[STATION].astype(str)
    data[TIME] = pd.to_datetime(data[TIME])

    # Context PM25 must be on the same physical unit scale as the corrected
    # expert prediction tables before overlap validation and gate training.
    if MUSICA not in data.columns:
        raise ValueError(f"Context dataset is missing required column: {MUSICA}")
    data[MUSICA] = data[MUSICA].astype(float) * PM25_UNIT_FACTOR

    if "row_id" not in data.columns:
        data["row_id"] = (
            data[STATION]
            + "__"
            + data[TIME].dt.strftime("%Y-%m-%dT%H:%M:%S")
        )

    if data["row_id"].duplicated().any():
        raise ValueError("Context dataset contains duplicated row_id values.")

    missing_required = [c for c in REQUIRED_CONTEXT if c not in data.columns]
    if missing_required:
        raise ValueError(
            f"Context dataset is missing required gating variables: {missing_required}"
        )

    context_cols = [c for c in CONTEXT_CANDIDATES if c in data.columns]
    print(f">>> Context variables used by gate ({len(context_cols)}): {context_cols}")

    context = data[["row_id"] + context_cols].copy()

    # All gate inputs must be finite.
    bad = context[context_cols].isna().sum()
    bad = bad[bad > 0]
    if len(bad):
        raise ValueError(
            "Context variables contain missing values:\n"
            + bad.to_string()
        )

    return context, context_cols


def attach_context(predictions, context):
    """
    Attach contextual variables without duplicating columns already present
    in the prediction files.

    PM25 is already retained from each expert prediction table. If PM25 is
    merged again from DATA with pandas' default behaviour, it becomes
    PM25_x / PM25_y, so the later gating code can no longer find "PM25".
    This function keeps the prediction-table version of overlapping columns
    and merges only genuinely new context columns.
    """
    predictions = predictions.copy()
    context = context.copy()

    # Context columns that are already present in predictions, e.g. PM25.
    overlap = [
        c for c in context.columns
        if c != "row_id" and c in predictions.columns
    ]

    # Validate overlapping numeric values before keeping the prediction copy.
    if overlap:
        check = predictions[["row_id"] + overlap].merge(
            context[["row_id"] + overlap],
            on="row_id",
            how="left",
            validate="one_to_one",
            suffixes=("_pred", "_ctx"),
        )

        for col in overlap:
            left = check[f"{col}_pred"]
            right = check[f"{col}_ctx"]

            if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
                if not np.allclose(
                    left.to_numpy(dtype=float),
                    right.to_numpy(dtype=float),
                    rtol=1e-6,
                    atol=1e-6,
                    equal_nan=True,
                ):
                    raise ValueError(
                        f"Overlapping context column '{col}' differs between "
                        "prediction files and DATA."
                    )
            else:
                if not left.fillna("__NA__").equals(right.fillna("__NA__")):
                    raise ValueError(
                        f"Overlapping context column '{col}' differs between "
                        "prediction files and DATA."
                    )

    # Merge only context variables not already carried by prediction files.
    new_context_cols = [
        c for c in context.columns
        if c == "row_id" or c not in predictions.columns
    ]

    out = predictions.merge(
        context[new_context_cols],
        on="row_id",
        how="left",
        validate="one_to_one",
    )

    if out.isna().any().any():
        missing_rows = int(out.isna().any(axis=1).sum())
        raise ValueError(
            f"{missing_rows} prediction rows have missing values after context merge."
        )

    return out


# =============================== MoE model =================================
def build_gate(n_gate_features):
    """
    Build a gating network.

    Input 1: standardised context/disagreement features.
    Input 2: raw three expert predictions.

    The network outputs:
        1. sample-specific softmax weights
        2. weighted bias prediction

    Training optimises only the final weighted bias.
    """
    gate_input = keras.Input(
        shape=(n_gate_features,),
        name="gate_features"
    )
    expert_input = keras.Input(
        shape=(3,),
        name="expert_predictions"
    )

    x = layers.Dense(
        GATE_HIDDEN_1,
        activation="relu",
        kernel_regularizer=regularizers.l2(GATE_L2),
        name="gate_dense_1",
    )(gate_input)
    x = layers.BatchNormalization(name="gate_bn_1")(x)
    x = layers.Dropout(GATE_DROPOUT, name="gate_dropout_1")(x)

    x = layers.Dense(
        GATE_HIDDEN_2,
        activation="relu",
        kernel_regularizer=regularizers.l2(GATE_L2),
        name="gate_dense_2",
    )(x)
    x = layers.Dropout(GATE_DROPOUT, name="gate_dropout_2")(x)

    weights = layers.Dense(
        3,
        activation="softmax",
        name="expert_weights",
    )(x)

    # Weighted sum of the three frozen expert predictions.
    weighted_bias = layers.Dot(
        axes=1,
        name="MoE_Predicted_Bias"
    )([weights, expert_input])

    train_model = keras.Model(
        inputs=[gate_input, expert_input],
        outputs=weighted_bias,
        name="PM25_Dynamic_MoE",
    )

    # Separate model for exporting interpretable per-sample weights.
    weight_model = keras.Model(
        inputs=[gate_input, expert_input],
        outputs=weights,
        name="PM25_Dynamic_MoE_Weights",
    )

    train_model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
            clipnorm=1.0,
        ),
        # Huber is robust to the strongly skewed bias distribution while
        # remaining smooth for optimisation.
        loss=keras.losses.Huber(delta=5.0),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return train_model, weight_model


def make_gate_matrix(data, context_cols, scaler=None, fit_scaler=False):
    """
    Construct gate inputs.

    Context is combined with expert disagreement information.
    The raw expert predictions are ALSO passed separately to the weighted-sum layer.
    """
    gate_cols = list(context_cols)

    # Expert outputs themselves can inform trust allocation.
    if USE_EXPERT_PREDS_AS_GATE_INPUT:
        gate_cols += EXPERT_COLS

    x = data[gate_cols].to_numpy(dtype=np.float32)

    if fit_scaler:
        scaler = StandardScaler()
        x = scaler.fit_transform(x).astype(np.float32)
    else:
        x = scaler.transform(x).astype(np.float32)

    experts = data[EXPERT_COLS].to_numpy(dtype=np.float32)
    return x, experts, scaler, gate_cols


def predict_with_weights(model, weight_model, x_gate, x_experts, batch_size=BATCH_SIZE):
    pred_bias = model.predict(
        [x_gate, x_experts],
        batch_size=batch_size,
        verbose=0,
    ).ravel()

    weights = weight_model.predict(
        [x_gate, x_experts],
        batch_size=batch_size,
        verbose=0,
    )

    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError("Softmax weights do not sum to 1.")

    return pred_bias, weights


def output_table(data, pred_bias, weights, phase, fold_id=None):
    out = data.copy()

    out["Weight_DeepKriging"] = weights[:, 0]
    out["Weight_CNN"] = weights[:, 1]
    out["Weight_LightGBM"] = weights[:, 2]

    out["MoE_Predicted_Bias"] = np.asarray(pred_bias).ravel()
    out["PM2.5_Corrected"] = out[MUSICA] - out["MoE_Predicted_Bias"]

    out["Residual_Before"] = out[MUSICA] - out[OBS]
    out["Residual_After"] = out["PM2.5_Corrected"] - out[OBS]
    out["Absolute_Error_Before"] = out["Residual_Before"].abs()
    out["Absolute_Error_After"] = out["Residual_After"].abs()

    out["MoE_Phase"] = phase
    if fold_id is not None:
        out["Meta_Fold_ID"] = int(fold_id)

    return out


# ======================== Weight-analysis utilities =========================
def weight_summary(data, group_col=None):
    weight_cols = [
        "Weight_DeepKriging",
        "Weight_CNN",
        "Weight_LightGBM",
    ]

    if group_col is None:
        rows = []
        for col in weight_cols:
            rows.append({
                "Expert": col.replace("Weight_", ""),
                "Mean_Weight": data[col].mean(),
                "Median_Weight": data[col].median(),
                "Std_Weight": data[col].std(),
                "P10": data[col].quantile(0.10),
                "P90": data[col].quantile(0.90),
                "Win_Rate": (data[col] == data[weight_cols].max(axis=1)).mean(),
            })
        return pd.DataFrame(rows)

    return (
        data.groupby(group_col, observed=True)[weight_cols]
        .agg(["mean", "median", "std"])
        .reset_index()
    )


# ============================== Main pipeline ===============================
def main():
    configure_gpu()
    set_seed()

    # ---------- Load already-computed expert predictions ----------
    oof = merge_three_predictions(DK_OOF, CNN_OOF, LGBM_OOF)
    test = merge_three_predictions(DK_TEST, CNN_TEST, LGBM_TEST)

    if oof[STATION].nunique() != 63:
        raise ValueError(
            f"Expected 63 development stations, found {oof[STATION].nunique()}."
        )
    if test[STATION].nunique() != 16:
        raise ValueError(
            f"Expected 16 test stations, found {test[STATION].nunique()}."
        )
    if set(oof[STATION]) & set(test[STATION]):
        raise ValueError("Development/test station leakage detected.")

    # ---------- Attach context ----------
    context, context_cols = load_context()
    oof = attach_context(oof, context)
    test = attach_context(test, context)

    print(f">>> Development OOF: {len(oof):,} rows, {oof[STATION].nunique()} stations")
    print(f">>> Final test:      {len(test):,} rows, {test[STATION].nunique()} stations")
    print(f">>> Experts: {EXPERT_COLS}")
    print(f">>> PM25 unit factor applied: x{PM25_UNIT_FACTOR:.0e}")

    # Correct physical-unit baseline for the 63 development stations.
    baseline_oof = metrics(
        oof[OBS],
        oof[MUSICA],
        oof[MUSICA],
    )
    baseline_test = metrics(
        test[OBS],
        test[MUSICA],
        test[MUSICA],
    )

    print("\n>>> Correct physical-unit OOF MUSICA baseline")
    print(
        f"R2={baseline_oof['R2_Before']:.6f} | "
        f"MAE={baseline_oof['MAE_Before']:.6f} | "
        f"RMSE={baseline_oof['RMSE_Before']:.6f} | "
        f"BIAS={baseline_oof['BIAS_Before']:.6f}"
    )

    # =====================================================================
    # 1) Meta-level outer GroupKFold OOF evaluation
    # =====================================================================
    outer = GroupKFold(n_splits=N_OUTER)

    meta_oof_parts = []
    fold_rows = []
    history_parts = []
    fold_best_epochs = []

    for fold, (outer_train_idx, outer_val_idx) in enumerate(
        outer.split(oof, groups=oof[STATION]), start=1
    ):
        print(f"\n{'=' * 76}")
        print(f">>> Dynamic MoE meta-OOF fold {fold}/{N_OUTER}")
        print(f"{'=' * 76}")

        outer_train = oof.iloc[outer_train_idx].copy()
        outer_val = oof.iloc[outer_val_idx].copy()

        if set(outer_train[STATION]) & set(outer_val[STATION]):
            raise RuntimeError(f"Outer fold {fold}: station leakage detected.")

        # -------------------------------------------------------------
        # Inner station split ONLY for early stopping.
        # Outer validation stations are never used for early stopping.
        # -------------------------------------------------------------
        inner_split = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=SEED + fold,
        )
        inner_train_idx, inner_val_idx = next(
            inner_split.split(
                outer_train,
                groups=outer_train[STATION],
            )
        )

        inner_train = outer_train.iloc[inner_train_idx].copy()
        inner_val = outer_train.iloc[inner_val_idx].copy()

        if set(inner_train[STATION]) & set(inner_val[STATION]):
            raise RuntimeError(f"Inner fold {fold}: station leakage detected.")

        # Fit scaler only on inner training rows.
        x_inner_train, e_inner_train, scaler, gate_cols = make_gate_matrix(
            inner_train,
            context_cols,
            scaler=None,
            fit_scaler=True,
        )
        x_inner_val, e_inner_val, _, _ = make_gate_matrix(
            inner_val,
            context_cols,
            scaler=scaler,
            fit_scaler=False,
        )
        x_outer_val, e_outer_val, _, _ = make_gate_matrix(
            outer_val,
            context_cols,
            scaler=scaler,
            fit_scaler=False,
        )

        y_inner_train = inner_train[TRUE_BIAS].to_numpy(np.float32)
        y_inner_val = inner_val[TRUE_BIAS].to_numpy(np.float32)

        keras.backend.clear_session()
        set_seed(SEED + fold)
        model, weight_model = build_gate(len(gate_cols))

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_mae",
                mode="min",
                patience=PATIENCE,
                min_delta=MIN_DELTA,
                restore_best_weights=True,
                verbose=1,
            )
        ]

        history = model.fit(
            [x_inner_train, e_inner_train],
            y_inner_train,
            validation_data=(
                [x_inner_val, e_inner_val],
                y_inner_val,
            ),
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            shuffle=True,
            verbose=2,
        )

        best_epoch = int(np.argmin(history.history["val_mae"]) + 1)
        fold_best_epochs.append(best_epoch)

        # Outer validation is touched for the first time here.
        pred_bias, weights = predict_with_weights(
            model,
            weight_model,
            x_outer_val,
            e_outer_val,
        )

        pred = output_table(
            outer_val,
            pred_bias,
            weights,
            phase="Meta_OOF",
            fold_id=fold,
        )
        meta_oof_parts.append(pred)

        row = metrics(
            pred[OBS],
            pred[MUSICA],
            pred["PM2.5_Corrected"],
        )
        row.update({
            "Fold_ID": fold,
            "Best_Epoch": best_epoch,
            "Outer_Train_Rows": len(outer_train),
            "Outer_Validation_Rows": len(outer_val),
            "Outer_Train_Stations": outer_train[STATION].nunique(),
            "Outer_Validation_Stations": outer_val[STATION].nunique(),
            "Inner_Train_Stations": inner_train[STATION].nunique(),
            "Inner_Validation_Stations": inner_val[STATION].nunique(),
            "Mean_Weight_DeepKriging": float(weights[:, 0].mean()),
            "Mean_Weight_CNN": float(weights[:, 1].mean()),
            "Mean_Weight_LightGBM": float(weights[:, 2].mean()),
            "PM25_Unit_Factor": PM25_UNIT_FACTOR,
        })
        fold_rows.append(row)

        history_parts.append(
            pd.DataFrame({
                "Fold_ID": fold,
                "Epoch": np.arange(1, len(history.history["loss"]) + 1),
                "Train_Loss": history.history["loss"],
                "Validation_Loss": history.history["val_loss"],
                "Train_MAE": history.history["mae"],
                "Validation_MAE": history.history["val_mae"],
            })
        )

        print(
            f">>> Fold {fold}: epoch={best_epoch} | "
            f"R2={row['R2_After']:.4f} | "
            f"MAE={row['MAE_After']:.4f} | "
            f"RMSE={row['RMSE_After']:.4f} | "
            f"BIAS={row['BIAS_After']:.4f}"
        )
        print(
            ">>> Mean dynamic weights: "
            f"DK={weights[:,0].mean():.3f}, "
            f"CNN={weights[:,1].mean():.3f}, "
            f"LGBM={weights[:,2].mean():.3f}"
        )

        del model, weight_model
        keras.backend.clear_session()

    # ---------- Combine meta-level OOF ----------
    meta_oof = (
        pd.concat(meta_oof_parts, ignore_index=True)
        .sort_values("row_id")
        .reset_index(drop=True)
    )

    if len(meta_oof) != len(oof):
        raise RuntimeError("Meta OOF does not cover all development rows.")
    if meta_oof["row_id"].duplicated().any():
        raise RuntimeError("Meta OOF contains duplicated row_id values.")
    if set(meta_oof["row_id"]) != set(oof["row_id"]):
        raise RuntimeError("Meta OOF row_id set differs from development OOF data.")

    oof_metric = metrics(
        meta_oof[OBS],
        meta_oof[MUSICA],
        meta_oof["PM2.5_Corrected"],
    )

    # ---------- Station-level metrics ----------
    station_rows = []
    for station, group in meta_oof.groupby(STATION, sort=True):
        row = metrics(
            group[OBS],
            group[MUSICA],
            group["PM2.5_Corrected"],
        )
        row[STATION] = station
        row["Mean_Weight_DeepKriging"] = group["Weight_DeepKriging"].mean()
        row["Mean_Weight_CNN"] = group["Weight_CNN"].mean()
        row["Mean_Weight_LightGBM"] = group["Weight_LightGBM"].mean()
        station_rows.append(row)

    station_metrics = pd.DataFrame(station_rows)

    # =====================================================================
    # 2) Choose final training epoch without touching final test
    # =====================================================================
    #
    # We use the median of the five leakage-free outer-fold best epochs.
    # This is stable and does not use any test information.
    final_epoch = int(np.median(fold_best_epochs))
    final_epoch = max(final_epoch, 1)

    print("\n>>> Meta-level OOF overall metrics")
    print(
        pd.DataFrame([oof_metric])[
            ["R2_After", "MAE_After", "RMSE_After", "BIAS_After"]
        ].to_string(index=False)
    )
    print(f">>> Fold best epochs: {fold_best_epochs}")
    print(f">>> Final gate epoch (median): {final_epoch}")

    # =====================================================================
    # 3) Train final gating network on all 63-station OOF rows
    # =====================================================================
    final_scaler = StandardScaler()

    final_gate_cols = list(context_cols)
    if USE_EXPERT_PREDS_AS_GATE_INPUT:
        final_gate_cols += EXPERT_COLS

    x_dev = final_scaler.fit_transform(
        oof[final_gate_cols].to_numpy(np.float32)
    ).astype(np.float32)
    e_dev = oof[EXPERT_COLS].to_numpy(np.float32)

    x_test = final_scaler.transform(
        test[final_gate_cols].to_numpy(np.float32)
    ).astype(np.float32)
    e_test = test[EXPERT_COLS].to_numpy(np.float32)

    keras.backend.clear_session()
    set_seed(SEED)
    final_model, final_weight_model = build_gate(len(final_gate_cols))

    final_history = final_model.fit(
        [x_dev, e_dev],
        oof[TRUE_BIAS].to_numpy(np.float32),
        epochs=final_epoch,
        batch_size=BATCH_SIZE,
        shuffle=True,
        verbose=2,
    )

    # =====================================================================
    # 4) Final independent 16-station test
    # =====================================================================
    test_bias, test_weights = predict_with_weights(
        final_model,
        final_weight_model,
        x_test,
        e_test,
    )

    test_output = output_table(
        test,
        test_bias,
        test_weights,
        phase="Final_Test",
    )

    test_metric = metrics(
        test_output[OBS],
        test_output[MUSICA],
        test_output["PM2.5_Corrected"],
    )

    # =====================================================================
    # 5) Weight interpretation summaries
    # =====================================================================
    overall_weights = weight_summary(meta_oof)

    # Month/hour summaries if available.
    monthly_weights = None
    hourly_weights = None
    if "Month" in meta_oof.columns:
        monthly_weights = weight_summary(meta_oof, "Month")
    if "Hour" in meta_oof.columns:
        hourly_weights = weight_summary(meta_oof, "Hour")

    # PM25-level dynamic weights.
    pm25_bins = [-np.inf, 5, 10, 15, 25, np.inf]
    pm25_labels = ["<=5", "5-10", "10-15", "15-25", ">25"]
    meta_oof["PM25_Bin"] = pd.cut(
        meta_oof[MUSICA],
        bins=pm25_bins,
        labels=pm25_labels,
        include_lowest=True,
    )
    concentration_weights = weight_summary(meta_oof, "PM25_Bin")

    # Which expert receives the largest gate weight for each sample?
    weight_cols = [
        "Weight_DeepKriging",
        "Weight_CNN",
        "Weight_LightGBM",
    ]
    winner_names = {
        "Weight_DeepKriging": "DeepKriging",
        "Weight_CNN": "CNN",
        "Weight_LightGBM": "LightGBM",
    }
    meta_oof["Dominant_Expert"] = (
        meta_oof[weight_cols]
        .idxmax(axis=1)
        .map(winner_names)
    )

    dominant_share = (
        meta_oof["Dominant_Expert"]
        .value_counts(normalize=True)
        .rename("Share")
        .reset_index()
        .rename(columns={"index": "Expert"})
    )

    # =====================================================================
    # 6) Save unit-correction audit, models and outputs
    # =====================================================================
    unit_audit = pd.DataFrame([
        {
            "PM25_Unit_Factor": PM25_UNIT_FACTOR,
            "Historical_PM25_Median_OOF": float(
                oof["PM25_Raw_Before_Unit_Fix"].median()
            ),
            "Corrected_PM25_Median_OOF": float(oof[MUSICA].median()),
            "Historical_Bias_MAE_OOF": float(
                np.mean(np.abs(oof["Historical_True_Bias"]))
            ),
            "Corrected_Bias_MAE_OOF": float(
                np.mean(np.abs(oof[TRUE_BIAS]))
            ),
            "OOF_R2_Before": baseline_oof["R2_Before"],
            "OOF_MAE_Before": baseline_oof["MAE_Before"],
            "OOF_RMSE_Before": baseline_oof["RMSE_Before"],
            "OOF_BIAS_Before": baseline_oof["BIAS_Before"],
            "Test_R2_Before": baseline_test["R2_Before"],
            "Test_MAE_Before": baseline_test["MAE_Before"],
            "Test_RMSE_Before": baseline_test["RMSE_Before"],
            "Test_BIAS_Before": baseline_test["BIAS_Before"],
        }
    ])
    unit_audit.to_csv(
        OUT / "PM25_unit_correction_audit.csv",
        index=False,
    )

    final_model.save(OUT / "MoE_final_gating_model.keras")
    final_weight_model.save(OUT / "MoE_final_weight_model.keras")
    joblib.dump(final_scaler, OUT / "MoE_gate_scaler.joblib")

    pd.DataFrame(fold_rows).to_csv(
        OUT / "MoE_meta_fold_metrics.csv",
        index=False,
    )

    pd.DataFrame([{
        **oof_metric,
        "Model": "Dynamic_MoE_DK_CNN_LGBM_unit_fixed",
        "Development_Stations": oof[STATION].nunique(),
        "Meta_CV": "5-fold GroupKFold by station",
        "Fold_Best_Epochs": "|".join(map(str, fold_best_epochs)),
        "Final_Epoch": final_epoch,
        "Gate_Features": "|".join(final_gate_cols),
        "PM25_Unit_Factor": PM25_UNIT_FACTOR,
    }]).to_csv(
        OUT / "MoE_OOF_overall_metrics.csv",
        index=False,
    )

    meta_oof.to_csv(
        OUT / "MoE_OOF_predictions_with_dynamic_weights.csv",
        index=False,
    )

    station_metrics.to_csv(
        OUT / "MoE_OOF_station_metrics.csv",
        index=False,
    )

    pd.concat(history_parts, ignore_index=True).to_csv(
        OUT / "MoE_training_history.csv",
        index=False,
    )

    pd.DataFrame(final_history.history).to_csv(
        OUT / "MoE_final_training_history.csv",
        index_label="Epoch",
    )

    test_output.to_csv(
        OUT / "MoE_final_test_predictions_with_dynamic_weights.csv",
        index=False,
    )

    pd.DataFrame([{
        **test_metric,
        "Model": "Dynamic_MoE_DK_CNN_LGBM_unit_fixed",
        "Development_Stations": oof[STATION].nunique(),
        "Test_Stations": test[STATION].nunique(),
        "Final_Epoch": final_epoch,
        "Test_Usage": "final independent evaluation only",
        "PM25_Unit_Factor": PM25_UNIT_FACTOR,
    }]).to_csv(
        OUT / "MoE_final_test_metrics.csv",
        index=False,
    )

    overall_weights.to_csv(
        OUT / "MoE_weight_summary_overall.csv",
        index=False,
    )

    concentration_weights.to_csv(
        OUT / "MoE_weight_summary_by_PM25.csv",
        index=False,
    )

    dominant_share.to_csv(
        OUT / "MoE_dominant_expert_share.csv",
        index=False,
    )

    if monthly_weights is not None:
        monthly_weights.to_csv(
            OUT / "MoE_weight_summary_by_month.csv",
            index=False,
        )

    if hourly_weights is not None:
        hourly_weights.to_csv(
            OUT / "MoE_weight_summary_by_hour.csv",
            index=False,
        )

    # Store the final gate setup in a simple CSV rather than JSON.
    pd.DataFrame([{
        "Model": "Dynamic_MoE_DK_CNN_LGBM_unit_fixed",
        "Experts": "|".join(EXPERT_COLS),
        "Context_Features": "|".join(context_cols),
        "Gate_Input_Features": "|".join(final_gate_cols),
        "Hidden_1": GATE_HIDDEN_1,
        "Hidden_2": GATE_HIDDEN_2,
        "Dropout": GATE_DROPOUT,
        "L2": GATE_L2,
        "Learning_Rate": LEARNING_RATE,
        "Batch_Size": BATCH_SIZE,
        "Max_Epochs_Meta_CV": MAX_EPOCHS,
        "Early_Stopping_Patience": PATIENCE,
        "Final_Epoch": final_epoch,
        "Loss": "Huber(delta=5.0)",
        "Weight_Function": "Softmax",
        "Weight_Constraint": "non-negative and sum-to-one per sample",
        "Development_Stations": 63,
        "Test_Stations": 16,
        "PM25_Unit_Factor": PM25_UNIT_FACTOR,
        "Unit_Correction": "PM25_raw*1e9; True_Bias and expert Predicted_Bias shifted by row-wise unit offset",
    }]).to_csv(
        OUT / "MoE_parameters.csv",
        index=False,
    )

    # =====================================================================
    # 7) Console summary
    # =====================================================================
    print("\n" + "=" * 76)
    print("FINAL RESULTS - UNIT CORRECTED")
    print("=" * 76)

    print("\n>>> Dynamic MoE meta-level OOF metrics")
    print(
        pd.DataFrame([oof_metric])[
            ["R2_After", "MAE_After", "RMSE_After", "BIAS_After"]
        ].to_string(index=False)
    )

    print("\n>>> Dynamic MoE independent-test metrics")
    print(
        pd.DataFrame([test_metric])[
            ["R2_After", "MAE_After", "RMSE_After", "BIAS_After"]
        ].to_string(index=False)
    )

    print("\n>>> Overall OOF dynamic-weight summary")
    print(overall_weights.to_string(index=False))

    print("\n>>> Dominant expert share")
    print(dominant_share.to_string(index=False))

    print(f"\n>>> All results saved to: {OUT}")


if __name__ == "__main__":
    main()
