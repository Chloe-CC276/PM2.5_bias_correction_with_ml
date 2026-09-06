"""
Piecewise PM2.5-bin gating for DK + CNN + LightGBM

This is the simplified alternative to the neural MoE gate:
    within each MUSICA PM2.5 bin, use one fixed simplex weight vector
        w >= 0,  w_DK + w_CNN + w_LGBM = 1,  no intercept
    Predicted_Bias = w(bin) · [DK, CNN, LGBM]
    PM2.5_Corrected = PM25 - Predicted_Bias

Bins match the existing MoE summary exactly:
    <=5, 5-10, 10-15, 15-25, >25   (ug/m^3)

Alignment
---------
Primary input is the already unit-corrected MoE prediction tables, so OOF and
TEST row_id / PM25 / AURN / expert predictions match
    results/MoE_Gating_DK_CNN_LGBM_unit_fixed/
exactly. Neural MoE columns are kept for side-by-side comparison.

If those files are absent, the script falls back to
    MetaLearner_DK_CNN_LGBM_unit_fixed
or reconstructs from single_model_results with the same x1e9 unit fix.

Leakage-free protocol
---------------------
OOF: station-grouped 5-fold. Bin weights are fit on training stations only,
     then applied to held-out stations in that fold.
TEST: one final set of bin weights is fit on all 63 development stations and
      applied once to the 16 test stations.

No AURN / True_Bias is used as a gating input. The only context is MUSICA PM25,
which is available at inference.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


SEED = 42
N_SPLITS = 5
PM25_UNIT_FACTOR = 1e9
MIN_BIN_ROWS = 200
EXPECTED_OOF_BASELINE = {
    "R2": -0.5408,
    "MAE": 6.9357,
    "RMSE": 10.7018,
    "BIAS": -3.7614,
}

HPC_ROOT = Path(
    "/mnt/iusers01/msc-stu/hum-msc-data-sci-2025-2026/"
    "g05844jc/scratch/ERP_project"
)
LOCAL_ROOT = Path(r"F:\Man\ERP")

STATION = "station"
TIME = "datetime"
MUSICA = "PM25"
OBS = "AURN_Observation"
TRUE_BIAS = "True_Bias"
EXPERT_COLS = [
    "DeepKriging_Predicted_Bias",
    "CNN_Predicted_Bias",
    "LightGBM_Predicted_Bias",
]
WEIGHT_COLS = [
    "Weight_DeepKriging",
    "Weight_CNN",
    "Weight_LightGBM",
]

PM25_EDGES = [-np.inf, 5, 10, 15, 25, np.inf]
PM25_LABELS = ["<=5", "5-10", "10-15", "15-25", ">25"]

# Recorded static ensembles and the empirical MoE mean weights by bin.
STATIC_403525 = np.array([0.40, 0.35, 0.25], dtype=float)
STATIC_503515 = np.array([0.50, 0.35, 0.15], dtype=float)
MOE_MEAN_BY_BIN = {
    "<=5": np.array([0.47560155, 0.32303740, 0.20136108]),
    "5-10": np.array([0.47380358, 0.30545630, 0.22074008]),
    "10-15": np.array([0.48167917, 0.29072022, 0.22760063]),
    "15-25": np.array([0.51916444, 0.25313870, 0.22769690]),
    ">25": np.array([0.67923534, 0.16034470, 0.16041997]),
}


def resolve_root():
    if (HPC_ROOT / "results").exists():
        return HPC_ROOT
    if (LOCAL_ROOT / "results").exists():
        return LOCAL_ROOT
    raise FileNotFoundError(
        "Cannot find ERP results under HPC_ROOT or LOCAL_ROOT."
    )


ROOT = resolve_root()
RESULTS = ROOT / "results"
OUT = RESULTS / "Piecewise_PM25Bin_Gating_DK_CNN_LGBM_unit_fixed"
OUT.mkdir(parents=True, exist_ok=True)

MOE_OOF = RESULTS / "MoE_Gating_DK_CNN_LGBM_unit_fixed" / (
    "MoE_OOF_predictions_with_dynamic_weights.csv"
)
MOE_TEST = RESULTS / "MoE_Gating_DK_CNN_LGBM_unit_fixed" / (
    "MoE_final_test_predictions_with_dynamic_weights.csv"
)
META_OOF = RESULTS / "MetaLearner_DK_CNN_LGBM_unit_fixed" / (
    "MetaLearner_OOF_predictions.csv"
)
META_TEST = RESULTS / "MetaLearner_DK_CNN_LGBM_unit_fixed" / (
    "MetaLearner_final_test_predictions.csv"
)
FOLD_FILE = RESULTS / "development_fold_split.csv"
if not FOLD_FILE.exists():
    FOLD_FILE = ROOT / "code" / "data" / "development_fold_split.csv"

DK_OOF = RESULTS / "single_model_results" / "DeepKriging_OOF_predictions.csv"
DK_TEST = RESULTS / "single_model_results" / "DeepKriging_final_test_predictions.csv"
CNN_OOF = RESULTS / "single_model_results" / "CNN_OOF_predictions.csv"
CNN_TEST = RESULTS / "single_model_results" / "CNN_final_test_predictions.csv"
LGBM_OOF = RESULTS / "single_model_results" / "LGBM_OOF_predictions.csv"
LGBM_TEST = RESULTS / "single_model_results" / "LGBM_final_test_predictions.csv"


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics(obs, raw_pm25, corrected_pm25):
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
        "MAE_Improvement_Pct": pct(before["MAE"] - after["MAE"], before["MAE"]),
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


def assign_pm25_bin(pm25):
    return pd.cut(
        pd.to_numeric(pm25, errors="raise"),
        bins=PM25_EDGES,
        labels=PM25_LABELS,
        include_lowest=True,
    ).astype(str)


def simplex_mae_weights(X, y, init=STATIC_503515):
    """Non-negative weights that sum to 1, fitted by MAE. No intercept."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    init = np.asarray(init, dtype=float)
    if len(y) < MIN_BIN_ROWS:
        return init.copy(), False, float("nan")

    def objective(w):
        return np.mean(np.abs(y - X @ w))

    result = minimize(
        objective,
        init,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * 3,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        options={"maxiter": 300, "ftol": 1e-12},
    )
    weights = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
    total = weights.sum()
    if total <= 0:
        return init.copy(), False, float("nan")
    weights = weights / total
    return weights, True, float(objective(weights))


def fit_bin_weights(data, init=STATIC_503515):
    """Fit one simplex vector per PM2.5 bin. Fallback is fold-level global simplex."""
    X_all = data[EXPERT_COLS].to_numpy(dtype=float)
    y_all = data[TRUE_BIAS].to_numpy(dtype=float)
    global_w, global_ok, global_mae = simplex_mae_weights(X_all, y_all, init)
    fallback = global_w if global_ok else init.copy()

    rows = []
    mapping = {}
    for label in PM25_LABELS:
        part = data[data["PM25_Bin"] == label]
        weights, ok, mae = simplex_mae_weights(
            part[EXPERT_COLS], part[TRUE_BIAS], fallback
        )
        if not ok:
            weights = fallback.copy()
            mae = float(
                np.mean(np.abs(part[TRUE_BIAS] - part[EXPERT_COLS].to_numpy() @ weights))
            ) if len(part) else float("nan")
        mapping[label] = weights
        rows.append({
            "PM25_Bin": label,
            "N": int(len(part)),
            "Stations": int(part[STATION].nunique()) if len(part) else 0,
            "Weight_DeepKriging": float(weights[0]),
            "Weight_CNN": float(weights[1]),
            "Weight_LightGBM": float(weights[2]),
            "Train_MAE": mae,
            "Used_Fallback": (not ok),
            "Global_Fallback_DK": float(fallback[0]),
            "Global_Fallback_CNN": float(fallback[1]),
            "Global_Fallback_LGBM": float(fallback[2]),
            "Global_Train_MAE": global_mae,
        })
    return mapping, pd.DataFrame(rows)


def weights_from_bins(bins, mapping, fallback=STATIC_503515):
    return np.vstack(
        [mapping.get(str(label), fallback) for label in bins]
    )


def apply_weights(data, mapping, fallback=STATIC_503515):
    out = data.copy()
    weights = weights_from_bins(out["PM25_Bin"], mapping, fallback)
    experts = out[EXPERT_COLS].to_numpy(dtype=float)
    predicted = np.sum(experts * weights, axis=1)
    out[WEIGHT_COLS[0]] = weights[:, 0]
    out[WEIGHT_COLS[1]] = weights[:, 1]
    out[WEIGHT_COLS[2]] = weights[:, 2]
    out["Piecewise_Predicted_Bias"] = predicted
    out["PM2.5_Corrected"] = out[MUSICA] - out["Piecewise_Predicted_Bias"]
    out["Residual_Before"] = out[MUSICA] - out[OBS]
    out["Residual_After"] = out["PM2.5_Corrected"] - out[OBS]
    out["Absolute_Error_Before"] = out["Residual_Before"].abs()
    out["Absolute_Error_After"] = out["Residual_After"].abs()
    out["Dominant_Expert"] = pd.Series(
        np.array(["DeepKriging", "CNN", "LightGBM"])[weights.argmax(axis=1)],
        index=out.index,
    )
    return out


def apply_fixed_scheme(data, mapping_or_vector, name):
    if isinstance(mapping_or_vector, dict):
        weights = weights_from_bins(data["PM25_Bin"], mapping_or_vector)
    else:
        weights = np.repeat(
            np.asarray(mapping_or_vector, dtype=float).reshape(1, 3),
            len(data),
            axis=0,
        )
    predicted = np.sum(data[EXPERT_COLS].to_numpy(dtype=float) * weights, axis=1)
    corrected = data[MUSICA].to_numpy(dtype=float) - predicted
    row = metrics(data[OBS], data[MUSICA], corrected)
    row["Model"] = name
    row["Rows"] = int(len(data))
    row["Stations"] = int(data[STATION].nunique())
    return row, predicted, corrected


def validate_and_fix_units(data, model_name):
    data = data.copy()
    raw_pm25 = data[MUSICA].astype(float).to_numpy()
    obs = data[OBS].astype(float).to_numpy()
    old_true_bias = data[TRUE_BIAS].astype(float).to_numpy()
    old_pred_bias = data["Predicted_Bias"].astype(float).to_numpy()
    expected_old_bias = raw_pm25 - obs
    if not np.allclose(old_true_bias, expected_old_bias, rtol=1e-5, atol=1e-5):
        max_diff = float(np.max(np.abs(old_true_bias - expected_old_bias)))
        raise ValueError(
            f"{model_name}: historical True_Bias is not PM25_raw - AURN. "
            f"Maximum difference = {max_diff:.6g}."
        )
    corrected_pm25 = raw_pm25 * PM25_UNIT_FACTOR
    unit_offset = corrected_pm25 - raw_pm25
    data["PM25_Raw_Before_Unit_Fix"] = raw_pm25
    data["Historical_True_Bias"] = old_true_bias
    data["Historical_Predicted_Bias"] = old_pred_bias
    data["PM25_Unit_Offset"] = unit_offset
    data[MUSICA] = corrected_pm25
    data[TRUE_BIAS] = corrected_pm25 - obs
    data["Predicted_Bias"] = old_pred_bias + unit_offset
    return data


def read_prediction(path, model_name):
    data = pd.read_csv(path, dtype={"row_id": str, STATION: str})
    if TIME in data.columns:
        data[TIME] = pd.to_datetime(data[TIME])
    data = validate_and_fix_units(data, model_name)
    keep = ["row_id", STATION]
    if TIME in data.columns:
        keep.append(TIME)
    keep += [
        MUSICA, OBS, TRUE_BIAS,
        "PM25_Raw_Before_Unit_Fix", "Historical_True_Bias",
        "PM25_Unit_Offset", "Predicted_Bias",
    ]
    data = data[keep].rename(
        columns={"Predicted_Bias": f"{model_name}_Predicted_Bias"}
    )
    return data


def merge_three(dk_path, cnn_path, lgbm_path):
    dk = read_prediction(dk_path, "DeepKriging")
    cnn = read_prediction(cnn_path, "CNN")
    gbm = read_prediction(lgbm_path, "LightGBM")
    merged = dk.merge(
        cnn[["row_id", "CNN_Predicted_Bias"]],
        on="row_id", how="inner", validate="one_to_one",
    ).merge(
        gbm[["row_id", "LightGBM_Predicted_Bias"]],
        on="row_id", how="inner", validate="one_to_one",
    )
    return merged.sort_values("row_id").reset_index(drop=True)


def prepare_aligned_table(path, source_name):
    data = pd.read_csv(path, dtype={"row_id": str, STATION: str})
    if TIME in data.columns:
        data[TIME] = pd.to_datetime(data[TIME])
    required = {"row_id", STATION, MUSICA, OBS, TRUE_BIAS, *EXPERT_COLS}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{source_name} missing columns: {sorted(missing)}")
    if data["row_id"].duplicated().any():
        raise ValueError(f"{source_name} has duplicated row_id.")
    data["PM25_Bin"] = assign_pm25_bin(data[MUSICA])
    if data["PM25_Bin"].eq("nan").any():
        raise ValueError(f"{source_name} produced missing PM25_Bin labels.")
    data["Input_Source"] = source_name
    return data.sort_values("row_id").reset_index(drop=True)


def load_aligned_split(moe_path, meta_path, dk_path, cnn_path, lgbm_path, split_name):
    if moe_path.exists():
        print(f">>> {split_name}: using MoE unit-fixed predictions")
        return prepare_aligned_table(moe_path, "MoE_Gating_DK_CNN_LGBM_unit_fixed")
    if meta_path.exists():
        print(f">>> {split_name}: using MetaLearner unit-fixed predictions")
        return prepare_aligned_table(meta_path, "MetaLearner_DK_CNN_LGBM_unit_fixed")
    print(f">>> {split_name}: reconstructing from single_model_results")
    merged = merge_three(dk_path, cnn_path, lgbm_path)
    merged["PM25_Bin"] = assign_pm25_bin(merged[MUSICA])
    merged["Input_Source"] = "single_model_results_unit_fixed"
    return merged.sort_values("row_id").reset_index(drop=True)


def attach_fold_ids(oof):
    if FOLD_FILE.exists():
        folds = pd.read_csv(FOLD_FILE, dtype={"row_id": str})
        if folds["row_id"].duplicated().any():
            raise ValueError("Duplicate row_id in development_fold_split.csv")
        oof = oof.drop(columns=["Meta_Fold_ID"], errors="ignore")
        oof = oof.merge(
            folds[["row_id", "fold_id"]].rename(columns={"fold_id": "Meta_Fold_ID"}),
            on="row_id",
            how="left",
            validate="one_to_one",
        )
        if oof["Meta_Fold_ID"].isna().any():
            missing = int(oof["Meta_Fold_ID"].isna().sum())
            raise ValueError(
                f"development_fold_split.csv does not cover {missing} OOF rows."
            )
        oof["Meta_Fold_ID"] = oof["Meta_Fold_ID"].astype(int)
        print(">>> OOF folds taken from development_fold_split.csv")
        return oof

    if "Meta_Fold_ID" in oof.columns and oof["Meta_Fold_ID"].notna().all():
        oof["Meta_Fold_ID"] = oof["Meta_Fold_ID"].astype(int)
        print(">>> OOF folds taken from the MoE Meta_Fold_ID column")
        return oof

    cv = GroupKFold(n_splits=N_SPLITS)
    fold_ids = np.empty(len(oof), dtype=int)
    for fold, (_, val_idx) in enumerate(
        cv.split(oof, groups=oof[STATION]), start=1
    ):
        fold_ids[val_idx] = fold
    oof["Meta_Fold_ID"] = fold_ids
    print(">>> OOF folds assigned by GroupKFold(station)")
    return oof


def warn_if_baseline_unexpected(split_name, metric_row):
    if split_name != "OOF":
        return
    mae = metric_row["MAE_Before"]
    if abs(mae - EXPECTED_OOF_BASELINE["MAE"]) > 0.02:
        print(
            "WARNING: OOF MUSICA baseline MAE is "
            f"{mae:.4f}, expected ~{EXPECTED_OOF_BASELINE['MAE']:.4f}. "
            "Check that the input predictions are unit-corrected."
        )


def comparison_table(data, piecewise_corrected, extra_models):
    rows = []
    schemes = [
        ("MUSICA_Before", None),
        ("DeepKriging", np.array([1.0, 0.0, 0.0])),
        ("CNN", np.array([0.0, 1.0, 0.0])),
        ("LightGBM", np.array([0.0, 0.0, 1.0])),
        ("Static_0.40_0.35_0.25", STATIC_403525),
        ("Static_0.50_0.35_0.15", STATIC_503515),
        ("MoE_mean_weights_by_PM25_bin", MOE_MEAN_BY_BIN),
    ]
    for name, spec in schemes:
        if spec is None:
            row = metrics(data[OBS], data[MUSICA], data[MUSICA])
            row["Model"] = name
            row["Rows"] = int(len(data))
            row["Stations"] = int(data[STATION].nunique())
        else:
            row, _, _ = apply_fixed_scheme(data, spec, name)
        rows.append(row)

    if "MoE_Predicted_Bias" in data.columns:
        moe_corr = data[MUSICA] - data["MoE_Predicted_Bias"]
        row = metrics(data[OBS], data[MUSICA], moe_corr)
        row.update({
            "Model": "Dynamic_MoE_recorded",
            "Rows": int(len(data)),
            "Stations": int(data[STATION].nunique()),
        })
        rows.append(row)

    piece = metrics(data[OBS], data[MUSICA], piecewise_corrected)
    piece.update({
        "Model": "Piecewise_PM25Bin_MAE_simplex",
        "Rows": int(len(data)),
        "Stations": int(data[STATION].nunique()),
    })
    rows.append(piece)
    rows.extend(extra_models)
    return pd.DataFrame(rows)


def bin_metric_table(data, corrected, split_name):
    rows = []
    tmp = data.copy()
    tmp["_corrected"] = np.asarray(corrected)
    for label in PM25_LABELS:
        part = tmp[tmp["PM25_Bin"] == label]
        if part.empty:
            continue
        row = metrics(part[OBS], part[MUSICA], part["_corrected"])
        row.update({
            "Split": split_name,
            "PM25_Bin": label,
            "N": int(len(part)),
            "Stations": int(part[STATION].nunique()),
            "Mean_Weight_DeepKriging": float(part[WEIGHT_COLS[0]].mean()),
            "Mean_Weight_CNN": float(part[WEIGHT_COLS[1]].mean()),
            "Mean_Weight_LightGBM": float(part[WEIGHT_COLS[2]].mean()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def station_metric_table(data):
    rows = []
    for station, group in data.groupby(STATION, sort=True):
        row = metrics(group[OBS], group[MUSICA], group["PM2.5_Corrected"])
        row.update({
            "station": station,
            "Mean_Weight_DeepKriging": float(group[WEIGHT_COLS[0]].mean()),
            "Mean_Weight_CNN": float(group[WEIGHT_COLS[1]].mean()),
            "Mean_Weight_LightGBM": float(group[WEIGHT_COLS[2]].mean()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    np.random.seed(SEED)

    oof = load_aligned_split(MOE_OOF, META_OOF, DK_OOF, CNN_OOF, LGBM_OOF, "OOF")
    test = load_aligned_split(MOE_TEST, META_TEST, DK_TEST, CNN_TEST, LGBM_TEST, "TEST")
    oof = attach_fold_ids(oof)

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
    if set(oof["row_id"]) & set(test["row_id"]):
        raise ValueError("OOF/TEST row_id overlap detected.")

    baseline_oof = metrics(oof[OBS], oof[MUSICA], oof[MUSICA])
    baseline_test = metrics(test[OBS], test[MUSICA], test[MUSICA])
    warn_if_baseline_unexpected("OOF", baseline_oof)

    print("=" * 80)
    print("PIECEWISE PM2.5-BIN GATING")
    print("=" * 80)
    print(f">>> ROOT: {ROOT}")
    print(f">>> OOF rows:  {len(oof):,} | stations: {oof[STATION].nunique()}")
    print(f">>> Test rows: {len(test):,} | stations: {test[STATION].nunique()}")
    print(
        ">>> OOF MUSICA baseline: "
        f"R2={baseline_oof['R2_Before']:.4f} | "
        f"MAE={baseline_oof['MAE_Before']:.4f} | "
        f"RMSE={baseline_oof['RMSE_Before']:.4f} | "
        f"BIAS={baseline_oof['BIAS_Before']:.4f}"
    )
    print(">>> OOF PM25_Bin counts:")
    print(oof["PM25_Bin"].value_counts().reindex(PM25_LABELS).to_string())

    fold_pred_parts = []
    fold_metric_rows = []
    fold_weight_parts = []

    for fold in sorted(oof["Meta_Fold_ID"].unique()):
        train = oof[oof["Meta_Fold_ID"] != fold].copy()
        val = oof[oof["Meta_Fold_ID"] == fold].copy()
        mapping, weight_table = fit_bin_weights(train)
        weight_table.insert(0, "Fold_ID", int(fold))
        fold_weight_parts.append(weight_table)

        pred = apply_weights(val, mapping)
        pred["Piecewise_Phase"] = "Meta_OOF"
        fold_pred_parts.append(pred)

        row = metrics(pred[OBS], pred[MUSICA], pred["PM2.5_Corrected"])
        row.update({
            "Fold_ID": int(fold),
            "Outer_Train_Rows": int(len(train)),
            "Outer_Validation_Rows": int(len(val)),
            "Outer_Train_Stations": int(train[STATION].nunique()),
            "Outer_Validation_Stations": int(val[STATION].nunique()),
            "Mean_Weight_DeepKriging": float(pred[WEIGHT_COLS[0]].mean()),
            "Mean_Weight_CNN": float(pred[WEIGHT_COLS[1]].mean()),
            "Mean_Weight_LightGBM": float(pred[WEIGHT_COLS[2]].mean()),
            "PM25_Unit_Factor": PM25_UNIT_FACTOR,
        })
        fold_metric_rows.append(row)
        print(
            f">>> Fold {int(fold)}: "
            f"R2={row['R2_After']:.4f} | "
            f"MAE={row['MAE_After']:.4f} | "
            f"mean w=({row['Mean_Weight_DeepKriging']:.3f}, "
            f"{row['Mean_Weight_CNN']:.3f}, "
            f"{row['Mean_Weight_LightGBM']:.3f})"
        )

    oof_out = (
        pd.concat(fold_pred_parts, ignore_index=True)
        .sort_values("row_id")
        .reset_index(drop=True)
    )
    if len(oof_out) != len(oof) or set(oof_out["row_id"]) != set(oof["row_id"]):
        raise RuntimeError("Piecewise OOF is not aligned with the source OOF rows.")

    final_mapping, final_weight_table = fit_bin_weights(oof)
    final_weight_table.insert(0, "Fold_ID", "Final_63_stations")
    test_out = apply_weights(test, final_mapping)
    test_out["Piecewise_Phase"] = "Final_Test"
    test_out["Meta_Fold_ID"] = np.nan

    oof_metric = metrics(oof_out[OBS], oof_out[MUSICA], oof_out["PM2.5_Corrected"])
    test_metric = metrics(test_out[OBS], test_out[MUSICA], test_out["PM2.5_Corrected"])

    oof_compare = comparison_table(oof_out, oof_out["PM2.5_Corrected"], [])
    test_compare = comparison_table(test_out, test_out["PM2.5_Corrected"], [])
    oof_bin_metrics = bin_metric_table(oof_out, oof_out["PM2.5_Corrected"], "OOF")
    test_bin_metrics = bin_metric_table(test_out, test_out["PM2.5_Corrected"], "TEST")
    station_metrics = station_metric_table(oof_out)

    pd.DataFrame([{
        "PM25_Unit_Factor": PM25_UNIT_FACTOR,
        "OOF_Input_Source": oof["Input_Source"].iloc[0],
        "TEST_Input_Source": test["Input_Source"].iloc[0],
        "OOF_Rows": int(len(oof)),
        "TEST_Rows": int(len(test)),
        "OOF_R2_Before": baseline_oof["R2_Before"],
        "OOF_MAE_Before": baseline_oof["MAE_Before"],
        "OOF_RMSE_Before": baseline_oof["RMSE_Before"],
        "OOF_BIAS_Before": baseline_oof["BIAS_Before"],
        "Test_R2_Before": baseline_test["R2_Before"],
        "Test_MAE_Before": baseline_test["MAE_Before"],
        "Test_RMSE_Before": baseline_test["RMSE_Before"],
        "Test_BIAS_Before": baseline_test["BIAS_Before"],
        "Alignment": "same row_id as MoE/MetaLearner unit-fixed prediction tables",
    }]).to_csv(OUT / "PM25_unit_correction_audit.csv", index=False)

    pd.DataFrame([{
        **oof_metric,
        "Model": "Piecewise_PM25Bin_Gating_DK_CNN_LGBM_unit_fixed",
        "Development_Stations": oof[STATION].nunique(),
        "Meta_CV": "5-fold station split; MAE simplex weights per PM25 bin",
        "Gate_Features": "PM25_Bin",
        "PM25_Unit_Factor": PM25_UNIT_FACTOR,
        "Input_Source": oof["Input_Source"].iloc[0],
    }]).to_csv(OUT / "Piecewise_OOF_overall_metrics.csv", index=False)

    pd.DataFrame([{
        **test_metric,
        "Model": "Piecewise_PM25Bin_Gating_DK_CNN_LGBM_unit_fixed",
        "Development_Stations": oof[STATION].nunique(),
        "Test_Stations": test[STATION].nunique(),
        "Test_Usage": "final independent evaluation only",
        "PM25_Unit_Factor": PM25_UNIT_FACTOR,
        "Input_Source": test["Input_Source"].iloc[0],
    }]).to_csv(OUT / "Piecewise_final_test_metrics.csv", index=False)

    pd.DataFrame(fold_metric_rows).to_csv(
        OUT / "Piecewise_meta_fold_metrics.csv", index=False
    )
    pd.concat(fold_weight_parts, ignore_index=True).to_csv(
        OUT / "Piecewise_bin_weights_by_fold.csv", index=False
    )
    final_weight_table.to_csv(OUT / "Piecewise_bin_weights_final.csv", index=False)
    oof_compare.to_csv(OUT / "Piecewise_OOF_model_comparison.csv", index=False)
    test_compare.to_csv(OUT / "Piecewise_TEST_model_comparison.csv", index=False)
    oof_bin_metrics.to_csv(OUT / "Piecewise_OOF_bin_metrics.csv", index=False)
    test_bin_metrics.to_csv(OUT / "Piecewise_TEST_bin_metrics.csv", index=False)
    station_metrics.to_csv(OUT / "Piecewise_OOF_station_metrics.csv", index=False)
    oof_out.to_csv(
        OUT / "Piecewise_OOF_predictions_with_bin_weights.csv", index=False
    )
    test_out.to_csv(
        OUT / "Piecewise_final_test_predictions_with_bin_weights.csv", index=False
    )
    pd.DataFrame([{
        "Model": "Piecewise_PM25Bin_Gating_DK_CNN_LGBM_unit_fixed",
        "Experts": "|".join(EXPERT_COLS),
        "Bins": "|".join(PM25_LABELS),
        "Bin_Edges_ugm3": "-inf|5|10|15|25|inf",
        "Weight_Constraint": "non-negative and sum-to-one per bin; no intercept",
        "Objective": "MAE of corrected PM2.5 / equivalently MAE of Bias",
        "OOF_Protocol": "station-grouped 5-fold; bin weights fit on train stations only",
        "TEST_Protocol": "final bin weights fit on all 63 development stations",
        "Development_Stations": 63,
        "Test_Stations": 16,
        "PM25_Unit_Factor": PM25_UNIT_FACTOR,
    }]).to_csv(OUT / "Piecewise_parameters.csv", index=False)

    print("\n>>> Final bin weights fitted on 63 development stations")
    print(
        final_weight_table[
            [
                "PM25_Bin", "N", "Weight_DeepKriging",
                "Weight_CNN", "Weight_LightGBM", "Train_MAE",
            ]
        ].to_string(index=False)
    )
    print("\n>>> Piecewise OOF")
    print(
        pd.DataFrame([oof_metric])[
            ["R2_After", "MAE_After", "RMSE_After", "BIAS_After"]
        ].to_string(index=False)
    )
    print("\n>>> Piecewise independent TEST")
    print(
        pd.DataFrame([test_metric])[
            ["R2_After", "MAE_After", "RMSE_After", "BIAS_After"]
        ].to_string(index=False)
    )
    print("\n>>> OOF comparison")
    print(
        oof_compare[["Model", "R2_After", "MAE_After", "RMSE_After", "BIAS_After"]]
        .to_string(index=False)
    )
    print("\n>>> TEST comparison")
    print(
        test_compare[["Model", "R2_After", "MAE_After", "RMSE_After", "BIAS_After"]]
        .to_string(index=False)
    )
    print(f"\n>>> Saved to: {OUT}")


if __name__ == "__main__":
    main()
