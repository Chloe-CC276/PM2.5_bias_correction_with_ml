"""
Five separate PM2.5-bin gating variants, evaluated one at a time.

Variants (experts frozen: DeepKriging / CNN / LightGBM)
------------------------------------------------------
V1 SoftBins          Hard-bin MAE simplex, soft linear interpolation at apply.
V2 MultiObj          Per-bin MAE+RMSE+|BIAS| simplex, bounded intercept |b|<=0.15.
V3 NestedDayNight    PM25 bin x Day/Night cells; fallback to parent PM25 bin.
V4 Shrinkage         Per-bin MAE simplex shrunk toward static 0.50/0.35/0.15.
V5 ContPath          Continuous simplex path w(PM25) at fixed knots.

Protocol
--------
Same unit-fixed MoE tables, 63/16 split, station-grouped 5-fold OOF.
TEST is computed once per isolated variant for reporting. Do not use TEST
to pick a combination.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


SEED = 42
THIS_FILE = Path(__file__).resolve()
SPEC = importlib.util.spec_from_file_location(
    "piecewise_gate",
    THIS_FILE.with_name("Piecewise_PM25Bin_Gating_DK_CNN_LGBM_unit_fixed.py"),
)
pw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pw)

STATION = pw.STATION
MUSICA = pw.MUSICA
OBS = pw.OBS
TRUE_BIAS = pw.TRUE_BIAS
EXPERT_COLS = pw.EXPERT_COLS
WEIGHT_COLS = pw.WEIGHT_COLS
PM25_LABELS = pw.PM25_LABELS
STATIC_503515 = pw.STATIC_503515
STATIC_403525 = pw.STATIC_403525
N_SPLITS = pw.N_SPLITS

OUT = pw.RESULTS / "PM25Bin_Gating_FiveVariants_unit_fixed"
OUT.mkdir(parents=True, exist_ok=True)

V2_LAM_RMSE = 0.25
V2_LAM_BIAS = 3.0
V2_MAX_ABS_B = 0.15
V3_MIN_CELL_ROWS = 500
V4_N0 = 10000.0
V5_KNOTS = np.array([1.0, 5.0, 10.0, 15.0, 25.0, 50.0], dtype=float)
V5_SMOOTH = 0.20
V5_MAXITER = 80

VARIANTS = [
    "V1_SoftBins",
    "V2_MultiObj",
    "V3_NestedDayNight",
    "V4_Shrinkage",
    "V5_ContPath",
]


def experts_xy(data):
    X = data[EXPERT_COLS].to_numpy(dtype=float)
    y = data[TRUE_BIAS].to_numpy(dtype=float)
    return X, y


def project_simplex(w):
    w = np.clip(np.asarray(w, dtype=float), 0.0, 1.0)
    total = w.sum()
    if total <= 0:
        return STATIC_503515.copy()
    return w / total


def simplex_mae_weights(X, y, init=STATIC_503515, min_rows=pw.MIN_BIN_ROWS):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    init = project_simplex(init)
    if len(y) < min_rows:
        return init.copy(), False, float("nan")

    def objective(w):
        return float(np.mean(np.abs(y - X @ w)))

    result = minimize(
        objective,
        init,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * 3,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        options={"maxiter": 300, "ftol": 1e-12},
    )
    weights = project_simplex(result.x)
    return weights, True, float(objective(weights))


def fit_hard_bin_weights(data, init=STATIC_503515, min_rows=pw.MIN_BIN_ROWS):
    X_all, y_all = experts_xy(data)
    global_w, global_ok, _ = simplex_mae_weights(X_all, y_all, init, min_rows)
    fallback = global_w if global_ok else project_simplex(init)
    mapping = {}
    counts = {}
    medians = {}
    for label in PM25_LABELS:
        part = data[data["PM25_Bin"] == label]
        counts[label] = int(len(part))
        medians[label] = float(part[MUSICA].median()) if len(part) else np.nan
        weights, ok, _ = simplex_mae_weights(
            part[EXPERT_COLS], part[TRUE_BIAS], fallback, min_rows
        )
        mapping[label] = weights if ok else fallback.copy()
    return mapping, fallback, counts, medians


def interpolate_weights(pm25, centers, weight_stack, fallback):
    pm25 = np.asarray(pm25, dtype=float)
    order = np.argsort(centers)
    x = np.asarray(centers, dtype=float)[order]
    W = np.asarray(weight_stack, dtype=float)[order]
    finite = np.isfinite(x)
    x = x[finite]
    W = W[finite]
    if len(x) == 0:
        return np.repeat(fallback.reshape(1, 3), len(pm25), axis=0)
    if len(x) == 1:
        return np.repeat(W[0].reshape(1, 3), len(pm25), axis=0)
    pm = np.clip(pm25, x[0], x[-1])
    out = np.column_stack([np.interp(pm, x, W[:, j]) for j in range(3)])
    out = np.clip(out, 0.0, 1.0)
    totals = out.sum(axis=1, keepdims=True)
    totals[totals <= 0] = 1.0
    return out / totals


def attach_day_night(data):
    out = data.copy()
    if "Hour" in out.columns:
        hour = pd.to_numeric(out["Hour"], errors="coerce")
    else:
        hour = pd.to_datetime(out["datetime"]).dt.hour
    out["DayNight"] = np.where((hour >= 7) & (hour <= 18), "Day", "Night")
    out["Cell"] = out["PM25_Bin"].astype(str) + "|" + out["DayNight"]
    return out


def simplex_multiobj_weights(X, y, init=STATIC_503515):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(y) < pw.MIN_BIN_ROWS:
        return project_simplex(init), 0.0, False

    start = np.concatenate([project_simplex(init), np.array([0.0])])

    def unpack(p):
        return project_simplex(p[:3]), float(np.clip(p[3], -V2_MAX_ABS_B, V2_MAX_ABS_B))

    def objective(p):
        w, b = unpack(p)
        resid = y - (X @ w + b)
        mae = np.mean(np.abs(resid))
        rmse = np.sqrt(np.mean(resid**2))
        bias = np.mean(resid)
        return float(mae + V2_LAM_RMSE * rmse + V2_LAM_BIAS * abs(bias))

    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * 3 + [(-V2_MAX_ABS_B, V2_MAX_ABS_B)],
        constraints={"type": "eq", "fun": lambda p: np.sum(p[:3]) - 1.0},
        options={"maxiter": 400, "ftol": 1e-12},
    )
    w, b = unpack(result.x)
    return w, b, True


def path_from_params(params):
    pairs = np.asarray(params, dtype=float).reshape(len(V5_KNOTS), 2)
    pairs = np.clip(pairs, 0.0, 1.0)
    third = 1.0 - pairs.sum(axis=1)
    W = np.column_stack([pairs[:, 0], pairs[:, 1], third])
    W = np.clip(W, 0.0, 1.0)
    totals = W.sum(axis=1, keepdims=True)
    totals[totals <= 0] = 1.0
    return W / totals


def params_from_path(W):
    W = np.asarray(W, dtype=float)
    return W[:, :2].ravel()


def interpolate_path(pm25, W):
    pm = np.clip(np.asarray(pm25, dtype=float), V5_KNOTS[0], V5_KNOTS[-1])
    out = np.column_stack([np.interp(pm, V5_KNOTS, W[:, j]) for j in range(3)])
    out = np.clip(out, 0.0, 1.0)
    totals = out.sum(axis=1, keepdims=True)
    totals[totals <= 0] = 1.0
    return out / totals


def initial_path_weights(mapping):
    W = np.vstack([mapping[label] for label in PM25_LABELS])
    # knots: 1,5,10,15,25,50  -> assign from nearest hard bin
    bin_rep = np.array([2.5, 7.5, 12.5, 20.0, 35.0])
    assigned = []
    for knot in V5_KNOTS:
        idx = int(np.argmin(np.abs(bin_rep - knot)))
        assigned.append(W[min(idx, len(W) - 1)])
    return np.vstack(assigned)


def fit_cont_path(data, mapping):
    if len(data) > 120_000:
        data = data.sample(n=120_000, random_state=SEED)
    X, y = experts_xy(data)
    pm25 = data[MUSICA].to_numpy(dtype=float)
    W0 = initial_path_weights(mapping)
    start = params_from_path(W0)

    def objective(p):
        W = path_from_params(p)
        weights = interpolate_path(pm25, W)
        pred = np.sum(X * weights, axis=1)
        mae = np.mean(np.abs(y - pred))
        smooth = np.sum((W[1:] - W[:-1]) ** 2)
        return float(mae + V5_SMOOTH * smooth)

    bounds = [(0.0, 1.0)] * (len(V5_KNOTS) * 2)
    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=bounds,
        options={"maxiter": V5_MAXITER, "ftol": 1e-9},
    )
    return path_from_params(result.x)


def fit_variant(name, train):
    mapping, fallback, counts, medians = fit_hard_bin_weights(train)
    pack = {
        "name": name,
        "hard_mapping": mapping,
        "fallback": fallback,
        "counts": counts,
        "medians": medians,
    }

    if name == "V1_SoftBins":
        return pack

    if name == "V2_MultiObj":
        w_map = {}
        b_map = {}
        for label in PM25_LABELS:
            part = train[train["PM25_Bin"] == label]
            w, b, ok = simplex_multiobj_weights(
                part[EXPERT_COLS], part[TRUE_BIAS], mapping[label]
            )
            if not ok:
                w, b = mapping[label], 0.0
            w_map[label] = w
            b_map[label] = float(b)
        pack["w_map"] = w_map
        pack["b_map"] = b_map
        return pack

    if name == "V3_NestedDayNight":
        nested_train = attach_day_night(train)
        cell_map = {}
        for cell, part in nested_train.groupby("Cell", sort=False):
            parent = str(cell).split("|", 1)[0]
            w, ok, _ = simplex_mae_weights(
                part[EXPERT_COLS],
                part[TRUE_BIAS],
                mapping.get(parent, fallback),
                V3_MIN_CELL_ROWS,
            )
            if ok:
                cell_map[str(cell)] = w
        pack["cell_map"] = cell_map
        return pack

    if name == "V4_Shrinkage":
        shrunk = {}
        alphas = {}
        for label in PM25_LABELS:
            n = float(counts[label])
            alpha = n / (n + V4_N0) if n > 0 else 0.0
            alphas[label] = alpha
            shrunk[label] = project_simplex(
                alpha * mapping[label] + (1.0 - alpha) * STATIC_503515
            )
        pack["w_map"] = shrunk
        pack["alphas"] = alphas
        return pack

    if name == "V5_ContPath":
        pack["path_W"] = fit_cont_path(train, mapping)
        return pack

    raise ValueError(name)


def lookup_hard(bins, mapping, fallback):
    return np.vstack([mapping.get(str(label), fallback) for label in bins])


def apply_variant(pack, data):
    name = pack["name"]
    fallback = pack["fallback"]
    n = len(data)
    intercept = np.zeros(n, dtype=float)

    if name == "V1_SoftBins":
        centers = []
        stack = []
        for label in PM25_LABELS:
            med = pack["medians"].get(label, np.nan)
            if np.isfinite(med):
                centers.append(med)
                stack.append(pack["hard_mapping"][label])
        weights = interpolate_weights(
            data[MUSICA], centers, stack, fallback
        )
    elif name == "V2_MultiObj":
        weights = lookup_hard(data["PM25_Bin"], pack["w_map"], fallback)
        intercept = np.array(
            [pack["b_map"].get(str(label), 0.0) for label in data["PM25_Bin"]],
            dtype=float,
        )
    elif name == "V3_NestedDayNight":
        nested = attach_day_night(data)
        weights = np.vstack(
            [
                pack["cell_map"].get(
                    str(cell),
                    pack["hard_mapping"].get(str(cell).split("|", 1)[0], fallback),
                )
                for cell in nested["Cell"]
            ]
        )
    elif name == "V4_Shrinkage":
        weights = lookup_hard(data["PM25_Bin"], pack["w_map"], fallback)
    elif name == "V5_ContPath":
        weights = interpolate_path(data[MUSICA], pack["path_W"])
    else:
        raise ValueError(name)

    experts = data[EXPERT_COLS].to_numpy(dtype=float)
    predicted = np.sum(experts * weights, axis=1) + intercept
    out = data.copy()
    out[WEIGHT_COLS[0]] = weights[:, 0]
    out[WEIGHT_COLS[1]] = weights[:, 1]
    out[WEIGHT_COLS[2]] = weights[:, 2]
    out["Gate_Intercept"] = intercept
    out["Variant_Predicted_Bias"] = predicted
    out["PM2.5_Corrected"] = out[MUSICA] - out["Variant_Predicted_Bias"]
    return out


def metric_row(data, corrected, model, split):
    row = pw.metrics(data[OBS], data[MUSICA], corrected)
    row.update({
        "Model": model,
        "Split": split,
        "Rows": int(len(data)),
        "Stations": int(data[STATION].nunique()),
    })
    return row


def beats(candidate, reference):
    return {
        "better_R2": candidate["R2_After"] > reference["R2_After"],
        "better_MAE": candidate["MAE_After"] < reference["MAE_After"],
        "better_RMSE": candidate["RMSE_After"] < reference["RMSE_After"],
        "better_absBIAS": abs(candidate["BIAS_After"]) < abs(reference["BIAS_After"]),
    }


def all_four(flag_dict):
    return all(flag_dict.values())


def run_oof(oof, name):
    parts = []
    fold_rows = []
    for fold in sorted(oof["Meta_Fold_ID"].unique()):
        train = oof[oof["Meta_Fold_ID"] != fold].copy()
        val = oof[oof["Meta_Fold_ID"] == fold].copy()
        pack = fit_variant(name, train)
        pred = apply_variant(pack, val)
        pred["Meta_Fold_ID"] = int(fold)
        parts.append(pred)
        row = metric_row(pred, pred["PM2.5_Corrected"], name, "OOF_fold")
        row.update({
            "Fold_ID": int(fold),
            "Mean_Weight_DeepKriging": float(pred[WEIGHT_COLS[0]].mean()),
            "Mean_Weight_CNN": float(pred[WEIGHT_COLS[1]].mean()),
            "Mean_Weight_LightGBM": float(pred[WEIGHT_COLS[2]].mean()),
            "Mean_Intercept": float(pred["Gate_Intercept"].mean()),
        })
        fold_rows.append(row)
        print(
            f"    fold {int(fold)}: MAE={row['MAE_After']:.4f} | "
            f"RMSE={row['RMSE_After']:.4f} | "
            f"BIAS={row['BIAS_After']:+.4f} | "
            f"R2={row['R2_After']:.4f}"
        )
    out = (
        pd.concat(parts, ignore_index=True)
        .sort_values("row_id")
        .reset_index(drop=True)
    )
    return out, pd.DataFrame(fold_rows)


def bin_rows(data, split, model):
    rows = []
    for label in PM25_LABELS:
        part = data[data["PM25_Bin"] == label]
        if part.empty:
            continue
        row = metric_row(part, part["PM2.5_Corrected"], model, split)
        row.update({
            "PM25_Bin": label,
            "N": int(len(part)),
            "Mean_Weight_DeepKriging": float(part[WEIGHT_COLS[0]].mean()),
            "Mean_Weight_CNN": float(part[WEIGHT_COLS[1]].mean()),
            "Mean_Weight_LightGBM": float(part[WEIGHT_COLS[2]].mean()),
            "Mean_Intercept": float(part["Gate_Intercept"].mean()),
        })
        rows.append(row)
    return rows


def apply_static(data, weights):
    w = np.asarray(weights, dtype=float).reshape(1, 3)
    pred = data[EXPERT_COLS].to_numpy(dtype=float) @ w.ravel()
    return data[MUSICA].to_numpy(dtype=float) - pred


def main():
    np.random.seed(SEED)
    oof = pw.load_aligned_split(
        pw.MOE_OOF, pw.META_OOF, pw.DK_OOF, pw.CNN_OOF, pw.LGBM_OOF, "OOF"
    )
    test = pw.load_aligned_split(
        pw.MOE_TEST, pw.META_TEST, pw.DK_TEST, pw.CNN_TEST, pw.LGBM_TEST, "TEST"
    )
    oof = pw.attach_fold_ids(oof)

    if oof[STATION].nunique() != 63 or test[STATION].nunique() != 16:
        raise ValueError("Unexpected station counts.")
    if set(oof[STATION]) & set(test[STATION]):
        raise ValueError("Development/test station leakage.")

    print("=" * 80)
    print("FIVE PM2.5-BIN GATING VARIANTS")
    print("=" * 80)
    print(f">>> OUT: {OUT}")
    print(f">>> OOF {len(oof):,} rows / {oof[STATION].nunique()} stations")
    print(f">>> TEST {len(test):,} rows / {test[STATION].nunique()} stations")

    print(">>> Baseline: original hard PM25 bins (MAE simplex)")
    hard_parts = []
    for fold in sorted(oof["Meta_Fold_ID"].unique()):
        train = oof[oof["Meta_Fold_ID"] != fold].copy()
        val = oof[oof["Meta_Fold_ID"] == fold].copy()
        mapping, fallback, _, _ = fit_hard_bin_weights(train)
        weights = lookup_hard(val["PM25_Bin"], mapping, fallback)
        pred = val.copy()
        pred[WEIGHT_COLS[0]] = weights[:, 0]
        pred[WEIGHT_COLS[1]] = weights[:, 1]
        pred[WEIGHT_COLS[2]] = weights[:, 2]
        pred["Gate_Intercept"] = 0.0
        pred["Variant_Predicted_Bias"] = np.sum(
            pred[EXPERT_COLS].to_numpy(dtype=float) * weights, axis=1
        )
        pred["PM2.5_Corrected"] = pred[MUSICA] - pred["Variant_Predicted_Bias"]
        hard_parts.append(pred)
    hard_oof = pd.concat(hard_parts, ignore_index=True).sort_values("row_id")
    hard_final_map, hard_fallback, _, _ = fit_hard_bin_weights(oof)
    hard_test_w = lookup_hard(test["PM25_Bin"], hard_final_map, hard_fallback)
    hard_test = test.copy()
    hard_test[WEIGHT_COLS[0]] = hard_test_w[:, 0]
    hard_test[WEIGHT_COLS[1]] = hard_test_w[:, 1]
    hard_test[WEIGHT_COLS[2]] = hard_test_w[:, 2]
    hard_test["Gate_Intercept"] = 0.0
    hard_test["PM2.5_Corrected"] = hard_test[MUSICA] - np.sum(
        hard_test[EXPERT_COLS].to_numpy(dtype=float) * hard_test_w, axis=1
    )

    baselines = {
        "MUSICA_Before": (
            metric_row(oof, oof[MUSICA], "MUSICA_Before", "OOF"),
            metric_row(test, test[MUSICA], "MUSICA_Before", "TEST"),
        ),
        "Static_0.50_0.35_0.15": (
            metric_row(oof, apply_static(oof, STATIC_503515), "Static_0.50_0.35_0.15", "OOF"),
            metric_row(test, apply_static(test, STATIC_503515), "Static_0.50_0.35_0.15", "TEST"),
        ),
        "Static_0.40_0.35_0.25": (
            metric_row(oof, apply_static(oof, STATIC_403525), "Static_0.40_0.35_0.25", "OOF"),
            metric_row(test, apply_static(test, STATIC_403525), "Static_0.40_0.35_0.25", "TEST"),
        ),
        "HardBins_MAE": (
            metric_row(hard_oof, hard_oof["PM2.5_Corrected"], "HardBins_MAE", "OOF"),
            metric_row(hard_test, hard_test["PM2.5_Corrected"], "HardBins_MAE", "TEST"),
        ),
    }

    overall_rows = []
    fold_rows = []
    bin_metric_rows = []
    param_rows = []

    for split_name, split_metrics in baselines.items():
        overall_rows.extend(split_metrics)

    variant_oof = {}
    variant_test = {}

    for name in VARIANTS:
        print(f"\n>>> {name}")
        oof_pred, fold_table = run_oof(oof, name)
        pack = fit_variant(name, oof)
        test_pred = apply_variant(pack, test)
        variant_oof[name] = oof_pred
        variant_test[name] = test_pred
        fold_rows.append(fold_table)
        overall_rows.append(
            metric_row(oof_pred, oof_pred["PM2.5_Corrected"], name, "OOF")
        )
        overall_rows.append(
            metric_row(test_pred, test_pred["PM2.5_Corrected"], name, "TEST")
        )
        bin_metric_rows.extend(bin_rows(oof_pred, "OOF", name))
        bin_metric_rows.extend(bin_rows(test_pred, "TEST", name))

        rec = {
            "Model": name,
            "Mean_Weight_DK_OOF": float(oof_pred[WEIGHT_COLS[0]].mean()),
            "Mean_Weight_CNN_OOF": float(oof_pred[WEIGHT_COLS[1]].mean()),
            "Mean_Weight_LGBM_OOF": float(oof_pred[WEIGHT_COLS[2]].mean()),
            "Mean_Intercept_OOF": float(oof_pred["Gate_Intercept"].mean()),
        }
        if name == "V2_MultiObj":
            for label in PM25_LABELS:
                rec[f"Intercept_{label}"] = pack["b_map"][label]
        if name == "V4_Shrinkage":
            for label in PM25_LABELS:
                rec[f"Alpha_{label}"] = pack["alphas"][label]
        if name == "V5_ContPath":
            for i, knot in enumerate(V5_KNOTS):
                rec[f"Knot_{knot:g}_DK"] = float(pack["path_W"][i, 0])
                rec[f"Knot_{knot:g}_CNN"] = float(pack["path_W"][i, 1])
                rec[f"Knot_{knot:g}_LGBM"] = float(pack["path_W"][i, 2])
        if name == "V3_NestedDayNight":
            rec["N_Cells_Used"] = len(pack["cell_map"])
        param_rows.append(rec)

    overall = pd.DataFrame(overall_rows)
    folds = pd.concat(fold_rows, ignore_index=True)
    bins = pd.DataFrame(bin_metric_rows)
    params = pd.DataFrame(param_rows)

    hard_oof_m = baselines["HardBins_MAE"][0]
    static_oof_m = baselines["Static_0.50_0.35_0.15"][0]
    hard_test_m = baselines["HardBins_MAE"][1]
    static_test_m = baselines["Static_0.50_0.35_0.15"][1]

    verdict_rows = []
    for name in VARIANTS:
        oof_m = overall[(overall.Model == name) & (overall.Split == "OOF")].iloc[0]
        test_m = overall[(overall.Model == name) & (overall.Split == "TEST")].iloc[0]
        vs_hard_oof = beats(oof_m, hard_oof_m)
        vs_static_oof = beats(oof_m, static_oof_m)
        vs_hard_test = beats(test_m, hard_test_m)
        vs_static_test = beats(test_m, static_test_m)
        verdict_rows.append({
            "Model": name,
            "OOF_R2": oof_m["R2_After"],
            "OOF_MAE": oof_m["MAE_After"],
            "OOF_RMSE": oof_m["RMSE_After"],
            "OOF_BIAS": oof_m["BIAS_After"],
            "TEST_R2": test_m["R2_After"],
            "TEST_MAE": test_m["MAE_After"],
            "TEST_RMSE": test_m["RMSE_After"],
            "TEST_BIAS": test_m["BIAS_After"],
            "OOF_beats_HardBins_all4": all_four(vs_hard_oof),
            "OOF_beats_Static503515_all4": all_four(vs_static_oof),
            "TEST_beats_HardBins_all4": all_four(vs_hard_test),
            "TEST_beats_Static503515_all4": all_four(vs_static_test),
            "OOF_vs_Hard_R2": oof_m["R2_After"] - hard_oof_m["R2_After"],
            "OOF_vs_Hard_MAE": oof_m["MAE_After"] - hard_oof_m["MAE_After"],
            "OOF_vs_Hard_RMSE": oof_m["RMSE_After"] - hard_oof_m["RMSE_After"],
            "OOF_vs_Hard_absBIAS": abs(oof_m["BIAS_After"]) - abs(hard_oof_m["BIAS_After"]),
            "OOF_vs_Static_R2": oof_m["R2_After"] - static_oof_m["R2_After"],
            "OOF_vs_Static_MAE": oof_m["MAE_After"] - static_oof_m["MAE_After"],
            "OOF_vs_Static_RMSE": oof_m["RMSE_After"] - static_oof_m["RMSE_After"],
            "OOF_vs_Static_absBIAS": abs(oof_m["BIAS_After"]) - abs(static_oof_m["BIAS_After"]),
            "Keep_for_merge": bool(
                all_four(vs_hard_oof) or (
                    vs_static_oof["better_MAE"]
                    and vs_static_oof["better_RMSE"]
                    and not vs_hard_oof["better_MAE"] is False
                )
            ),
        })

    verdict = pd.DataFrame(verdict_rows)

    overall.to_csv(OUT / "FiveVariants_overall_metrics.csv", index=False)
    folds.to_csv(OUT / "FiveVariants_fold_metrics.csv", index=False)
    bins.to_csv(OUT / "FiveVariants_bin_metrics.csv", index=False)
    params.to_csv(OUT / "FiveVariants_parameters.csv", index=False)
    verdict.to_csv(OUT / "FiveVariants_verdict.csv", index=False)

    print("\n>>> OOF overall")
    show = overall[overall.Split == "OOF"][
        ["Model", "R2_After", "MAE_After", "RMSE_After", "BIAS_After"]
    ]
    print(show.to_string(index=False))
    print("\n>>> TEST overall")
    show = overall[overall.Split == "TEST"][
        ["Model", "R2_After", "MAE_After", "RMSE_After", "BIAS_After"]
    ]
    print(show.to_string(index=False))
    print("\n>>> Verdict (selection uses OOF; TEST is one-shot reporting)")
    print(
        verdict[
            [
                "Model",
                "OOF_MAE",
                "OOF_RMSE",
                "OOF_BIAS",
                "OOF_beats_HardBins_all4",
                "OOF_beats_Static503515_all4",
                "TEST_MAE",
                "TEST_beats_HardBins_all4",
            ]
        ].to_string(index=False)
    )
    print(f"\n>>> Saved to {OUT}")


if __name__ == "__main__":
    main()
