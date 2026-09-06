"""
Hard PM2.5-bin gating on Ensemble_DK_CNN_LGBM_403525 expert predictions.

Experts come from Ensemble_OOF_predictions_ugm3.csv and
Ensemble_final_test_predictions_ugm3.csv. Predicted bias is rebuilt from
corrected concentrations so the scale matches ug/m^3:

    True_Bias = MUSICA_ugm3 - AURN
    Expert_Predicted_Bias = MUSICA_ugm3 - Expert_PM2.5_Corrected

Protocol matches the original HardBins gate:
    station-grouped 5-fold OOF; simplex MAE weights per bin; no intercept.
    Final weights fit on all 63 development stations, applied once to 16 test
    stations. Full tables, no 2018-12-01 date filter.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


THIS = Path(__file__).resolve()
SPEC = importlib.util.spec_from_file_location(
    "piecewise_gate",
    THIS.with_name("Piecewise_PM25Bin_Gating_DK_CNN_LGBM_unit_fixed.py"),
)
pw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pw)

STATION = pw.STATION
OBS = "AURN_Observation"
MUSICA = "PM25"
TRUE_BIAS = pw.TRUE_BIAS
EXPERT_COLS = pw.EXPERT_COLS
WEIGHT_COLS = pw.WEIGHT_COLS
STATIC_403525 = pw.STATIC_403525
STATIC_503515 = pw.STATIC_503515

ROOT = pw.ROOT
ENS = ROOT / "results" / "Ensemble_DK_CNN_LGBM_403525"
OOF_FILE = ENS / "Ensemble_OOF_predictions_ugm3.csv"
TEST_FILE = ENS / "Ensemble_final_test_predictions_ugm3.csv"
OUT = ROOT / "results" / "HardBins_Gating_Ensemble403525"
OUT.mkdir(parents=True, exist_ok=True)

CORR_MAP = {
    "DeepKriging_Predicted_Bias": "DeepKriging_PM2.5_Corrected",
    "CNN_Predicted_Bias": "CNN_PM2.5_Corrected",
    "LightGBM_Predicted_Bias": "LightGBM_PM2.5_Corrected",
}


def load_ensemble(path, split_name):
    data = pd.read_csv(path, dtype={"row_id": str, STATION: str})
    if "datetime" in data.columns:
        data["datetime"] = pd.to_datetime(data["datetime"])
    if "MUSICA_ugm3" not in data.columns:
        raise ValueError(f"{split_name} missing MUSICA_ugm3")
    missing = [c for c in CORR_MAP.values() if c not in data.columns]
    if missing:
        raise ValueError(f"{split_name} missing {missing}")

    out = data.copy()
    out[MUSICA] = pd.to_numeric(out["MUSICA_ugm3"], errors="raise")
    out[OBS] = pd.to_numeric(out[OBS], errors="raise")
    out[TRUE_BIAS] = out[MUSICA] - out[OBS]
    for bias_col, corr_col in CORR_MAP.items():
        out[bias_col] = out[MUSICA] - pd.to_numeric(out[corr_col], errors="raise")
    out["Static403525_Corrected"] = pd.to_numeric(out["PM2.5_Corrected"], errors="raise")
    out["PM25_Bin"] = pw.assign_pm25_bin(out[MUSICA])
    out["Input_Source"] = "Ensemble_DK_CNN_LGBM_403525"
    keep = [
        "row_id", STATION, MUSICA, OBS, TRUE_BIAS, "PM25_Bin", "Input_Source",
        "Static403525_Corrected", *EXPERT_COLS,
    ]
    if "datetime" in out.columns:
        keep.insert(2, "datetime")
    if "Fold_ID" in out.columns:
        keep.append("Fold_ID")
    print(
        f">>> {split_name}: {len(out):,} rows | "
        f"{out[STATION].nunique()} stations | {path.name}"
    )
    return out[keep].sort_values("row_id").reset_index(drop=True)


def apply_hard(data, mapping, fallback=STATIC_503515):
    weights = pw.weights_from_bins(data["PM25_Bin"], mapping, fallback)
    experts = data[EXPERT_COLS].to_numpy(dtype=float)
    predicted = np.sum(experts * weights, axis=1)
    out = data.copy()
    out[WEIGHT_COLS[0]] = weights[:, 0]
    out[WEIGHT_COLS[1]] = weights[:, 1]
    out[WEIGHT_COLS[2]] = weights[:, 2]
    out["HardBins_Predicted_Bias"] = predicted
    out["PM2.5_Corrected"] = out[MUSICA] - predicted
    return out


def four(data, corrected):
    return pw.metrics(data[OBS], data[MUSICA], corrected)


def main():
    np.random.seed(pw.SEED)
    oof = load_ensemble(OOF_FILE, "OOF")
    test = load_ensemble(TEST_FILE, "TEST")
    oof = pw.attach_fold_ids(oof)

    if oof[STATION].nunique() != 63 or test[STATION].nunique() != 16:
        raise ValueError("Expected 63/16 stations.")
    if set(oof[STATION]) & set(test[STATION]):
        raise ValueError("Development/test station overlap.")

    parts = []
    fold_rows = []
    weight_parts = []
    for fold in sorted(oof["Meta_Fold_ID"].unique()):
        train = oof[oof["Meta_Fold_ID"] != fold]
        val = oof[oof["Meta_Fold_ID"] == fold]
        mapping, weight_table = pw.fit_bin_weights(train)
        weight_table.insert(0, "Fold_ID", int(fold))
        weight_parts.append(weight_table)
        pred = apply_hard(val, mapping)
        parts.append(pred)
        row = four(pred, pred["PM2.5_Corrected"])
        row["Fold_ID"] = int(fold)
        fold_rows.append(row)
        print(
            f">>> Fold {int(fold)}: "
            f"R2={row['R2_After']:.4f} | MAE={row['MAE_After']:.4f} | "
            f"RMSE={row['RMSE_After']:.4f} | BIAS={row['BIAS_After']:+.4f}"
        )

    oof_out = pd.concat(parts, ignore_index=True).sort_values("row_id")
    final_map, final_w = pw.fit_bin_weights(oof)
    final_w.insert(0, "Fold_ID", "Final_63_stations")
    test_out = apply_hard(test, final_map)

    oof_m = four(oof_out, oof_out["PM2.5_Corrected"])
    test_m = four(test_out, test_out["PM2.5_Corrected"])
    static_oof = four(oof, oof["Static403525_Corrected"])
    static_test = four(test, test["Static403525_Corrected"])
    static50_oof = four(
        oof,
        oof[MUSICA] - oof[EXPERT_COLS].to_numpy(float) @ STATIC_503515,
    )
    static50_test = four(
        test,
        test[MUSICA] - test[EXPERT_COLS].to_numpy(float) @ STATIC_503515,
    )

    overall = pd.DataFrame([
        {**four(oof, oof[MUSICA]), "Model": "MUSICA_Before", "Split": "OOF"},
        {**static_oof, "Model": "Static_0.40_0.35_0.25_file", "Split": "OOF"},
        {**static50_oof, "Model": "Static_0.50_0.35_0.15", "Split": "OOF"},
        {**oof_m, "Model": "HardBins_MAE_Ensemble403525", "Split": "OOF"},
        {**four(test, test[MUSICA]), "Model": "MUSICA_Before", "Split": "TEST"},
        {**static_test, "Model": "Static_0.40_0.35_0.25_file", "Split": "TEST"},
        {**static50_test, "Model": "Static_0.50_0.35_0.15", "Split": "TEST"},
        {**test_m, "Model": "HardBins_MAE_Ensemble403525", "Split": "TEST"},
    ])
    overall["Rows"] = overall["Split"].map({"OOF": len(oof), "TEST": len(test)})
    overall["Stations"] = overall["Split"].map({"OOF": 63, "TEST": 16})

    overall.to_csv(OUT / "HardBins_Ensemble403525_overall_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "HardBins_Ensemble403525_fold_metrics.csv", index=False)
    pd.concat(weight_parts, ignore_index=True).to_csv(
        OUT / "HardBins_Ensemble403525_bin_weights_by_fold.csv", index=False
    )
    final_w.to_csv(OUT / "HardBins_Ensemble403525_bin_weights_final.csv", index=False)
    pw.bin_metric_table(oof_out, oof_out["PM2.5_Corrected"], "OOF").to_csv(
        OUT / "HardBins_Ensemble403525_OOF_bin_metrics.csv", index=False
    )
    pw.bin_metric_table(test_out, test_out["PM2.5_Corrected"], "TEST").to_csv(
        OUT / "HardBins_Ensemble403525_TEST_bin_metrics.csv", index=False
    )

    show = ["Model", "Split", "R2_After", "MAE_After", "RMSE_After", "BIAS_After"]
    print("\n>>> Four metrics (full 403525 tables, 63/16, no date filter)")
    print(overall[show].to_string(index=False))
    print("\n>>> Final bin weights on 63 stations")
    print(
        final_w[
            ["PM25_Bin", "N", "Weight_DeepKriging", "Weight_CNN", "Weight_LightGBM"]
        ].to_string(index=False)
    )
    print(f"\n>>> Saved to {OUT}")


if __name__ == "__main__":
    main()
