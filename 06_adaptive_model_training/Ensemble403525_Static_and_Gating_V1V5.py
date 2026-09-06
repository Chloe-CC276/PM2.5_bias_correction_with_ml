"""
Aligned full-table metrics on Ensemble_DK_CNN_LGBM_403525 experts:
    Static 0.50/0.35/0.15
    HardBins
    Gating V1-V5

OOF: 465,659 rows / 63 stations (station 5-fold)
TEST: 127,695 rows / 16 stations (once)
No 2018-12-01 date filter.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().with_name(filename)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hb = load_module("hardbins_403525", "HardBins_Gating_Ensemble403525.py")
fv = load_module("five_variants", "PM25Bin_Gating_FiveVariants_unit_fixed.py")
pw = hb.pw

OUT = pw.RESULTS / "Ensemble403525_Static_and_Gating_V1V5"
OUT.mkdir(parents=True, exist_ok=True)

STATION = hb.STATION
MUSICA = hb.MUSICA
OBS = hb.OBS
EXPERT_COLS = hb.EXPERT_COLS
STATIC_503515 = hb.STATIC_503515
STATIC_403525 = hb.STATIC_403525
VARIANTS = fv.VARIANTS


def load_split(path, name):
    data = hb.load_ensemble(path, name)
    if "datetime" in data.columns:
        data["Hour"] = pd.to_datetime(data["datetime"]).dt.hour
    return data


def row_metrics(data, corrected, model, split):
    return fv.metric_row(data, corrected, model, split)


def main():
    np.random.seed(pw.SEED)
    oof = load_split(hb.OOF_FILE, "OOF")
    test = load_split(hb.TEST_FILE, "TEST")
    oof = pw.attach_fold_ids(oof)

    if oof[STATION].nunique() != 63 or test[STATION].nunique() != 16:
        raise ValueError("Expected 63/16 stations.")
    if set(oof[STATION]) & set(test[STATION]):
        raise ValueError("Station leakage.")
    if set(oof["row_id"]) & set(test["row_id"]):
        raise ValueError("row_id overlap.")

    print("=" * 80)
    print("403525 ALIGNED STATIC + GATING V1-V5")
    print("=" * 80)
    print(f">>> OOF  {len(oof):,} rows / {oof[STATION].nunique()} stations")
    print(f">>> TEST {len(test):,} rows / {test[STATION].nunique()} stations")
    print(f">>> OUT  {OUT}")

    rows = []
    rows.append(row_metrics(oof, oof[MUSICA], "MUSICA_Before", "OOF"))
    rows.append(row_metrics(test, test[MUSICA], "MUSICA_Before", "TEST"))
    rows.append(row_metrics(
        oof, fv.apply_static(oof, STATIC_403525), "Static_0.40_0.35_0.25", "OOF"
    ))
    rows.append(row_metrics(
        test, fv.apply_static(test, STATIC_403525), "Static_0.40_0.35_0.25", "TEST"
    ))
    rows.append(row_metrics(
        oof, fv.apply_static(oof, STATIC_503515), "Static_0.50_0.35_0.15", "OOF"
    ))
    rows.append(row_metrics(
        test, fv.apply_static(test, STATIC_503515), "Static_0.50_0.35_0.15", "TEST"
    ))

    # HardBins
    print("\n>>> HardBins")
    hard_parts = []
    for fold in sorted(oof["Meta_Fold_ID"].unique()):
        train = oof[oof["Meta_Fold_ID"] != fold]
        val = oof[oof["Meta_Fold_ID"] == fold]
        mapping, fallback, _, _ = fv.fit_hard_bin_weights(train)
        weights = fv.lookup_hard(val["PM25_Bin"], mapping, fallback)
        pred = val.copy()
        pred[fv.WEIGHT_COLS[0]] = weights[:, 0]
        pred[fv.WEIGHT_COLS[1]] = weights[:, 1]
        pred[fv.WEIGHT_COLS[2]] = weights[:, 2]
        pred["Gate_Intercept"] = 0.0
        pred["PM2.5_Corrected"] = pred[MUSICA] - np.sum(
            pred[EXPERT_COLS].to_numpy(float) * weights, axis=1
        )
        hard_parts.append(pred)
        m = fv.metric_row(pred, pred["PM2.5_Corrected"], "HardBins", "OOF_fold")
        print(
            f"    fold {int(fold)}: MAE={m['MAE_After']:.4f} | "
            f"RMSE={m['RMSE_After']:.4f} | BIAS={m['BIAS_After']:+.4f} | "
            f"R2={m['R2_After']:.4f}"
        )
    hard_oof = pd.concat(hard_parts, ignore_index=True).sort_values("row_id")
    hard_map, hard_fb, _, _ = fv.fit_hard_bin_weights(oof)
    hard_tw = fv.lookup_hard(test["PM25_Bin"], hard_map, hard_fb)
    hard_test_corr = test[MUSICA] - np.sum(
        test[EXPERT_COLS].to_numpy(float) * hard_tw, axis=1
    )
    rows.append(row_metrics(hard_oof, hard_oof["PM2.5_Corrected"], "HardBins_MAE", "OOF"))
    rows.append(row_metrics(test, hard_test_corr, "HardBins_MAE", "TEST"))

    fold_tables = []
    for name in VARIANTS:
        print(f"\n>>> {name}")
        oof_pred, fold_table = fv.run_oof(oof, name)
        pack = fv.fit_variant(name, oof)
        test_pred = fv.apply_variant(pack, test)
        fold_tables.append(fold_table)
        rows.append(row_metrics(oof_pred, oof_pred["PM2.5_Corrected"], name, "OOF"))
        rows.append(row_metrics(test_pred, test_pred["PM2.5_Corrected"], name, "TEST"))

    overall = pd.DataFrame(rows)
    overall.to_csv(OUT / "Aligned_403525_overall_metrics.csv", index=False)
    if fold_tables:
        pd.concat(fold_tables, ignore_index=True).to_csv(
            OUT / "Aligned_403525_gating_fold_metrics.csv", index=False
        )

    show = ["Model", "Split", "R2_After", "MAE_After", "RMSE_After", "BIAS_After", "Rows", "Stations"]
    print("\n>>> Four metrics (403525 full tables, aligned 63/16)")
    print(overall[show].to_string(index=False))
    print(f"\n>>> Saved to {OUT}")


if __name__ == "__main__":
    main()
