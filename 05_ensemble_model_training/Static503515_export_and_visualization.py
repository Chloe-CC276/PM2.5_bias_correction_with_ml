"""
Export Static 0.50/0.35/0.15 ensemble predictions (403525 expert pool)
and run Ensemble_final_visualization-style analysis on the outputs.

Inputs
------
results/Ensemble_DK_CNN_LGBM_403525/Ensemble_OOF_predictions_ugm3.csv
results/Ensemble_DK_CNN_LGBM_403525/Ensemble_final_test_predictions_ugm3.csv

Outputs
-------
results/Static_503515_Ensemble403525/
    Static503515_OOF_predictions_ugm3.csv
    Static503515_final_test_predictions_ugm3.csv
    Static503515_overall_metrics.csv
    Table_S1_overall_before_after.csv
    Table_S2_model_comparison.csv
    Table_S7_fold_stability.csv
    Table_P3_site_type_summary.csv
    Table_P4_station_metrics_with_type.csv
    Fig_*.png
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(r"results")
SRC = ROOT / "Ensemble_DK_CNN_LGBM_403525"
OUT = ROOT / "Static_503515_Ensemble403525"
OUT.mkdir(parents=True, exist_ok=True)

OOF_SRC = SRC / "Ensemble_OOF_predictions_ugm3.csv"
TEST_SRC = SRC / "Ensemble_final_test_predictions_ugm3.csv"
FLAML_OOF = ROOT / "FLAML_AutoML_baseline" / "FLAML_OOF_predictions.csv"
FLAML_TEST = ROOT / "FLAML_AutoML_baseline" / "FLAML_final_test_predictions.csv"

OOF_OUT = OUT / "Static503515_OOF_predictions_ugm3.csv"
TEST_OUT = OUT / "Static503515_final_test_predictions_ugm3.csv"

STATIC = np.array([0.50, 0.35, 0.15], dtype=float)
CORR_COLS = [
    "DeepKriging_PM2.5_Corrected",
    "CNN_PM2.5_Corrected",
    "LightGBM_PM2.5_Corrected",
]
BIAS_COLS = [
    "DeepKriging_Predicted_Bias",
    "CNN_Predicted_Bias",
    "LightGBM_Predicted_Bias",
]
MODEL_NAME = "Static_0.50_0.35_0.15"

DATE_START = pd.Timestamp("2017-12-01")
DATE_END = pd.Timestamp("2018-12-01")

RURAL = {
    "Auchencorth Moss",
    "Chilbolton Observatory",
    "Lough Navar",
    "Narberth",
    "Eskdalemuir",
    "High Muffles",
    "Yarner Wood",
    "Glazebury",
    "Rochester Stoke",
    "Charlton Mackrell",
    "St Osyth",
}

C = {
    "obs": "black",
    "raw": "dimgray",
    "ens": "#1f77b4",
    "dk": "#9467bd",
    "cnn": "#2ca02c",
    "lgbm": "#ff7f0e",
    "flaml": "#b8860b",
    "before": "dimgray",
    "after": "#1f77b4",
}


def metrics(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ss_res = ((p - y) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return {
        "R2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "MAE": float(np.abs(p - y).mean()),
        "RMSE": float(np.sqrt(((p - y) ** 2).mean())),
        "BIAS": float((p - y).mean()),
    }


def apply_static(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    musica = pd.to_numeric(out["MUSICA_ugm3"], errors="raise")
    obs = pd.to_numeric(out["AURN_Observation"], errors="raise")
    corrected = out[CORR_COLS].to_numpy(dtype=float) @ STATIC
    pred_bias = musica - corrected

    out["True_Bias"] = musica - obs
    for bias_col, corr_col in zip(BIAS_COLS, CORR_COLS):
        out[bias_col] = musica - pd.to_numeric(out[corr_col], errors="raise")
    out["Ensemble_Predicted_Bias"] = pred_bias
    out["PM2.5_Corrected"] = corrected
    out["Residual_Before"] = out["True_Bias"]
    out["Residual_After"] = corrected - obs
    out["Absolute_Error_Before"] = out["Residual_Before"].abs()
    out["Absolute_Error_After"] = out["Residual_After"].abs()
    out["Weight_DeepKriging"] = STATIC[0]
    out["Weight_CNN"] = STATIC[1]
    out["Weight_LightGBM"] = STATIC[2]
    out["Model"] = MODEL_NAME
    return out


def export_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    oof = pd.read_csv(OOF_SRC, dtype={"row_id": str, "station": str})
    test = pd.read_csv(TEST_SRC, dtype={"row_id": str, "station": str})

    oof = apply_static(oof).sort_values("row_id").reset_index(drop=True)
    test = apply_static(test).sort_values("row_id").reset_index(drop=True)

    oof.to_csv(OOF_OUT, index=False)
    test.to_csv(TEST_OUT, index=False)

    rows = []
    for split_name, frame in [("OOF", oof), ("TEST", test)]:
        m = metrics(frame["AURN_Observation"], frame["PM2.5_Corrected"])
        mb = metrics(frame["AURN_Observation"], frame["MUSICA_ugm3"])
        rows.append(
            {
                "Model": MODEL_NAME,
                "Split": split_name,
                "Rows": len(frame),
                "Stations": frame["station"].nunique(),
                "R2_Before": mb["R2"],
                "R2_After": m["R2"],
                "MAE_Before": mb["MAE"],
                "MAE_After": m["MAE"],
                "RMSE_Before": mb["RMSE"],
                "RMSE_After": m["RMSE"],
                "BIAS_Before": mb["BIAS"],
                "BIAS_After": m["BIAS"],
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "Static503515_overall_metrics.csv", index=False)
    print(f">>> Saved predictions to {OUT}")
    print(f"    OOF  {len(oof):,} rows / {oof['station'].nunique()} stations")
    print(f"    TEST {len(test):,} rows / {test['station'].nunique()} stations")
    return oof, test


def load_viz(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["station"] = df["station"].astype(str).str.strip()
    df["datetime"] = pd.to_datetime(df["datetime"])
    raw_col = "MUSICA_ugm3" if "MUSICA_ugm3" in df.columns else "PM25"
    df["obs"] = df["AURN_Observation"]
    df["raw"] = df[raw_col]
    df["corr"] = df["PM2.5_Corrected"]
    return df


def site_type(name: str) -> str:
    if name in RURAL:
        return "Rural background"
    if re.search(r"Roadside|Kerbside|\bA\d+|Road$", name):
        return "Urban traffic"
    return "Urban background"


def filter_training_window(oof: pd.DataFrame, test: pd.DataFrame):
    n_oof, n_test = len(oof), len(test)
    oof = oof[(oof["datetime"] >= DATE_START) & (oof["datetime"] < DATE_END)].copy()
    test = test[(test["datetime"] >= DATE_START) & (test["datetime"] < DATE_END)].copy()
    print(
        f">>> Date filter [{DATE_START.date()} , {DATE_END.date()}): "
        f"dropped OOF {n_oof - len(oof)}, Test {n_test - len(test)}"
    )
    for d in (oof, test):
        d["site_type"] = d["station"].map(site_type)
    return oof, test


def pct_improve(old, new):
    return np.nan if np.isclose(old, 0) else 100.0 * (old - new) / abs(old)


def M(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {
        "R2": 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum(),
        "MAE": float(np.abs(p - y).mean()),
        "RMSE": float(np.sqrt(((p - y) ** 2).mean())),
        "BIAS": float((p - y).mean()),
    }


def run_visualization(oof: pd.DataFrame, test: pd.DataFrame) -> None:
    # Figure code is a direct port of Ensemble_final_visualization.ipynb
    # (cells 2–14). After = Static 0.50/0.35/0.15; outputs go to OUT.
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )

    oof = load_viz(OOF_OUT)
    test = load_viz(TEST_OUT)
    oof, test = filter_training_window(oof, test)

    print(f"OOF : {len(oof):,} rows, {oof['station'].nunique()} stations")
    print(f"Test: {len(test):,} rows, {test['station'].nunique()} stations")
    print("\nStations per (rough) site type:")
    print(
        pd.concat(
            [
                oof.groupby("site_type")["station"].nunique().rename("OOF"),
                test.groupby("site_type")["station"].nunique().rename("Test"),
            ],
            axis=1,
        )
    )

    flaml_oof = pd.read_csv(FLAML_OOF)[["row_id", "PM2.5_Corrected"]].rename(
        columns={"PM2.5_Corrected": "flaml"}
    )
    flaml_test = pd.read_csv(FLAML_TEST)[["row_id", "PM2.5_Corrected"]].rename(
        columns={"PM2.5_Corrected": "flaml"}
    )
    oof_f = oof.merge(flaml_oof, on="row_id", how="left")
    test_f = test.merge(flaml_test, on="row_id", how="left")

    # ---- Tables S1 / S2 / S7 (Ensemble notebook cell 2) ----
    s1 = []
    for name, d in [("OOF (63 dev stations)", oof), ("Test (16 unseen stations)", test)]:
        b, a = M(d["obs"], d["raw"]), M(d["obs"], d["corr"])
        s1.append(
            {
                "Dataset": name,
                **{f"{k}_before": v for k, v in b.items()},
                **{f"{k}_after": v for k, v in a.items()},
                "MAE_improve_pct": 100 * (b["MAE"] - a["MAE"]) / b["MAE"],
                "RMSE_improve_pct": 100 * (b["RMSE"] - a["RMSE"]) / b["RMSE"],
                "BIAS_reduce_pct": 100 * (abs(b["BIAS"]) - abs(a["BIAS"])) / abs(b["BIAS"]),
            }
        )
    t_s1 = pd.DataFrame(s1)
    t_s1.to_csv(OUT / "Table_S1_overall_before_after.csv", index=False)

    MODEL_COLS = {
        "Raw MUSICA": "raw",
        "DeepKriging": "DeepKriging_PM2.5_Corrected",
        "CNN": "CNN_PM2.5_Corrected",
        "LightGBM": "LightGBM_PM2.5_Corrected",
        "FLAML AutoML": "flaml",
        "Static 0.50/0.35/0.15": "corr",
    }
    s2 = []
    for ds, d in [("OOF", oof_f), ("Test", test_f)]:
        for model, col in MODEL_COLS.items():
            s2.append({"Dataset": ds, "Model": model, **M(d["obs"], d[col])})
    t_s2 = pd.DataFrame(s2)
    t_s2.to_csv(OUT / "Table_S2_model_comparison.csv", index=False)

    fold_rows = []
    for _, g in oof.groupby("Fold_ID"):
        fold_rows.append(M(g["obs"], g["corr"]))
    t_s7 = pd.DataFrame(fold_rows)[["R2", "MAE", "RMSE", "BIAS"]].agg(["mean", "std"]).T
    t_s7.to_csv(OUT / "Table_S7_fold_stability.csv")

    print("\nTable S1:")
    print(t_s1.round(3).to_string(index=False))
    print("\nTable S2:")
    print(t_s2.round(4).to_string(index=False))
    print("\nTable S7:")
    print(t_s7.round(4).to_string())

    # ---- Fig S3: observation vs prediction density scatter ----
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 10))
    lim = float(np.percentile(oof["obs"], 99.9))
    for row, (ds, d) in enumerate([("OOF", oof), ("Test", test)]):
        for col, (lab, p) in enumerate(
            [("Before (raw MUSICA)", d["raw"]), ("After (Static 0.50/0.35/0.15)", d["corr"])]
        ):
            ax = axes[row, col]
            hb = ax.hexbin(
                d["obs"],
                p,
                gridsize=70,
                bins="log",
                cmap="viridis",
                extent=(0, lim, 0, lim),
                mincnt=1,
            )
            ax.plot([0, lim], [0, lim], "r--", lw=1)
            m = M(d["obs"], p)
            ax.text(
                0.03,
                0.97,
                f"R² = {m['R2']:.3f}\nMAE = {m['MAE']:.2f}\nRMSE = {m['RMSE']:.2f}",
                transform=ax.transAxes,
                va="top",
                fontsize=9,
                bbox=dict(fc="white", alpha=0.85, ec="none"),
            )
            ax.set(
                xlim=(0, lim),
                ylim=(0, lim),
                xlabel=r"Observed PM$_{2.5}$ (µg m$^{-3}$)",
                ylabel=r"Modelled PM$_{2.5}$ (µg m$^{-3}$)",
                title=f"{ds} – {lab}",
            )
            fig.colorbar(hb, ax=ax, label="count (log)")
    fig.tight_layout()
    fig.savefig(OUT / "Fig_S3_density_scatter.png")
    plt.close(fig)

    # ---- Fig S4: normalised Taylor diagram (OOF) ----
    obs_arr = oof_f["obs"].to_numpy(float)
    s_obs = obs_arr.std()
    pts = {}
    for model, col in MODEL_COLS.items():
        p = oof_f[col].to_numpy(float)
        mask = ~np.isnan(p)
        r = float(np.corrcoef(obs_arr[mask], p[mask])[0, 1])
        pts[model] = (p[mask].std() / s_obs, r)

    smax = 1.7
    fig = plt.figure(figsize=(7.5, 7))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_xticks([])
    for corr in [0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        ax.plot([np.arccos(corr)] * 2, [0, smax], color="lightgray", lw=0.6, zorder=0)
        ax.text(np.arccos(corr), smax * 1.04, f"{corr}", fontsize=8, ha="center")
    th = np.linspace(0, np.pi / 2, 600)
    for rms in [0.25, 0.5, 0.75, 1.0]:
        disc = rms ** 2 - np.sin(th) ** 2
        ok = disc >= 0
        for sign in (1, -1):
            s_val = np.cos(th[ok]) + sign * np.sqrt(disc[ok])
            good = (s_val > 0) & (s_val <= smax)
            ax.plot(th[ok][good], s_val[good], ":", color="tan", lw=0.7, zorder=0)

    markers = dict(zip(MODEL_COLS, ["s", "^", "v", "D", "P", "o"]))
    mcolor = {
        "Raw MUSICA": C["raw"],
        "DeepKriging": C["dk"],
        "CNN": C["cnn"],
        "LightGBM": C["lgbm"],
        "FLAML AutoML": C["flaml"],
        "Static 0.50/0.35/0.15": C["ens"],
    }
    for model, (s, r) in pts.items():
        ax.plot(
            np.arccos(r),
            s,
            markers[model],
            ms=10,
            color=mcolor[model],
            mec="k",
            mew=0.4,
            label=f"{model} (r={r:.2f})",
        )
    ax.plot(0, 1, "k*", ms=15, label="Observation (reference)")
    ax.set_rmax(smax)
    ax.set_rticks([0.5, 1.0, 1.5])
    ax.text(np.deg2rad(45), smax * 1.16, "Correlation", ha="center", fontsize=10)
    ax.set_title("Normalised Taylor diagram (OOF)", pad=30)
    ax.legend(loc="upper right", bbox_to_anchor=(1.42, 1.06), fontsize=9)
    fig.savefig(OUT / "Fig_S4_taylor_diagram.png")
    plt.close(fig)

    # ---- Fig S5: residual distribution ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    bins = np.linspace(-40, 25, 130)
    for ax, (ds, d) in zip(axes, [("OOF", oof), ("Test", test)]):
        before = d["raw"] - d["obs"]
        after = d["corr"] - d["obs"]
        ax.hist(before, bins=bins, density=True, alpha=0.55, color=C["before"], label="Before")
        ax.hist(after, bins=bins, density=True, alpha=0.55, color=C["after"], label="After")
        ax.axvline(0, color="k", lw=0.8)
        ax.text(
            0.02,
            0.97,
            f"before: mean {before.mean():.2f}, skew {stats.skew(before):.2f}\n"
            f"after:  mean {after.mean():.2f}, skew {stats.skew(after):.2f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(fc="white", alpha=0.85, ec="none"),
        )
        ax.set(
            title=f"{ds}: residual distribution",
            xlabel=r"Model − Observation (µg m$^{-3}$)",
        )
        ax.legend()
    axes[0].set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(OUT / "Fig_S5_bias_distribution.png")
    plt.close(fig)

    # ---- Fig S6: error by concentration decile ----
    d = oof.copy()
    d["decile"] = pd.qcut(d["obs"], 10, labels=False, duplicates="drop")
    g = d.groupby("decile").apply(
        lambda x: pd.Series(
            {
                "obs_med": x["obs"].median(),
                "MAE_before": np.abs(x["raw"] - x["obs"]).mean(),
                "MAE_after": np.abs(x["corr"] - x["obs"]).mean(),
                "BIAS_before": (x["raw"] - x["obs"]).mean(),
                "BIAS_after": (x["corr"] - x["obs"]).mean(),
            }
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    x = np.arange(len(g))
    labels = [f"{v:.0f}" for v in g["obs_med"]]
    for ax, met in zip(axes, ["MAE", "BIAS"]):
        ax.plot(x, g[f"{met}_before"], "o--", color=C["before"], label="Before")
        ax.plot(x, g[f"{met}_after"], "o-", color=C["after"], label="After")
        if met == "BIAS":
            ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set(
            xlabel=r"Observed-concentration decile (median, µg m$^{-3}$)",
            ylabel=rf"{met} (µg m$^{-3}$)",
            title=f"{met} by concentration decile (OOF)",
        )
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "Fig_S6_concentration_decile.png")
    plt.close(fig)

    # ---- Fig T1: daily series at best / median / worst OOF stations ----
    st_r2 = oof.groupby("station").apply(lambda x: M(x["obs"], x["corr"])["R2"]).sort_values()
    picks = {
        "Worst": st_r2.index[0],
        "Median": st_r2.index[len(st_r2) // 2],
        "Best": st_r2.index[-1],
    }

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, (tag, stn) in zip(axes, picks.items()):
        d = (
            oof[oof["station"] == stn]
            .set_index("datetime")
            .sort_index()[["obs", "raw", "corr"]]
            .resample("D")
            .mean()
        )
        ax.plot(d.index, d["obs"], color=C["obs"], lw=1.2, label="Observed")
        ax.plot(d.index, d["raw"], color=C["raw"], lw=0.9, ls="--", label="Raw MUSICA")
        ax.plot(d.index, d["corr"], color=C["ens"], lw=1.0, label="Static 0.50/0.35/0.15")
        ax.set_ylabel(r"PM$_{2.5}$ (µg m$^{-3}$)")
        ax.set_title(f"{tag}: {stn} (R² = {st_r2[stn]:.2f})", fontsize=10)
    axes[0].legend(ncol=3)
    axes[-1].set_xlabel("Date (daily mean)")
    fig.tight_layout()
    fig.savefig(OUT / "Fig_T1_daily_timeseries.png")
    plt.close(fig)

    # ---- Fig T2: two pollution-episode zooms ----
    net = oof.groupby("datetime")[["obs", "raw", "corr"]].mean()
    daily = net["obs"].resample("D").mean()
    peak1 = daily.idxmax()
    far = daily[np.abs((daily.index - peak1).days) > 30]
    peak2 = far.idxmax()

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5))
    for ax, peak in zip(axes, sorted([peak1, peak2])):
        w = net.loc[peak - pd.Timedelta(days=4) : peak + pd.Timedelta(days=5)]
        ax.plot(w.index, w["obs"], color=C["obs"], lw=1.2, label="Observed")
        ax.plot(w.index, w["raw"], color=C["raw"], ls="--", lw=1.0, label="Raw MUSICA")
        ax.plot(w.index, w["corr"], color=C["ens"], lw=1.1, label="Static 0.50/0.35/0.15")
        ax.set_ylabel(r"Network-mean PM$_{2.5}$ (µg m$^{-3}$)")
        ax.set_title(f"Episode around {peak.date()} (hourly, 63 OOF stations)")
    axes[0].legend(ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "Fig_T2_episode_zoom.png")
    plt.close(fig)

    # ---- Fig T3: diurnal cycle ----
    d = oof.copy()
    d["hour"] = d["datetime"].dt.hour
    g = d.groupby("hour").apply(
        lambda x: pd.Series(
            {
                "MAE_before": np.abs(x["raw"] - x["obs"]).mean(),
                "MAE_after": np.abs(x["corr"] - x["obs"]).mean(),
                "BIAS_before": (x["raw"] - x["obs"]).mean(),
                "BIAS_after": (x["corr"] - x["obs"]).mean(),
            }
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, met in zip(axes, ["MAE", "BIAS"]):
        ax.plot(g.index, g[f"{met}_before"], "o--", color=C["before"], label="Before")
        ax.plot(g.index, g[f"{met}_after"], "o-", color=C["after"], label="After")
        if met == "BIAS":
            ax.axhline(0, color="k", lw=0.8)
        ax.set(
            xlabel="Hour of day (UTC)",
            ylabel=rf"{met} (µg m$^{-3}$)",
            title=f"Diurnal cycle of {met} (OOF)",
            xticks=range(0, 24, 3),
        )
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "Fig_T3_diurnal_cycle.png")
    plt.close(fig)

    # ---- Fig T4: monthly MAE and BIAS ----
    d = oof.copy()
    d["ym"] = d["datetime"].dt.to_period("M").astype(str)
    g = (
        d.groupby("ym")
        .apply(
            lambda x: pd.Series(
                {
                    "MAE_before": np.abs(x["raw"] - x["obs"]).mean(),
                    "MAE_after": np.abs(x["corr"] - x["obs"]).mean(),
                    "BIAS_before": (x["raw"] - x["obs"]).mean(),
                    "BIAS_after": (x["corr"] - x["obs"]).mean(),
                }
            )
        )
        .sort_index()
    )

    x = np.arange(len(g))
    w = 0.38
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7), sharex=True)
    axes[0].bar(x - w / 2, g["MAE_before"], w, color=C["before"], label="Before")
    axes[0].bar(x + w / 2, g["MAE_after"], w, color=C["after"], label="After")
    axes[0].set_ylabel(r"MAE (µg m$^{-3}$)")
    axes[0].set_title("Monthly MAE (OOF)")
    axes[0].legend()
    axes[1].bar(x - w / 2, g["BIAS_before"], w, color=C["before"], label="Before")
    axes[1].bar(x + w / 2, g["BIAS_after"], w, color=C["after"], label="After")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_ylabel(r"BIAS (µg m$^{-3}$)")
    axes[1].set_title("Monthly BIAS (OOF)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(g.index, rotation=45, ha="right")
    axes[1].set_xlabel("Month")
    fig.tight_layout()
    fig.savefig(OUT / "Fig_T4_monthly_metrics.png")
    plt.close(fig)

    # ---- Station-level metrics (P1–P4) ----
    def station_table(df, dataset):
        rows = []
        for stn, g in df.groupby("station"):
            m = M(g["obs"], g["corr"])
            mb = M(g["obs"], g["raw"])
            rows.append(
                {
                    "station": stn,
                    "dataset": dataset,
                    "site_type": g["site_type"].iloc[0],
                    "lat": g["grid_lat"].median(),
                    "lon": g["grid_lon"].median(),
                    "n": len(g),
                    "R2_before": mb["R2"],
                    "R2_after": m["R2"],
                    "MAE_before": mb["MAE"],
                    "MAE_after": m["MAE"],
                    "RMSE_before": mb["RMSE"],
                    "RMSE_after": m["RMSE"],
                    "BIAS_before": mb["BIAS"],
                    "BIAS_after": m["BIAS"],
                    "MAE_improve_pct": 100 * (mb["MAE"] - m["MAE"]) / mb["MAE"],
                }
            )
        return pd.DataFrame(rows)

    st_all = pd.concat(
        [station_table(oof, "OOF"), station_table(test, "Test")], ignore_index=True
    )
    st_all.to_csv(OUT / "Table_P4_station_metrics_with_type.csv", index=False)
    print(st_all.groupby(["dataset", "site_type"]).size().unstack(fill_value=0))

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        HAS_CARTOPY = True
    except Exception:
        HAS_CARTOPY = False
        print(">>> cartopy not available — using plain lon/lat scatter.")

    def uk_map(ax):
        if HAS_CARTOPY:
            ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.7)
            ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.5, ls=":")
            ax.set_extent([-8.5, 2.2, 49.8, 59.2], crs=ccrs.PlateCarree())
            ax.set_xticks(np.arange(-8, 3, 2), crs=ccrs.PlateCarree())
            ax.set_yticks(np.arange(50, 60, 2), crs=ccrs.PlateCarree())
        else:
            ax.set_xlim(-8.5, 2.2)
            ax.set_ylim(49.8, 59.2)
            ax.set_aspect(1.4)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    def plot_station_map(metric, cmap, vlim, fname, title, diverging=False):
        if HAS_CARTOPY:
            fig, ax = plt.subplots(
                figsize=(6.5, 8.5), subplot_kw={"projection": ccrs.PlateCarree()}
            )
            transform = ccrs.PlateCarree()
        else:
            fig, ax = plt.subplots(figsize=(6.5, 8.5))
            transform = None
        uk_map(ax)
        vmin, vmax = vlim
        for ds, marker, size in [("OOF", "o", 55), ("Test", "^", 90)]:
            g = st_all[st_all["dataset"] == ds]
            kw = dict(
                c=g[metric],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                s=size,
                marker=marker,
                edgecolors="k",
                linewidths=0.4,
                label=ds,
                zorder=3,
            )
            if transform is not None:
                kw["transform"] = transform
            sc = ax.scatter(g["lon"], g["lat"], **kw)
        cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.04)
        cb.set_label(title)
        ax.legend(loc="upper left")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(OUT / fname)
        plt.close(fig)

    plot_station_map(
        "R2_after",
        "YlGnBu",
        (0, 1),
        "Fig_P1_station_R2_map.png",
        "Station-level R² after correction",
    )
    bmax = float(np.nanpercentile(st_all["BIAS_after"].abs(), 95))
    plot_station_map(
        "BIAS_after",
        "RdBu_r",
        (-bmax, bmax),
        "Fig_P2_residual_bias_map.png",
        r"Station residual BIAS after correction (µg m$^{-3}$)",
        diverging=True,
    )

    # ---- Fig P3: site-type boxplots ----
    order = ["Urban traffic", "Urban background", "Rural background"]
    oof_st = st_all[st_all["dataset"] == "OOF"].copy()
    oof_st["site_type"] = pd.Categorical(oof_st["site_type"], categories=order, ordered=True)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0))
    for ax, met, ylab in zip(
        axes,
        ["R2_after", "MAE_after", "BIAS_after"],
        [r"R² after", r"MAE after (µg m$^{-3}$)", r"BIAS after (µg m$^{-3}$)"],
    ):
        data = [oof_st.loc[oof_st["site_type"] == t, met].dropna() for t in order]
        labels = [f"{t.replace(' ', chr(10))}\n(n={len(d)})" for t, d in zip(order, data)]
        try:
            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=True)
        except TypeError:
            bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=True)
        for patch, color in zip(bp["boxes"], ["#d62728", "#1f77b4", "#2ca02c"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        if met == "BIAS_after":
            ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel(ylab, labelpad=6)
        ax.tick_params(axis="x", labelsize=8.5)
    fig.suptitle("Performance by (rough) AURN site type – OOF stations", fontsize=12)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.22, wspace=0.38)
    fig.savefig(OUT / "Fig_P3_site_type_boxplots.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary = (
        oof_st.groupby("site_type")[["R2_after", "MAE_after", "RMSE_after", "BIAS_after"]]
        .agg(["mean", "median", "std"])
        .round(3)
    )
    summary.to_csv(OUT / "Table_P3_site_type_summary.csv")

    print("\n>>> Written outputs:")
    written = sorted(p.name for p in OUT.glob("Fig_*.png")) + sorted(
        p.name for p in OUT.glob("Table_*.csv")
    )
    for name in written:
        print(f"    {name}")
    print(f"\nWrote {len(written)} files to {OUT}")


def main():
    print("=" * 80)
    print("STATIC 0.50/0.35/0.15 — export + visualization")
    print("=" * 80)
    export_predictions()
    run_visualization(None, None)


if __name__ == "__main__":
    main()
