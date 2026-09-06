# PM2.5_bias_correction_with_ml
Data and research code for the study on PM2.5 bias correction based on machine learning and spatial extrapolation

# UK PM2.5 bias-correction workflow

Code for MUSICAv0 PM2.5 bias correction against AURN, from data preparation through adaptive ensembles. Shared inputs live in `Data/`. Derived tables and splits are reused by every later step; do not recreate them.

## Layout

```
ERP_ENV/
  Data/                          # shared inputs (MUSICA_allvariables_CAMSAIR_CAMSv51_QFED_vsl.parquet)
  01_fetch_data/                 # optional raw retrieval
  02_data_preprocessing/         # cleaning, matching, feature table
  03_hyperparameter_tuning/      # two-stage HPO
  04_base_models_training/       # 63/16 split + 5-fold OOF
  05_ensemble_model_training/    # static weighted ensemble
  06_adaptive_model_training/    # bin gating and neural MoE
```

## Data

Place the fused MUSICA–AURN–emission table here:

```
Data/MUSICA_allvariables_CAMSAIR_CAMSv51_QFED_vsl.parquet
```

This parquet is stored with Git LFS (about 355 MB). Clone with Git LFS installed, or run `git lfs pull` afterwards if you only see a small pointer file.

Preprocessing writes the modelling table used downstream:

```
Data/Extracted_features_dataset.csv            # 34 predictors + labels
Data/Extracted_features_dataset_withgeo.csv    # same table plus grid_lat, grid_lon (DeepKriging / MLP / ensemble)
```

Point each script’s `DATA`, `SPLIT_FILE`, and `FOLD_FILE` to these files (or copy them into the working directory). Keep the filenames above.

`01_fetch_data/` rebuilds MUSICA from 'MUSICA_allvariables_CAMSAIR_CAMSv51_QFED_vsl' and  fetches AURN from obsaq package (https://github.com/envdes/obsaq) . 

## Station splits (run XGBoost first)

There is no standalone split script. **`04_base_models_training/ML_XGB_final.py` creates the two CSVs used by the rest of the study:**

| File | Role |
| --- | --- |
| `Data/station_split.csv` | 79 stations → 63 development / 16 held-out test (`seed=42`) |
| `Data/development_fold_split.csv` | station-grouped 5-fold GroupKFold on the 63 development stations |

After XGBoost has written them:

1. Copy both CSVs into `Data/` (and into `results/` if a script reads them there).
2. Run every later model **with those files already present**. Scripts will reuse them and will not redraw the split.
3. Do not delete or regenerate them. CNN/MLP tuning can write `station_split.csv` if it is missing; that is not the official fold file.

## Experimental order

**0. Optional fetch** .

- `01_fetch_data/fetch_aurn_data.ipynb`
- `01_fetch_data/export_musica.ipynb`

**1. Preprocessing**

- `02_data_preprocessing/data_preprocessing_EDA.ipynb`  
  Converts MUSICA PM2.5 to µg m⁻³, aggregates emissions, writes the extracted feature tables.

**2. Hyperparameter tuning** (80,000-row subsample; MAE plus R² / RMSE / BIAS)

- `03_hyperparameter_tuning/RF_opti_hyperpara.py`
- `03_hyperparameter_tuning/XGBoost_opti_hyperpara.py`
- `03_hyperparameter_tuning/LightGBM_opti_hyperpara.py`
- `03_hyperparameter_tuning/CNN_opti_hyperpara.py`
- `03_hyperparameter_tuning/MLP_opti_hyperpara.py`
- `03_hyperparameter_tuning/opti_heatmap.ipynb` (figures)

DeepKriging uses the selected MLP width/depth. Tuning does not replace the XGBoost station/fold CSVs.

**3. Base-model training** — **XGBoost first**, then the others.

```text
04_base_models_training/ML_XGB_final.py      # creates the two split CSVs
04_base_models_training/ML_RF_final.py       # reuse splits
04_base_models_training/ML_LightGBM_final.py
04_base_models_training/CNN_Final.py
04_base_models_training/MLP_final.py
04_base_models_training/Deepkriging_final.py
```

Each script: 5-fold station OOF on development stations, retrain on all 63, evaluate the 16 test stations once.

**4. Static ensemble** (DeepKriging + CNN + LightGBM)

```text
05_ensemble_model_training/Deep_CNN_LGBM.py   # 0.33 / 0.33 / 0.33
05_ensemble_model_training/ensemble_weight_search.ipynb
05_ensemble_model_training/Static503515_export_and_visualization.py   # 0.50 / 0.35 / 0.15
```

**5. Adaptive weighting** (frozen experts; reuse the same OOF/test tables and fold CSV)

```text
06_adaptive_model_training/MoE_Gating_DK_CNN_LGBM_unit_fixed.py
06_adaptive_model_training/Piecewise_PM25Bin_Gating_DK_CNN_LGBM_unit_fixed.py
06_adaptive_model_training/HardBins_Gating_Ensemble403525.py
06_adaptive_model_training/PM25Bin_Gating_FiveVariants_unit_fixed.py
06_adaptive_model_training/Ensemble403525_Static_and_Gating_V1V5.py
```

Run the last three from the same folder as `Piecewise_PM25Bin_Gating_DK_CNN_LGBM_unit_fixed.py` (relative imports).

## Notes

- Protocol: development-only tuning and selection; the 16 test stations are scored once.
- MUSICA `PM25` in the parquet is kg m⁻³; convert with ×10⁹ before modelling.
- Adaptive scripts apply the same unit correction to historical expert prediction files.
- Keep `Data/station_split.csv` and `Data/development_fold_split.csv` under version control or backup; they define the paper splits.
