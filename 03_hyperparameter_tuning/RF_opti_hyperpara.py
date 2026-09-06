"""
Station-wise hyperparameter grid search for Random Forest bias correction.

A fixed 80,000-row subsample is used to scan max_depth and min_samples_leaf.
Performance is reported on held-out stations using R2, MAE, RMSE and mean
BIAS of the bias-corrected PM2.5 field.
"""

import time
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

from scipy.interpolate import griddata

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

start_total_time = time.time()


data_path = ("Extracted_features_dataset.csv")

df = pd.read_csv(data_path)

print(df.info())
print(f"Sample count:{len(df)}")
print(f"Site num:{df['station'].nunique()}")

unique_stations = df['station'].unique()
np.random.seed(42)  
train_stations = np.random.choice(unique_stations, size=int(len(unique_stations) * 0.8), replace=False)
test_stations = np.setdiff1d(unique_stations, train_stations)

train_df = df[df['station'].isin(train_stations)].copy()
test_df = df[df['station'].isin(test_stations)].copy()


meta_and_leak_cols = ['Unnamed: 0', 'PM2.5 (Hourly measured)', 'Bias', 'datetime', 'station']
feature_cols = [col for col in df.columns if col not in meta_and_leak_cols]

X_train = train_df[feature_cols].values
y_train = train_df['Bias'].values
X_test = test_df[feature_cols].values
y_test = test_df['Bias'].values


np.random.seed(42)
sample_idx = np.random.choice(len(X_train), size=80000, replace=False)
X_train_sample = X_train[sample_idx]
y_train_sample = y_train[sample_idx]


max_depth_range = [5, 10, 15, 20, 25, 30, 40, 50]

min_samples_leaf_range = [1, 2, 5, 10, 20, 50, 100, 200]

rf_results = []

grid_start_time = time.time()


# Grid search over max_depth and min_samples_leaf
for max_depth in max_depth_range:

    for min_samples_leaf in min_samples_leaf_range:

        loop_start_time = time.time()

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=1.0,
            criterion="squared_error",
            bootstrap=True,
            random_state=42,
            n_jobs=12
        )

        model.fit(X_train_sample, y_train_sample)
        pred_bias = model.predict(X_test)
        corrected_pm25 = (test_df["PM25"].values - pred_bias)
        real_pm25 = test_df["PM2.5 (Hourly measured)"].values

        r2 = r2_score(real_pm25, corrected_pm25)
        rmse = np.sqrt(mean_squared_error(real_pm25, corrected_pm25))
        mae = mean_absolute_error(real_pm25, corrected_pm25)
        bias = np.mean(corrected_pm25 - real_pm25)

        rf_results.append({
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
            "BIAS": bias
        })

        loop_time = time.time() - loop_start_time

        print(
            f"[RF] "
            f"MaxDepth={max_depth:<3} | "
            f"MinLeaf={min_samples_leaf:<3} | "
            f"R2={r2:.4f} | "
            f"MAE={mae:.4f} | "
            f"RMSE={rmse:.4f} | "
            f"BIAS={bias:.4f} | "
            f"耗时={loop_time:.2f}秒"
        )

rf_results_df = pd.DataFrame(rf_results)
result_path = ("RandomForest_AllResults.csv")
rf_results_df.to_csv(result_path, index=False)
print(f"Tuning results save to:{result_path}")


X_dots = rf_results_df['max_depth'].values
Y_dots = rf_results_df['min_samples_leaf'].values

# Interpolate metrics onto a 200 x 200 visualisation grid
grid_x, grid_y = np.meshgrid(
    np.linspace(min(rf_results_df), max(rf_results_df), 200),
    np.linspace(min(rf_results_df), max(rf_results_df), 200)
)

grid_r2 = griddata((X_dots, Y_dots), rf_results_df['R2'].values, (grid_x, grid_y), method='cubic')
grid_rmse = griddata((X_dots, Y_dots), rf_results_df['RMSE'].values, (grid_x, grid_y), method='cubic')
grid_mae = griddata((X_dots, Y_dots), rf_results_df['MAE'].values, (grid_x, grid_y), method='cubic')
grid_bias = griddata((X_dots, Y_dots), rf_results_df['BIAS'].values, (grid_x, grid_y), method='cubic')


# Three-dimensional response surfaces
fig3d = plt.figure(figsize=(24, 10), dpi=200)
metrics_grids = [grid_r2, grid_mae, grid_rmse, grid_bias]
metrics_names = ['R2', 'MAE', 'RMSE', 'BIAS']

for i, (grid_z, name) in enumerate(zip(metrics_grids, metrics_names)):
    ax = fig3d.add_subplot(1, 4, i+1, projection='3d')
    
    surf = ax.plot_surface(
        grid_x, grid_y, grid_z, 
        cmap=cm.viridis, 
        linewidth=0.4, 
        antialiased=True, 
        edgecolors='black',
        rcount=40, ccount=40
    )
    
    ax.view_init(elev=22, azim=-60)
    ax.set_title(f'Random Forest Hyperparameter Response ({name})', fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('max_depth', fontsize=9, labelpad=8)
    ax.set_ylabel('min_samples_leaf', fontsize=9, labelpad=8)
    ax.set_zlabel(name, fontsize=9, labelpad=8)
    
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"]['color'] = (0.9, 0.9, 0.9, 0.4)
        
    fig3d.colorbar(surf, ax=ax, shrink=0.4, aspect=12, pad=0.08)

plt.tight_layout()
plt.savefig('RF_3D.png', dpi=300)
plt.close()