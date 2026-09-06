"""
Station-wise hyperparameter grid search for XGBoost bias correction.

A fixed 80,000-row subsample is used to scan max_depth and min_child_weight.
Metrics are computed on held-out stations after applying the predicted bias
to the MUSICA PM2.5 field.
"""

import time
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata
import seaborn as sns

import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

start_total_time = time.time()

df = pd.read_csv('Extracted_features_dataset.csv')

unique_stations = df['station'].unique()
np.random.seed(42)  
train_stations = np.random.choice(unique_stations, size=int(len(unique_stations) * 0.8), replace=False)
test_stations = np.setdiff1d(unique_stations, train_stations)

train_df = df[df['station'].isin(train_stations)].copy()
test_df = df[df['station'].isin(test_stations)].copy()

# Drop leakage columns and standardise predictors
meta_and_leak_cols = ['Unnamed: 0', 'PM2.5 (Hourly measured)', 'Bias', 'datetime', 'station']
feature_cols = [col for col in df.columns if col not in meta_and_leak_cols]

X_train_raw = train_df[feature_cols].values
y_train = train_df['Bias'].values
X_test_raw = test_df[feature_cols].values
y_test = test_df['Bias'].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# Draw a fixed 80,000-row subsample for the hyperparameter scan
np.random.seed(42)
sample_idx = np.random.choice(len(X_train), size=80000, replace=False)
X_train_sample = X_train[sample_idx]
y_train_sample = y_train[sample_idx]


max_depth_range = [4, 6, 9, 12, 14, 16, 20, 30, 40]
min_child_weight_range = [3, 5, 7, 9, 15, 20, 30, 40]

results_list = []

grid_start_time = time.time()  # Start the grid-search timer

for depth in max_depth_range:
    for mcw in min_child_weight_range:
        loop_start = time.time()
        
        # Fit XGBoost with the current hyperparameter pair
        model = xgb.XGBRegressor(
            max_depth=depth,
            min_child_weight=mcw,
            n_estimators=80,      # fixed tree count to keep the scan computationally tractable
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1              # use all available cores
        )
        model.fit(X_train_sample, y_train_sample)
        
        # Evaluate bias-corrected PM2.5 on held-out stations
        pred_bias = model.predict(X_test)
        corrected_pm25 = test_df['PM25'].values - pred_bias
        real_pm25 = test_df['PM2.5 (Hourly measured)'].values
        
        # Compute R2, RMSE, MAE and mean BIAS
        r2 = r2_score(real_pm25, corrected_pm25)
        rmse = np.sqrt(mean_squared_error(real_pm25, corrected_pm25))
        mae = mean_absolute_error(real_pm25, corrected_pm25)
        bias = np.mean(corrected_pm25 - real_pm25)
        
        results_list.append({'max_depth': depth, 'min_child_weight': mcw, 'R2': r2, 'RMSE': rmse, 'MAE': mae, 'BIAS': bias})
        
        loop_end = time.time()
        # Log the trial in a compact, aligned format
        print(f"   [Completed] Depth={depth:<3} | MCW={mcw:<3} | MAE: {mae:.4f} | Time: {loop_end - loop_start:.2f}s")

grid_end_time = time.time()
grid_duration = grid_end_time - grid_start_time

results_df = pd.DataFrame(results_list)

X_dots = results_df['max_depth'].values
Y_dots = results_df['min_child_weight'].values

# Interpolate metrics onto a 200 x 200 visualisation grid
grid_x, grid_y = np.meshgrid(
    np.linspace(min(max_depth_range), max(max_depth_range), 200),
    np.linspace(min(min_child_weight_range), max(min_child_weight_range), 200)
)


results_df.to_csv("XGBoost_AllResults.csv", index=False)

# Cubic interpolation of each metric onto the visualisation grid
grid_r2 = griddata((X_dots, Y_dots), results_df['R2'].values, (grid_x, grid_y), method='cubic')
grid_rmse = griddata((X_dots, Y_dots), results_df['RMSE'].values, (grid_x, grid_y), method='cubic')
grid_mae = griddata((X_dots, Y_dots), results_df['MAE'].values, (grid_x, grid_y), method='cubic')
grid_bias = griddata((X_dots, Y_dots), results_df['BIAS'].values, (grid_x, grid_y), method='cubic')


# Three-dimensional response surfaces
fig = plt.figure(figsize=(24, 10), dpi=200) # high-resolution figure layout
metrics_grids = [grid_r2, grid_mae, grid_rmse, grid_bias]
metrics_names = ['R2', 'MAE', 'RMSE', 'BIAS']

for i, (grid_z, name) in enumerate(zip(metrics_grids, metrics_names)):
    ax = fig.add_subplot(1, 4, i+1, projection='3d')
    
    # Render the interpolated response surface
    surf = ax.plot_surface(
        grid_x, grid_y, grid_z, 
        cmap=cm.viridis,          # perceptually uniform viridis colormap
        linewidth=0.4, 
        antialiased=True, 
        edgecolors='black',       # mesh edges
        rcount=40, ccount=40      # mesh resolution
    )

    ax.view_init(elev=22, azim=-60)
    
    # Axis labels for the XGBoost hyperparameters
    ax.set_title(f'XGBoost Hyperparameter Response ({name})', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Max Depth', fontsize=9, labelpad=8)
    ax.set_ylabel('Min Child Weight', fontsize=9, labelpad=8)
    ax.set_zlabel(name, fontsize=9, labelpad=8)
    
    # Lighten axis grids so the surface remains the focus
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"]['color'] = (0.9, 0.9, 0.9, 0.4)
        
    # Add a colour bar
    fig.colorbar(surf, ax=ax, shrink=0.4, aspect=12, pad=0.08)

plt.tight_layout()
plt.savefig('XGB_3D.png', dpi=300)
plt.close()


# Two-dimensional heatmaps of the same metrics
fig = plt.figure(figsize=(24, 10), dpi=200)
cmaps = [cm.YlOrRd, cm.YlOrRd, cm.YlOrRd, cm.RdBu_r]
for i, (grid_z, name, cmap) in enumerate(zip(metrics_grids, metrics_names, cmaps)):
    ax = fig.add_subplot(1, 4, i+1) # planar heatmap; no 3-D projection
    
    mesh = ax.pcolormesh(
        grid_x, grid_y, grid_z, 
        cmap=cmap, 
        shading='auto',
        edgecolors='none'
    )
    
    # Invert the y-axis so larger hyperparameter values appear at the top
    ax.invert_yaxis()  # larger min_child_weight at the top
    
    # Axis labels and layout for the heatmap
    ax.set_title(f'XGBoost Hyperparameter Tuning Heatmap ({name})', fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel('Max Depth', fontsize=10, labelpad=10)
    ax.set_ylabel('Min Child Weight', fontsize=10, labelpad=10)
    
    # Rotate tick labels to avoid overlap
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    
    # Add a colour bar
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.05)
    cbar.set_label(name, fontsize=9)

plt.tight_layout()

# Save a high-resolution figure and close the canvas
plt.savefig('XGB_heatmap.png', dpi=300)
plt.close()