"""
Station-wise hyperparameter grid search for LightGBM bias correction.

A fixed 80,000-row subsample is used to scan num_leaves and min_data_in_leaf
under an L1 objective. Metrics are computed on held-out stations.
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

import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

start_total_time = time.time()

df = pd.read_csv('Extracted_features_dataset.csv')

print(df.info())

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

print(f">>> Data prepared. Sample dataset ({X_train_sample.shape[0]} ) scaning...")

num_leaves_range = [20, 40, 60, 100, 120, 150, 200, 300, 400]
min_data_range = [5, 10, 20, 50, 100, 200, 300, 400]

lgb_results = []
grid_start_time = time.time()

for num_leaves in num_leaves_range:
    for min_data in min_data_range:
        loop_start = time.time()
        
        # Fit LightGBM with an L1 (MAE) objective
        model = lgb.LGBMRegressor(
            objective='regression_l1',
            num_leaves=num_leaves,
            min_data_in_leaf=min_data,
            n_estimators=80,          # fixed boosting rounds during the scan
            random_state=42,
            n_jobs=12,                 # use 12 parallel workers
            verbose=-1
        )
        model.fit(X_train_sample, y_train_sample)
        
        # Evaluate spatial generalisation on held-out stations
        pred_bias = model.predict(X_test)
        corrected_pm25 = test_df['PM25'].values - pred_bias
        real_pm25 = test_df['PM2.5 (Hourly measured)'].values
        
        # Compute R2, RMSE, MAE and mean BIAS
        r2 = r2_score(real_pm25, corrected_pm25)
        rmse = np.sqrt(mean_squared_error(real_pm25, corrected_pm25))
        mae = mean_absolute_error(real_pm25, corrected_pm25)
        bias = np.mean(corrected_pm25 - real_pm25)
        
        lgb_results.append({
            'num_leaves': num_leaves, 
            'min_data_in_leaf': min_data, 
            'R2': r2, 
            'RMSE': rmse, 
            'MAE': mae, 
            'BIAS': bias
        })
        
        loop_end = time.time()
        print(f"   [LGBM] Leaves={num_leaves:<3} | MinData={min_data:<3} | MAE: {mae:.4f} | BIAS: {bias:.4f} | Time: {loop_end - loop_start:.2f}s")

grid_end_time = time.time()
lgb_results_df = pd.DataFrame(lgb_results)

X_dots = lgb_results_df['num_leaves'].values
Y_dots = lgb_results_df['min_data_in_leaf'].values

# Interpolate metrics onto a 200 x 200 visualisation grid
grid_x, grid_y = np.meshgrid(
    np.linspace(min(num_leaves_range), max(num_leaves_range), 200),
    np.linspace(min(min_data_range), max(min_data_range), 200)
)

lgb_results_df.to_csv("LightGBM_GridSearch_AllResults.csv", index=False)
# Cubic interpolation of each metric onto the visualisation grid
grid_r2 = griddata((X_dots, Y_dots), lgb_results_df['R2'].values, (grid_x, grid_y), method='cubic')
grid_rmse = griddata((X_dots, Y_dots), lgb_results_df['RMSE'].values, (grid_x, grid_y), method='cubic')
grid_mae = griddata((X_dots, Y_dots), lgb_results_df['MAE'].values, (grid_x, grid_y), method='cubic')
grid_bias = griddata((X_dots, Y_dots), lgb_results_df['BIAS'].values, (grid_x, grid_y), method='cubic')

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
    ax.set_title(f'LightGBM Hyperparameter Response ({name})', fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('num_leaves', fontsize=9, labelpad=8)
    ax.set_ylabel('min_data_in_leaf', fontsize=9, labelpad=8)
    ax.set_zlabel(name, fontsize=9, labelpad=8)
    
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"]['color'] = (0.9, 0.9, 0.9, 0.4)
        
    fig3d.colorbar(surf, ax=ax, shrink=0.4, aspect=12, pad=0.08)

plt.tight_layout()
plt.savefig('LGBM_3D.png', dpi=300)
plt.close()


fig = plt.figure(figsize=(24, 10), dpi=200)
cmaps = [cm.YlOrRd, cm.YlOrRd, cm.YlOrRd, cm.RdBu_r]
for i, (grid_z, name, cmap) in enumerate(zip(metrics_grids, metrics_names, cmaps)):
    ax = fig.add_subplot(1, 4, i+1) 
    
    mesh = ax.pcolormesh(
        grid_x, grid_y, grid_z, 
        cmap=cmap, 
        shading='auto',
        edgecolors='none'
    )
    
    # Invert the y-axis so larger hyperparameter values appear at the top
    ax.invert_yaxis()  # larger min_child_weight at the top
    
    # Axis labels and layout for the heatmap
    ax.set_title(f'LightGBM Hyperparameter Tuning Heatmap ({name})', fontsize=12, fontweight='bold', pad=15)
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
plt.savefig('LGBM_heatmap.png', dpi=300)
plt.close()
