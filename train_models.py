"""
MSIS 522 HW1 – Model Training Script
=====================================
Predicts mortgage rate_spread (interest rate premium above 10-yr Treasury).
Run once:  python train_models.py
Outputs: models/*.joblib, models/mlp.pt, models/results.json, figures/*.png
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
import shap
import joblib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_PATH   = 'C:/Users/nancy/Downloads/typical_loans (1).csv'
MODELS_DIR  = 'models'
FIGURES_DIR = 'figures'
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Feature Definitions ────────────────────────────────────────────────────
NUM_FEATURES = [
    'Credit Score',
    'Mortgage Insurance Percentage (MI %)',
    'Original Combined Loan-to-Value (CLTV)',
    'Original Debt-to-Income (DTI) Ratio',
    'Original UPB',
    'Original Loan-to-Value (LTV)',
    'Number of Borrowers',
]
CAT_FEATURES = [
    'Channel',
    'Loan Purpose',
    'First Time Homebuyer Flag',
    'Property State',
]
TARGET = 'rate_spread'

# ── Helpers ────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {'MAE': round(mae, 4), 'RMSE': round(rmse, 4), 'R2': round(r2, 4)}

def save_fig(name):
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/{name}', dpi=130, bbox_inches='tight')
    plt.close('all')

def pred_vs_actual_plot(y_true, y_pred, title, fname):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.08, s=4, color='steelblue')
    lo = min(float(y_true.min()), float(y_pred.min()))
    hi = max(float(y_true.max()), float(y_pred.max()))
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='Perfect Fit')
    ax.set_xlabel('Actual Rate Spread (pp)', fontsize=11)
    ax.set_ylabel('Predicted Rate Spread (pp)', fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend()
    save_fig(fname)

# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD & CLEAN DATA
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Loading data...")
df_raw = pd.read_csv(DATA_PATH)
df = df_raw[NUM_FEATURES + CAT_FEATURES + [TARGET]].copy()

# Fix rare '9' code in First Time Homebuyer Flag → treat as 'N'
df['First Time Homebuyer Flag'] = df['First Time Homebuyer Flag'].replace('9', 'N')

print(f"  Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}")
print(f"  Missing values: {df.isnull().sum().sum()}")
print(f"  Target (rate_spread): mean={df[TARGET].mean():.3f}, "
      f"std={df[TARGET].std():.3f}, min={df[TARGET].min():.3f}, max={df[TARGET].max():.3f}")

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ══════════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════════════
print("\nBuilding preprocessor...")
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), NUM_FEATURES),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_FEATURES),
], remainder='drop')

preprocessor.fit(X_train)
joblib.dump(preprocessor, f'{MODELS_DIR}/preprocessor.joblib')

X_train_pp = preprocessor.transform(X_train)
X_test_pp  = preprocessor.transform(X_test)

ohe = preprocessor.named_transformers_['cat']
cat_feat_names = ohe.get_feature_names_out(CAT_FEATURES).tolist()
feature_names  = NUM_FEATURES + cat_feat_names
np.save(f'{MODELS_DIR}/feature_names.npy', np.array(feature_names))
print(f"  Preprocessed feature count: {X_train_pp.shape[1]}")

# ══════════════════════════════════════════════════════════════════════════
# 3. EDA FIGURES
# ══════════════════════════════════════════════════════════════════════════
print("\nGenerating EDA figures...")
sns.set_style('whitegrid')

# 3a. Target distribution
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(y, bins=45, color='steelblue', edgecolor='white', alpha=0.85)
ax.axvline(y.mean(),   color='red',    linestyle='--', lw=1.5, label=f'Mean: {y.mean():.3f}')
ax.axvline(y.median(), color='orange', linestyle='--', lw=1.5, label=f'Median: {y.median():.3f}')
ax.set_xlabel('Rate Spread (percentage points)', fontsize=12)
ax.set_ylabel('Number of Loans', fontsize=12)
ax.set_title('Distribution of Target: Rate Spread', fontsize=14)
ax.legend()
save_fig('fig_target.png')

# 3b. Credit Score vs Rate Spread
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df['Credit Score'], y, alpha=0.04, s=4, color='steelblue')
ax.set_xlabel('Credit Score', fontsize=12)
ax.set_ylabel('Rate Spread (pp)', fontsize=12)
ax.set_title('Credit Score vs Rate Spread', fontsize=14)
save_fig('fig_credit_score.png')

# 3c. DTI vs Rate Spread
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df['Original Debt-to-Income (DTI) Ratio'], y, alpha=0.04, s=4, color='coral')
ax.set_xlabel('Debt-to-Income Ratio (%)', fontsize=12)
ax.set_ylabel('Rate Spread (pp)', fontsize=12)
ax.set_title('DTI Ratio vs Rate Spread', fontsize=14)
save_fig('fig_dti.png')

# 3d. LTV vs Rate Spread
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df['Original Loan-to-Value (LTV)'], y, alpha=0.04, s=4, color='seagreen')
ax.set_xlabel('Loan-to-Value Ratio (%)', fontsize=12)
ax.set_ylabel('Rate Spread (pp)', fontsize=12)
ax.set_title('LTV Ratio vs Rate Spread', fontsize=14)
save_fig('fig_ltv.png')

# 3e. Channel vs Rate Spread
fig, ax = plt.subplots(figsize=(7, 5))
df_plot = df.copy()
df_plot['Channel_label'] = df_plot['Channel'].map({'R': 'Retail', 'C': 'Correspondent', 'B': 'Broker'})
sns.boxplot(data=df_plot, x='Channel_label', y=TARGET, ax=ax,
            palette='Set2', order=['Retail', 'Correspondent', 'Broker'])
ax.set_xlabel('Origination Channel', fontsize=12)
ax.set_ylabel('Rate Spread (pp)', fontsize=12)
ax.set_title('Rate Spread by Origination Channel', fontsize=14)
save_fig('fig_channel.png')

# 3f. Loan Purpose vs Rate Spread
fig, ax = plt.subplots(figsize=(7, 5))
df_plot['Purpose_label'] = df_plot['Loan Purpose'].map(
    {'P': 'Purchase', 'C': 'Cash-Out Refi', 'N': 'No Cash-Out Refi'})
sns.boxplot(data=df_plot, x='Purpose_label', y=TARGET, ax=ax, palette='Set3')
ax.set_xlabel('Loan Purpose', fontsize=12)
ax.set_ylabel('Rate Spread (pp)', fontsize=12)
ax.set_title('Rate Spread by Loan Purpose', fontsize=14)
save_fig('fig_loan_purpose.png')

# 3g. MI % vs Rate Spread
fig, ax = plt.subplots(figsize=(8, 5))
mi_vals = sorted(df['Mortgage Insurance Percentage (MI %)'].unique())
df['MI_cat'] = df['Mortgage Insurance Percentage (MI %)'].apply(lambda x: f'{int(x)}%')
mi_order = [f'{int(m)}%' for m in mi_vals]
sns.boxplot(data=df, x='MI_cat', y=TARGET, order=mi_order, ax=ax, palette='Blues')
ax.set_xlabel('Mortgage Insurance Percentage', fontsize=12)
ax.set_ylabel('Rate Spread (pp)', fontsize=12)
ax.set_title('Rate Spread by MI Percentage', fontsize=14)
save_fig('fig_mi.png')

# 3h. Correlation heatmap
fig, ax = plt.subplots(figsize=(9, 7))
corr_cols = NUM_FEATURES + [TARGET]
corr_mat  = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
sns.heatmap(corr_mat, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, linewidths=0.5, annot_kws={'size': 9})
ax.set_title('Correlation Heatmap – Numerical Features', fontsize=14)
save_fig('fig_corr.png')

print("  EDA figures saved.")

# ══════════════════════════════════════════════════════════════════════════
# 4. LINEAR REGRESSION BASELINE
# ══════════════════════════════════════════════════════════════════════════
print("\n[1/5] Training Linear Regression...")
lr = LinearRegression()
lr.fit(X_train_pp, y_train)
y_pred_lr = lr.predict(X_test_pp)

results = {}
results['Linear Regression'] = compute_metrics(y_test, y_pred_lr)
pred_vs_actual_plot(y_test, y_pred_lr, 'Linear Regression: Predicted vs Actual', 'fig_lr_pred.png')
joblib.dump(lr, f'{MODELS_DIR}/lr.joblib')
print(f"  {results['Linear Regression']}")

# ══════════════════════════════════════════════════════════════════════════
# 5. DECISION TREE (GridSearchCV)
# ══════════════════════════════════════════════════════════════════════════
print("\n[2/5] Training Decision Tree (GridSearchCV 5-fold)...")
dt_param_grid = {
    'max_depth':        [3, 5, 7, 10],
    'min_samples_leaf': [5, 10, 20, 50],
}
dt_gs = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    dt_param_grid,
    cv=5, scoring='neg_mean_squared_error',
    n_jobs=-1, verbose=0
)
dt_gs.fit(X_train_pp, y_train)
best_dt    = dt_gs.best_estimator_
y_pred_dt  = best_dt.predict(X_test_pp)

results['Decision Tree'] = {
    **compute_metrics(y_test, y_pred_dt),
    'best_params': dt_gs.best_params_
}
pred_vs_actual_plot(y_test, y_pred_dt, 'Decision Tree: Predicted vs Actual', 'fig_dt_pred.png')
joblib.dump(best_dt, f'{MODELS_DIR}/dt.joblib')

# Tree visualization (top 3 levels)
fig, ax = plt.subplots(figsize=(22, 8))
plot_tree(best_dt, max_depth=3, feature_names=feature_names,
          filled=True, rounded=True, fontsize=7, ax=ax)
ax.set_title(f"Decision Tree – Best Params: {dt_gs.best_params_}", fontsize=12)
save_fig('fig_dt_tree.png')
print(f"  Best params: {dt_gs.best_params_}")
print(f"  {results['Decision Tree']}")

# ══════════════════════════════════════════════════════════════════════════
# 6. RANDOM FOREST (GridSearchCV)
# ══════════════════════════════════════════════════════════════════════════
print("\n[3/5] Training Random Forest (GridSearchCV 5-fold)...")
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth':    [3, 5, 8],
}
rf_gs = GridSearchCV(
    RandomForestRegressor(random_state=42),
    rf_param_grid,
    cv=5, scoring='neg_mean_squared_error',
    n_jobs=-1, verbose=0
)
rf_gs.fit(X_train_pp, y_train)
best_rf   = rf_gs.best_estimator_
y_pred_rf = best_rf.predict(X_test_pp)

results['Random Forest'] = {
    **compute_metrics(y_test, y_pred_rf),
    'best_params': rf_gs.best_params_
}
pred_vs_actual_plot(y_test, y_pred_rf, 'Random Forest: Predicted vs Actual', 'fig_rf_pred.png')
joblib.dump(best_rf, f'{MODELS_DIR}/rf.joblib')
print(f"  Best params: {rf_gs.best_params_}")
print(f"  {results['Random Forest']}")

# ══════════════════════════════════════════════════════════════════════════
# 7. XGBOOST (GridSearchCV)
# ══════════════════════════════════════════════════════════════════════════
print("\n[4/5] Training XGBoost (GridSearchCV 5-fold)...")
xgb_param_grid = {
    'n_estimators':  [50, 100, 200],
    'max_depth':     [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
}
xgb_gs = GridSearchCV(
    xgb.XGBRegressor(random_state=42, tree_method='hist', verbosity=0),
    xgb_param_grid,
    cv=5, scoring='neg_mean_squared_error',
    n_jobs=-1, verbose=0
)
xgb_gs.fit(X_train_pp, y_train)
best_xgb   = xgb_gs.best_estimator_
y_pred_xgb = best_xgb.predict(X_test_pp)

results['XGBoost'] = {
    **compute_metrics(y_test, y_pred_xgb),
    'best_params': xgb_gs.best_params_
}
pred_vs_actual_plot(y_test, y_pred_xgb, 'XGBoost: Predicted vs Actual', 'fig_xgb_pred.png')
joblib.dump(best_xgb, f'{MODELS_DIR}/xgb.joblib')
print(f"  Best params: {xgb_gs.best_params_}")
print(f"  {results['XGBoost']}")

# ══════════════════════════════════════════════════════════════════════════
# 8. MLP – PyTorch
# ══════════════════════════════════════════════════════════════════════════
print("\n[5/5] Training MLP (PyTorch, 100 epochs)...")

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

input_dim = X_train_pp.shape[1]
device    = 'cuda' if torch.cuda.is_available() else 'cpu'

X_tr_t = torch.tensor(X_train_pp, dtype=torch.float32)
y_tr_t = torch.tensor(y_train.values, dtype=torch.float32)
X_te_t = torch.tensor(X_test_pp,  dtype=torch.float32)
y_te_t = torch.tensor(y_test.values,  dtype=torch.float32)

train_ds = TensorDataset(X_tr_t, y_tr_t)
train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)

mlp       = MLP(input_dim).to(device)
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
criterion = nn.MSELoss()

history = {'loss': [], 'val_loss': []}
EPOCHS  = 100

for epoch in range(EPOCHS):
    mlp.train()
    batch_losses = []
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        pred = mlp(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())

    mlp.eval()
    with torch.no_grad():
        val_pred = mlp(X_te_t.to(device))
        val_loss = criterion(val_pred, y_te_t.to(device)).item()

    history['loss'].append(float(np.mean(batch_losses)))
    history['val_loss'].append(float(val_loss))

    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS} | "
              f"train_loss={history['loss'][-1]:.5f} | val_loss={val_loss:.5f}")

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history['loss'],     label='Train', color='steelblue')
axes[0].plot(history['val_loss'], label='Val',   color='coral')
axes[0].set_title('MLP Training History – MSE Loss')
axes[0].set_xlabel('Epoch');  axes[0].set_ylabel('MSE')
axes[0].legend()

axes[1].plot([np.sqrt(v) for v in history['loss']],     label='Train', color='steelblue')
axes[1].plot([np.sqrt(v) for v in history['val_loss']], label='Val',   color='coral')
axes[1].set_title('MLP Training History – RMSE')
axes[1].set_xlabel('Epoch');  axes[1].set_ylabel('RMSE')
axes[1].legend()
save_fig('fig_mlp_history.png')

mlp.eval()
with torch.no_grad():
    y_pred_mlp = mlp(X_te_t.to(device)).cpu().numpy()

results['MLP (PyTorch)'] = compute_metrics(y_test, y_pred_mlp)
pred_vs_actual_plot(y_test, y_pred_mlp, 'MLP: Predicted vs Actual', 'fig_mlp_pred.png')
torch.save({'state_dict': mlp.state_dict(), 'input_dim': input_dim}, f'{MODELS_DIR}/mlp.pt')
print(f"  {results['MLP (PyTorch)']}")

# ══════════════════════════════════════════════════════════════════════════
# 9. MODEL COMPARISON CHART
# ══════════════════════════════════════════════════════════════════════════
print("\nGenerating model comparison chart...")
model_names = list(results.keys())
rmse_vals   = [results[m]['RMSE'] for m in model_names]
colors      = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(model_names, rmse_vals, color=colors, edgecolor='white', linewidth=0.8)
ax.set_ylabel('RMSE (lower is better)', fontsize=12)
ax.set_title('Model Comparison – Test-Set RMSE', fontsize=14)
ax.set_ylim(0, max(rmse_vals) * 1.2)
for bar, val in zip(bars, rmse_vals):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(rmse_vals) * 0.01,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.xticks(rotation=15, ha='right')
save_fig('fig_model_comparison.png')

# ══════════════════════════════════════════════════════════════════════════
# 10. SHAP ANALYSIS (on XGBoost, best tree model)
# ══════════════════════════════════════════════════════════════════════════
print("\nRunning SHAP analysis on XGBoost...")
shap_explainer = shap.TreeExplainer(best_xgb)

# Use a random sample of 2000 test rows for efficiency
rng  = np.random.default_rng(42)
idx  = rng.choice(len(X_test_pp), size=min(2000, len(X_test_pp)), replace=False)
X_shap = X_test_pp[idx]
shap_values = shap_explainer.shap_values(X_shap)

# Summary (beeswarm) plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_shap, feature_names=feature_names, show=False)
plt.title('SHAP Summary Plot – XGBoost', fontsize=14)
save_fig('fig_shap_summary.png')

# Feature importance bar plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_shap, feature_names=feature_names,
                  plot_type='bar', show=False)
plt.title('SHAP Mean |SHAP| – Feature Importance', fontsize=14)
save_fig('fig_shap_bar.png')

# Waterfall plot for highest-predicted-spread loan
high_idx   = int(np.argmax(best_xgb.predict(X_test_pp)))
expl_high  = shap_explainer(X_test_pp[high_idx : high_idx + 1])
plt.figure(figsize=(10, 7))
shap.waterfall_plot(expl_high[0], show=False)
plt.title('SHAP Waterfall – Loan with Highest Predicted Spread', fontsize=13)
save_fig('fig_shap_waterfall.png')

print("  SHAP figures saved.")

# ══════════════════════════════════════════════════════════════════════════
# 11. SAVE RESULTS JSON
# ══════════════════════════════════════════════════════════════════════════
with open(f'{MODELS_DIR}/results.json', 'w') as f:
    json.dump(results, f, indent=2)

# ── Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"{'Model':<22} {'MAE':>8} {'RMSE':>8} {'R²':>8}  Best Params")
print("-" * 60)
for name, m in results.items():
    params = str(m.get('best_params', 'N/A'))
    print(f"{name:<22} {m['MAE']:>8.4f} {m['RMSE']:>8.4f} {m['R2']:>8.4f}  {params}")
best_model = min(results, key=lambda k: results[k]['RMSE'])
print(f"\nBest model: {best_model} (RMSE={results[best_model]['RMSE']:.4f})")
print(f"\nModels saved to '{MODELS_DIR}/'")
print(f"Figures saved to '{FIGURES_DIR}/'")
