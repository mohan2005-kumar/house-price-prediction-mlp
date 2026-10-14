"""
evaluate.py
Evaluates trained Embedded MLP model performance on unseen test data
using standard regression metrics: MAE, RMSE, and R2 score (as highlighted in resume).
Generates diagnostic plots: Actual vs Predicted, Residuals Distribution, and Training Curves.
"""

import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless backend
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error

from preprocess import HousingDataPreprocessor, CONTINUOUS_COLS
from model import EmbeddedMLP

def evaluate_model():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, 'models')
    reports_dir = os.path.join(project_root, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1. Load artifacts
    meta_path = os.path.join(models_dir, 'metadata.json')
    prep_path = os.path.join(models_dir, 'preprocessor.pkl')
    model_path = os.path.join(models_dir, 'best_model.pt')
    test_path = os.path.join(models_dir, 'test_split.npz')
    history_path = os.path.join(models_dir, 'history.json')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run train.py first.")
        
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
        
    preprocessor = HousingDataPreprocessor.load(prep_path)
    test_data = np.load(test_path)
    
    test_cat = torch.tensor(test_data['test_cat'], dtype=torch.long)
    test_cont = torch.tensor(test_data['test_cont'], dtype=torch.float32)
    y_true_raw = test_data['test_y_raw'].flatten()
    
    # 2. Recreate and load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EmbeddedMLP(
        cat_dims=metadata['cat_dims'],
        emb_dims=metadata['emb_dims'],
        num_continuous=len(metadata['continuous_cols']),
        hidden_units=[256, 128, 64, 32]
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 3. Predict on Test Set
    with torch.no_grad():
        preds_scaled = model(test_cat.to(device), test_cont.to(device)).cpu().numpy()
        
    y_pred_raw = preprocessor.inverse_transform_target(preds_scaled).flatten()
    
    # 4. Calculate Key Resume Metrics
    mae = float(mean_absolute_error(y_true_raw, y_pred_raw))
    rmse = float(root_mean_squared_error(y_true_raw, y_pred_raw))
    r2 = float(r2_score(y_true_raw, y_pred_raw))
    mape = float(mean_absolute_percentage_error(y_true_raw, y_pred_raw)) * 100
    
    residuals = y_true_raw - y_pred_raw
    mean_res = float(np.mean(residuals))
    std_res = float(np.std(residuals))
    
    metrics = {
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2_Score': round(r2, 4),
        'MAPE_Percent': round(mape, 2),
        'Residual_Mean': round(mean_res, 2),
        'Residual_Std': round(std_res, 2),
        'Test_Samples_Count': len(y_true_raw)
    }
    
    # Save metrics JSON
    with open(os.path.join(reports_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
        
    # 5. Generate Diagnostic Plots
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Plot 1: Actual vs Predicted
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(y_true_raw / 1000, y_pred_raw / 1000, alpha=0.45, color='#2563eb', edgecolors='none', s=25)
    min_val = min(y_true_raw.min(), y_pred_raw.min()) / 1000
    max_val = max(y_true_raw.max(), y_pred_raw.max()) / 1000
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction ($y = \\hat{y}$)')
    ax.set_title('Embedded MLP: Actual vs. Predicted House Prices', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Actual Price ($ in Thousands)', fontsize=12)
    ax.set_ylabel('Predicted Price ($ in Thousands)', fontsize=12)
    ax.legend(frameon=True, facecolor='white', loc='upper left')
    ax.text(
        0.05, 0.85,
        f"$R^2 = {r2:.4f}$\nMAE = ${mae:,.0f}\nRMSE = ${rmse:,.0f}",
        transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor='#cbd5e1')
    )
    plt.tight_layout()
    fig.savefig(os.path.join(reports_dir, 'actual_vs_predicted.png'), dpi=200)
    plt.close()
    
    # Plot 2: Residuals Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals / 1000, bins=40, color='#0284c7', edgecolor='white', alpha=0.85)
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Error')
    ax.set_title('Residuals Error Distribution ($y - \\hat{y}$)', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Residual ($ in Thousands)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(reports_dir, 'residuals_distribution.png'), dpi=200)
    plt.close()
    
    # Plot 3: Training Loss Curve
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            hist = json.load(f)
        fig, ax = plt.subplots(figsize=(8, 5))
        epochs_range = hist['epochs']
        ax.plot(epochs_range, hist['train_loss'], label='Training Loss (SmoothL1)', color='#3b82f6', lw=2)
        ax.plot(epochs_range, hist['val_loss'], label='Validation Loss (SmoothL1)', color='#10b981', lw=2)
        ax.set_title('Training and Validation Loss Convergence', fontsize=14, fontweight='bold', pad=12)
        ax.set_xlabel('Epochs', fontsize=12)
        ax.set_ylabel('Loss (Huber / SmoothL1)', fontsize=12)
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(reports_dir, 'loss_curves.png'), dpi=200)
        plt.close()
        
    print("\n" + "=" * 60)
    print(" MODEL PERFORMANCE EVALUATION (TEST SET)")
    print("=" * 60)
    print(f"  • Mean Absolute Error (MAE):     ${mae:,.2f}")
    print(f"  • Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print(f"  • R² Score (Accuracy Metric):    {r2:.4f} ({r2*100:.2f}%)")
    print(f"  • Mean Absolute % Error (MAPE):  {mape:.2f}%")
    print(f"  • Test Dataset Size:             {len(y_true_raw)} samples")
    print("=" * 60)
    print(f"Plots and summary saved in '{reports_dir}/'")
    return metrics

if __name__ == '__main__':
    evaluate_model()
