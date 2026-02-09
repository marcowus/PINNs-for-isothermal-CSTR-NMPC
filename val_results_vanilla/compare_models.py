# compare_models.py
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
CSV_FILE = 'cstr_simulation_data.csv'
PINN_MODEL = 'train_results/trained_model.pth'
VANILLA_MODEL = 'train_results_vanilla/vanilla_model.pth'

SAVE_DIR = 'comparison_results'
os.makedirs(SAVE_DIR, exist_ok=True)

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

# ────────────────────────────────────────────────
# MODEL DEFINITIONS
# ────────────────────────────────────────────────
class PINN_CSTR(torch.nn.Module):
    def __init__(self, hidden_layers=4, neurons=64):
        super().__init__()
        layers = [torch.nn.Linear(3, neurons), torch.nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([torch.nn.Linear(neurons, neurons), torch.nn.Tanh()])
        layers.append(torch.nn.Linear(neurons, 1))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, C_A0, u, t):
        x = torch.cat([C_A0, u, t], dim=1)
        return self.network(x)

class Vanilla_NN(torch.nn.Module):
    def __init__(self, hidden_layers=4, neurons=64):
        super().__init__()
        layers = [torch.nn.Linear(3, neurons), torch.nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([torch.nn.Linear(neurons, neurons), torch.nn.Tanh()])
        layers.append(torch.nn.Linear(neurons, 1))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, C_A0, u, t):
        x = torch.cat([C_A0, u, t], dim=1)
        return self.network(x)

# ────────────────────────────────────────────────
# ROLLOUT PREDICTION
# ────────────────────────────────────────────────
def rollout_prediction(model, df):
    t = df['time'].values
    u = df['u_input'].values
    CA_true = df['CA_concentration'].values

    CA_pred = np.zeros_like(CA_true)
    CA_pred[0] = CA_true[0]
    CA_current = CA_true[0]

    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        u_now = u[i-1]

        with torch.no_grad():
            inp = torch.tensor([[CA_current, u_now, dt]], dtype=torch.float32)
            CA_next = model(inp[:,0:1], inp[:,1:2], inp[:,2:3]).item()

        CA_pred[i] = CA_next
        CA_current = CA_next

    return CA_pred, t

# ────────────────────────────────────────────────
# MAIN COMPARISON
# ────────────────────────────────────────────────
def compare():
    print("="*60)
    print("PINN vs Vanilla NN Comparison")
    print("="*60)

    # ──────────────────────────────────────────────
    # 1. LOAD METRICS
    # ──────────────────────────────────────────────
    print("\n[1/4] Loading metrics...")
    
    with open('val_results/val_metrics.json', 'r') as f:
        pinn_val = json.load(f)
    with open('test_results/test_metrics.json', 'r') as f:
        pinn_test = json.load(f)
    
    with open('val_results_vanilla/val_metrics.json', 'r') as f:
        vanilla_val = json.load(f)
    with open('test_results_vanilla/test_metrics.json', 'r') as f:
        vanilla_test = json.load(f)

    # ──────────────────────────────────────────────
    # 2. CREATE METRICS TABLE
    # ──────────────────────────────────────────────
    print("[2/4] Creating metrics comparison table...")
    
    metrics_data = {
        'Metric': ['R²', 'RMSE', 'MAE', 'MAPE (%)'],
        'PINN (Val)': [
            f"{pinn_val['r2']:.6f}",
            f"{pinn_val['rmse']:.6f}",
            f"{pinn_val['mae']:.6f}",
            f"{pinn_val['mape']:.2f}"
        ],
        'Vanilla (Val)': [
            f"{vanilla_val['r2']:.6f}",
            f"{vanilla_val['rmse']:.6f}",
            f"{vanilla_val['mae']:.6f}",
            f"{vanilla_val['mape']:.2f}"
        ],
        'PINN (Test)': [
            f"{pinn_test['r2']:.6f}",
            f"{pinn_test['rmse']:.6f}",
            f"{pinn_test['mae']:.6f}",
            f"{pinn_test['mape']:.2f}"
        ],
        'Vanilla (Test)': [
            f"{vanilla_test['r2']:.6f}",
            f"{vanilla_test['rmse']:.6f}",
            f"{vanilla_test['mae']:.6f}",
            f"{vanilla_test['mape']:.2f}"
        ]
    }
    
    df_metrics = pd.DataFrame(metrics_data)
    df_metrics.to_csv(os.path.join(SAVE_DIR, 'metrics_comparison.csv'), index=False)
    
    print("\nMetrics Comparison:")
    print(df_metrics.to_string(index=False))

    # ──────────────────────────────────────────────
    # 3. LOAD DATA AND MODELS FOR TEST SET
    # ──────────────────────────────────────────────
    print("\n[3/4] Loading models and generating predictions on test set...")
    
    df = pd.read_csv(CSV_FILE)
    n = len(df)
    train_size = int(TRAIN_FRAC * n)
    val_size = int(VAL_FRAC * n)
    df_test = df.iloc[train_size + val_size:]
    
    CA_true = df_test['CA_concentration'].values
    
    # PINN predictions
    pinn_model = PINN_CSTR()
    pinn_model.load_state_dict(torch.load(PINN_MODEL))
    pinn_model.eval()
    CA_pinn, t = rollout_prediction(pinn_model, df_test)
    
    # Vanilla predictions
    vanilla_model = Vanilla_NN()
    vanilla_model.load_state_dict(torch.load(VANILLA_MODEL))
    vanilla_model.eval()
    CA_vanilla, _ = rollout_prediction(vanilla_model, df_test)

    # ──────────────────────────────────────────────
    # 4. GENERATE PLOTS
    # ──────────────────────────────────────────────
    print("[4/4] Generating comparison plots...")
    
    # Plot 1: Side-by-side predictions
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, CA_true, label='True (Simulink)', color='#1f77b4', lw=2)
    ax.plot(t, CA_pinn, '--', label='PINN', color='#ff7f0e', lw=2)
    ax.plot(t, CA_vanilla, ':', label='Vanilla NN', color='#2ca02c', lw=2.5)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('C_A concentration', fontsize=12)
    ax.set_title('Test Set: PINN vs Vanilla NN Predictions', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'predictions_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Error analysis
    error_pinn = np.abs(CA_true - CA_pinn)
    error_vanilla = np.abs(CA_true - CA_vanilla)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, error_pinn, label='PINN Error', color='#ff7f0e', lw=2, alpha=0.8)
    ax.plot(t, error_vanilla, label='Vanilla NN Error', color='#2ca02c', lw=2, alpha=0.8)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Absolute Error', fontsize=12)
    ax.set_title('Test Set: Absolute Prediction Errors', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'error_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Metrics bar chart
    metrics_names = ['R²', 'RMSE', 'MAE', 'MAPE (%)']
    pinn_vals = [pinn_test['r2'], pinn_test['rmse'], pinn_test['mae'], pinn_test['mape']]
    vanilla_vals = [vanilla_test['r2'], vanilla_test['rmse'], vanilla_test['mae'], vanilla_test['mape']]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, (metric, pinn_v, vanilla_v) in enumerate(zip(metrics_names, pinn_vals, vanilla_vals)):
        x = ['PINN', 'Vanilla']
        y = [pinn_v, vanilla_v]
        colors = ['#ff7f0e', '#2ca02c']
        axes[i].bar(x, y, color=colors, alpha=0.8)
        axes[i].set_ylabel(metric, fontsize=11)
        axes[i].set_title(f'{metric} Comparison (Test Set)', fontsize=12, fontweight='bold')
        axes[i].grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for j, v in enumerate(y):
            axes[i].text(j, v, f'{v:.4f}' if i > 0 else f'{v:.6f}', 
                        ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'metrics_bars.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # ──────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("COMPARISON COMPLETE!")
    print("="*60)
    print(f"\nResults saved in '{SAVE_DIR}/'")
    print("\nGenerated files:")
    print("  • metrics_comparison.csv")
    print("  • predictions_comparison.png")
    print("  • error_comparison.png")
    print("  • metrics_bars.png")
    print("\n" + "="*60)

if __name__ == '__main__':
    compare()