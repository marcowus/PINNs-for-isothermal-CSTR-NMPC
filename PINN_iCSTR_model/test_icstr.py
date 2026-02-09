# test.py
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os


# CONFIG
# Import shared path
from config import CSV_FILE, SAVE_DIR
MODEL_PATH = 'train_results/trained_model.pth'

SAVE_DIR = 'test_results'
os.makedirs(SAVE_DIR, exist_ok=True)

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

C_Ai = 1.0
k = 0.028

# MODEL 
# 
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


# ROLLOUT PREDICTION
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

    errors = CA_true - CA_pred

    rmse = np.sqrt(np.mean(errors**2))
    mae  = np.mean(np.abs(errors))
    mape = np.mean(np.abs(errors / (CA_true + 1e-10))) * 100
    ss_res = np.sum(errors**2)
    ss_tot = np.sum((CA_true - np.mean(CA_true))**2)
    r2   = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2
    }, CA_true, CA_pred, t

# MAIN TEST

def test():
    print("Loading test data...")
    df = pd.read_csv(CSV_FILE)
    n = len(df)
    train_size = int(TRAIN_FRAC * n)
    val_size   = int(VAL_FRAC * n)
    df_test = df.iloc[train_size + val_size:]

    print(f"Testing on last {len(df_test)} points ({(1-TRAIN_FRAC-VAL_FRAC)*100:.0f}%)")

    model = PINN_CSTR()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    metrics, CA_true, CA_pred, t = rollout_prediction(model, df_test)

    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(t, CA_true, label='True (Simulink)', lw=2)
    plt.plot(t, CA_pred, '--', label='PINN rollout', lw=2.5)
    plt.xlabel('Time')
    plt.ylabel('C_A concentration')
    plt.title(f'Test Rollout (unseen segment)\nR² = {metrics["r2"]:.4f} | RMSE = {metrics["rmse"]:.6f} | MAE = {metrics["mae"]:.6f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(SAVE_DIR, 'test_rollout.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # Save metrics
    with open(os.path.join(SAVE_DIR, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    print("Test evaluation complete.")
    print(f"R²    : {metrics['r2']:.6f}")
    print(f"RMSE  : {metrics['rmse']:.6f}")
    print(f"MAE   : {metrics['mae']:.6f}")
    print(f"MAPE  : {metrics['mape']:.2f}%")
    print(f"Results saved in '{SAVE_DIR}'")

if __name__ == '__main__':
    test()