# compare_models.py
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys

from koopman import KoopmanGenerator, discretize_generator

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
CSV_FILE = 'cstr_simulation_data.csv'
# PINN and Vanilla models paths (assumed to exist or we skip them if missing)
PINN_MODEL = 'train_results/trained_model.pth'
VANILLA_MODEL = 'train_results_vanilla/vanilla_model.pth'

SAVE_DIR = 'comparison_results'
os.makedirs(SAVE_DIR, exist_ok=True)

# ────────────────────────────────────────────────
# MODEL DEFINITIONS (Placeholder if files missing)
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

def predict_koopman(df):
    print("Fitting Koopman model for comparison...")
    if not os.path.exists('koopman_data.csv'):
        print("koopman_data.csv not found, skipping Koopman.")
        return None

    generator = KoopmanGenerator('koopman_data.csv')
    K, L = generator.fit(lambda_reg=1e-5)

    t = df['time'].values
    u = df['u_input'].values
    CA_true = df['CA_concentration'].values

    # Compute v
    v = np.zeros_like(u)
    # v[i] approx (u[i+1] - u[i]) / dt[i+1]
    # But we need v applied at step i.
    # u[i+1] = u[i] + v[i]*dt
    dt_vals = np.diff(t)
    v[:-1] = np.diff(u) / dt_vals
    v[-1] = v[-2] # Last v

    preds = np.zeros_like(CA_true)
    preds[0] = CA_true[0]

    z_curr = generator.lifting.lift([CA_true[0]], [u[0]])[0]

    for i in range(len(t)-1):
        dt = dt_vals[i]
        A, B = discretize_generator(K, L, dt)

        v_k = v[i]

        # z_next = A z + B v
        z_next = A @ z_curr + B.flatten() * v_k

        preds[i+1] = z_next[0]

        # Update z_curr
        # Correct u-component to prevent drift
        # z_next[1] corresponds to u[i+1]
        # We can force it to be true u[i+1] for open loop prediction
        z_next[1] = u[i+1]

        # Re-lift dependent variables
        z_next[2] = z_next[0] * z_next[1] # xu
        z_next[3] = 1.0
        if generator.lifting.n_z > 4:
            z_next[4] = z_next[0]**2
            z_next[5] = z_next[1]**2

        z_curr = z_next

    return preds

# ────────────────────────────────────────────────
# MAIN COMPARISON
# ────────────────────────────────────────────────
def compare():
    print("="*60)
    print("Model Comparison: PINN vs Vanilla vs Koopman")
    print("="*60)

    # Load test data
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found. Run simulate_data.py.")
        return

    df = pd.read_csv(CSV_FILE)
    CA_true = df['CA_concentration'].values
    t = df['time'].values

    results = {}
    results['True'] = CA_true

    # Koopman
    CA_koopman = predict_koopman(df)
    if CA_koopman is not None:
        results['Koopman'] = CA_koopman

    # PINN / Vanilla (Try to load)
    try:
        if os.path.exists(PINN_MODEL):
            pinn_model = PINN_CSTR()
            pinn_model.load_state_dict(torch.load(PINN_MODEL))
            pinn_model.eval()
            CA_pinn, _ = rollout_prediction(pinn_model, df)
            results['PINN'] = CA_pinn
        else:
            print("PINN model not found, skipping.")
    except Exception as e:
        print(f"Error loading PINN: {e}")

    try:
        if os.path.exists(VANILLA_MODEL):
            vanilla_model = Vanilla_NN()
            vanilla_model.load_state_dict(torch.load(VANILLA_MODEL))
            vanilla_model.eval()
            CA_vanilla, _ = rollout_prediction(vanilla_model, df)
            results['Vanilla'] = CA_vanilla
        else:
            print("Vanilla model not found, skipping.")
    except Exception as e:
        print(f"Error loading Vanilla: {e}")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(t, CA_true, label='True (Simulink)', color='black', lw=2)

    if 'Koopman' in results:
        plt.plot(t, results['Koopman'], '--', label='Koopman', color='red', lw=2)

    if 'PINN' in results:
        plt.plot(t, results['PINN'], ':', label='PINN', color='orange', lw=2)

    if 'Vanilla' in results:
        plt.plot(t, results['Vanilla'], '-.', label='Vanilla', color='green', lw=2)

    plt.xlabel('Time')
    plt.ylabel('C_A')
    plt.title('Model Prediction Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, 'all_models_comparison.png'))
    print(f"Saved comparison plot to {SAVE_DIR}/all_models_comparison.png")

    # Calculate Metrics
    print("\nMetrics (RMSE):")
    for name, data in results.items():
        if name == 'True': continue
        rmse = np.sqrt(np.mean((data - CA_true)**2))
        print(f"{name}: {rmse:.6f}")

if __name__ == '__main__':
    compare()
