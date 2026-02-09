# train.py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

# CONFIG
# Import shared path
from config import CSV_FILE, SAVE_DIR
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15   

SAVE_DIR = 'train_results'
os.makedirs(SAVE_DIR, exist_ok=True)

EPOCHS = 8000
LR = 1e-3
BATCH_SIZE = 256
PHYSICS_WEIGHT = 1.0
DATA_WEIGHT = 10.0

# System constants
C_Ai = 1.0
k = 0.028


# MODEL
class PINN_CSTR(nn.Module):
    def __init__(self, hidden_layers=4, neurons=64):
        super().__init__()
        layers = [nn.Linear(3, neurons), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(neurons, neurons), nn.Tanh()])
        layers.append(nn.Linear(neurons, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, C_A0, u, t):
        x = torch.cat([C_A0, u, t], dim=1)
        return self.network(x)

# DATASET (sequential transitions)
class TransitionDataset(Dataset):
    def __init__(self, df):
        self.rows = []
        t = df['time'].values
        u = df['u_input'].values
        CA = df['CA_concentration'].values
        for i in range(len(t)-1):
            dt = t[i+1] - t[i]
            self.rows.append({
                'C_A0': CA[i],
                'u': u[i],
                'dt': dt,
                'CA_target': CA[i+1]
            })

    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        return {
            'C_A0': torch.tensor([r['C_A0']], dtype=torch.float32),
            'u':    torch.tensor([r['u']],    dtype=torch.float32),
            'dt':   torch.tensor([r['dt']],   dtype=torch.float32),
            'CA_target': torch.tensor([r['CA_target']], dtype=torch.float32)
        }


# LOSS
def compute_loss(model, batch):
    C_A0 = batch['C_A0']
    u    = batch['u']
    dt   = batch['dt'].clone().detach().requires_grad_(True)
    CA_target = batch['CA_target']

    CA_pred = model(C_A0, u, dt)

    dCA_dt = torch.autograd.grad(
        CA_pred, dt,
        grad_outputs=torch.ones_like(CA_pred),
        create_graph=True
    )[0]

    rhs = u * (C_Ai - CA_pred) - k * CA_pred
    residual = dCA_dt - rhs
    physics_loss = torch.mean(residual ** 2)

    data_loss = torch.mean((CA_pred - CA_target) ** 2)

    total_loss = PHYSICS_WEIGHT * physics_loss + DATA_WEIGHT * data_loss

    return physics_loss, data_loss, total_loss


# TRAINING
def train():
    print("Loading data...")
    df = pd.read_csv(CSV_FILE)
    n = len(df)
    train_size = int(TRAIN_FRAC * n)
    df_train = df.iloc[:train_size]

    print(f"Training on first {train_size} points ({TRAIN_FRAC*100:.0f}%)")

    dataset = TransitionDataset(df_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = PINN_CSTR()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    loss_hist = {'physics': [], 'data': [], 'total': []}

    print("Training started...")
    start_time = datetime.now()

    for epoch in range(EPOCHS):
        epoch_p, epoch_d, epoch_t = 0, 0, 0
        n_batch = 0

        for batch in loader: #Get Data Batch
            optimizer.zero_grad()  #clear memory
            p_loss, d_loss, t_loss = compute_loss(model, batch) #calculate error
            t_loss.backward() #calculate adjustments..That is, did back propagation
            optimizer.step() #apply adjustments and update weight

            epoch_p += p_loss.item()
            epoch_d += d_loss.item()
            epoch_t += t_loss.item()
            n_batch += 1

        avg_p = epoch_p / n_batch
        avg_d = epoch_d / n_batch
        avg_t = epoch_t / n_batch

        loss_hist['physics'].append(avg_p)
        loss_hist['data'].append(avg_d)
        loss_hist['total'].append(avg_t)

        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1:5d} | P: {avg_p:.2e} | D: {avg_d:.2e} | Total: {avg_t:.2e}")

    training_time = (datetime.now() - start_time).total_seconds()

    # Save model & history
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'trained_model.pth'))
    pd.DataFrame(loss_hist).to_csv(os.path.join(SAVE_DIR, 'loss_history.csv'), index=False)

    # Loss plot
    epochs = np.arange(1, EPOCHS+1)
    plt.figure(figsize=(9,5))
    plt.semilogy(epochs, loss_hist['physics'], label='Physics')
    plt.semilogy(epochs, loss_hist['data'],    label='Data')
    plt.semilogy(epochs, loss_hist['total'],   label='Total', lw=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Training Loss Curves')
    plt.savefig(os.path.join(SAVE_DIR, 'train_loss.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # Metrics summary
    metrics = {
        'final_physics_loss': float(loss_hist['physics'][-1]),
        'final_data_loss':    float(loss_hist['data'][-1]),
        'final_total_loss':   float(loss_hist['total'][-1]),
        'epochs': EPOCHS,
        'training_time_s': training_time,
        'training_time_min': training_time/60
    }

    with open(os.path.join(SAVE_DIR, 'train_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Training finished. Results saved in '{SAVE_DIR}'")

if __name__ == '__main__':
    train()