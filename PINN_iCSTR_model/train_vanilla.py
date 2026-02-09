# train_vanilla.py
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
CSV_FILE = 'cstr_simulation_data.csv'
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15

SAVE_DIR = 'train_results_vanilla'
os.makedirs(SAVE_DIR, exist_ok=True)

EPOCHS = 8000
LR = 1e-3
BATCH_SIZE = 256


# MODEL 
class Vanilla_NN(nn.Module):
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


# LOSS (I made it pure data driven since vanilla)
def compute_loss(model, batch):
    C_A0 = batch['C_A0']
    u    = batch['u']
    dt   = batch['dt']
    CA_target = batch['CA_target']

    CA_pred = model(C_A0, u, dt)
    data_loss = torch.mean((CA_pred - CA_target) ** 2)

    return data_loss


# TRAINING
def train():
    print("Loading data...")
    df = pd.read_csv(CSV_FILE)
    n = len(df)
    train_size = int(TRAIN_FRAC * n)
    df_train = df.iloc[:train_size]

    print(f"Training Vanilla NN on first {train_size} points ({TRAIN_FRAC*100:.0f}%)")

    dataset = TransitionDataset(df_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Vanilla_NN()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    loss_hist = []

    print("Training started...")
    start_time = datetime.now()

    for epoch in range(EPOCHS):
        epoch_loss = 0
        n_batch = 0

        for batch in loader:
            optimizer.zero_grad()
            loss = compute_loss(model, batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batch += 1

        avg_loss = epoch_loss / n_batch
        loss_hist.append(avg_loss)

        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1:5d} | Loss: {avg_loss:.2e}")

    training_time = (datetime.now() - start_time).total_seconds()

    # Save model & history
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'vanilla_model.pth'))
    pd.DataFrame({'loss': loss_hist}).to_csv(os.path.join(SAVE_DIR, 'loss_history.csv'), index=False)

    # Loss plot
    epochs = np.arange(1, EPOCHS+1)
    plt.figure(figsize=(9,5))
    plt.semilogy(epochs, loss_hist, label='Data Loss', lw=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Vanilla NN Training Loss')
    plt.savefig(os.path.join(SAVE_DIR, 'train_loss.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # Metrics summary
    metrics = {
        'final_loss': float(loss_hist[-1]),
        'epochs': EPOCHS,
        'training_time_s': training_time,
        'training_time_min': training_time/60
    }

    with open(os.path.join(SAVE_DIR, 'train_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Training finished. Results saved in '{SAVE_DIR}'")

if __name__ == '__main__':
    train()