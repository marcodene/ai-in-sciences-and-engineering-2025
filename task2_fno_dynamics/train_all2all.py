"""
Task 2.3: All-to-All FNO Training
Train a time-dependent FNO using all time snapshots.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

from models.fno_time import FNO1d

# ============================================
# CONFIGURAZIONE
# ============================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
batch_size = 32
learning_rate = 1e-3
epochs = 100
step_size = 100

# Model parameters
modes = 16
width = 64

# Data paths
train_data_path = "data/task2/data_train_128.npy"
val_data_path = "data/task2/data_val_128.npy"
test_data_path = "data/task2/data_test_128.npy"

# ============================================
# DATASET
# ============================================
class All2AllDataset(Dataset):
    """
    Dataset per all2all training: crea coppie (t_i, t_j) con t_i < t_j.
    
    Input:  [x-coords, u(x, t_i), Δt]
    Output: u(x, t_j)
    
    Dove Δt = t_j - t_i
    """
    def __init__(self, data_path):
        # Load data
        self.data = np.load(data_path)  # (N_traj, 5, 128)
        self.n_trajectories = self.data.shape[0]
        self.n_times = self.data.shape[1]
        self.resolution = self.data.shape[2]
        
        # Timesteps: t = [0.0, 0.25, 0.50, 0.75, 1.0]
        self.time_values = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        
        # Griglia spaziale (fissa per tutti)
        self.x_grid = np.linspace(0, 1, self.resolution, dtype=np.float32)
        
        # Crea tutte le coppie (i, j) con i < j (solo forward in time)
        self.time_pairs = [
            (i, j) for i in range(self.n_times) 
            for j in range(i + 1, self.n_times)
        ]
        self.n_pairs = len(self.time_pairs)  # 4+3+2+1 = 10 coppie
        
        # Numero totale di samples
        self.length = self.n_trajectories * self.n_pairs
        
        print(f"Dataset: {self.length} samples ({self.n_trajectories} traj × {self.n_pairs} pairs)")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        """
        Returns:
            x_input: (128, 3) - [x-coords, u(x, t_i), Δt]
            y_target: (128, 1) - u(x, t_j)
        """
        # Mappa indice lineare -> (trajectory, time_pair)
        traj_idx = index // self.n_pairs
        pair_idx = index % self.n_pairs
        
        # Estrai indici temporali
        i, j = self.time_pairs[pair_idx]
        
        # Estrai u(x) ai due tempi
        u_in = self.data[traj_idx, i, :]   # (128,) - u(x, t_i)
        u_out = self.data[traj_idx, j, :]  # (128,) - u(x, t_j)
        
        # Calcola Δt = t_j - t_i
        delta_t = self.time_values[j] - self.time_values[i]
        
        # Ripeti Δt per ogni punto spaziale
        delta_t_repeated = np.full(self.resolution, delta_t, dtype=np.float32)
        
        # Crea input: [x-coords, u(x, t_i), Δt]
        x_input = np.stack([
            self.x_grid,
            u_in,
            delta_t_repeated
        ], axis=-1)  # (128, 3)
        
        # Target: u(x, t_j)
        y_target = u_out[..., np.newaxis]  # (128, 1)
        
        return (
            torch.from_numpy(x_input).float(),
            torch.from_numpy(y_target).float()
        )

# ============================================
# CARICA DATI
# ============================================
print("\nLoading datasets...")
train_dataset = All2AllDataset(train_data_path)
test_dataset = All2AllDataset(test_data_path)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ============================================
# MODELLO
# ============================================
model = FNO1d(modes=modes, width=width, in_dim=3, out_dim=1).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"\nModel: {num_params:,} parameters")
print(f"  Modes: {modes}, Width: {width}")

# ============================================
# TRAINING SETUP
# ============================================
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)

def compute_relative_l2_error(pred, true):
    """Calcola la SOMMA degli errori relativi L2 per il batch."""
    pred = pred.squeeze(-1)  # (batch, resolution)
    true = true.squeeze(-1)  # (batch, resolution)
    
    errors = torch.norm(pred - true, dim=1) / torch.norm(true, dim=1)
    return errors.sum().item()

# ============================================
# TRAINING LOOP
# ============================================
print("\nStarting training...")

for epoch in range(epochs):
    # Training
    model.train()
    train_loss = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Scheduler
    scheduler.step()
    
    if (epoch + 1) % 5 == 0:
        print(f'Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.6f}')

print("\nTraining complete!")

# Save final model
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/task2_all2all.pt')
print("Model saved to models/task2_all2all.pt")

# ============================================
# TASK 2.3.1: TEST AT t=1.0 (compare with Task 1)
# ============================================
print("\n" + "="*60)
print("TASK 2.3.1: Testing at t=1.0 (compare with one2one)")
print("="*60)

# Filtra solo samples con Δt che portano a t=1.0
# Dobbiamo creare input specifici: [x, u(t=0), Δt=1.0]
test_data = np.load(test_data_path)  # (128, 5, 128)
n_test = test_data.shape[0]
resolution = test_data.shape[2]
x_grid = np.linspace(0, 1, resolution, dtype=np.float32)

# Input: u(t=0) con Δt=1.0 → target: u(t=1.0)
u_initial = test_data[:, 0, :]  # (128, 128) - tutti iniziano da t=0
u_final = test_data[:, 4, :]    # (128, 128) - target a t=1.0

delta_t = np.full((n_test, resolution), 1.0, dtype=np.float32)  # (128, 128)
x_grid_batch = np.tile(x_grid, (n_test, 1))  # (128, 128)

# Stack: [x, u(t=0), Δt=1.0]
X_test_t1 = np.stack([x_grid_batch, u_initial, delta_t], axis=-1)  # (128, 128, 3)
Y_test_t1 = u_final[..., np.newaxis]  # (128, 128, 1)

# Convert to tensors
X_test_t1 = torch.from_numpy(X_test_t1).float().to(device)
Y_test_t1 = torch.from_numpy(Y_test_t1).float().to(device)

# Evaluate
model.eval()
with torch.no_grad():
    Y_pred_t1 = model(X_test_t1)
    test_error_t1 = compute_relative_l2_error(Y_pred_t1, Y_test_t1) / n_test

print(f"Average Relative L2 Error at t=1.0: {test_error_t1:.6f}")
print("Compare this to Task 1 (one2one) result!")

# ============================================
# TASK 2.3.2: TEST AT MULTIPLE TIMESTEPS
# ============================================
print("\n" + "="*60)
print("TASK 2.3.2: Testing at Multiple Timesteps")
print("="*60)

time_values = np.array([0.25, 0.50, 0.75, 1.0])
time_indices = [1, 2, 3, 4]  # Indici nel data array
errors_by_time = {}

for t_val, t_idx in zip(time_values, time_indices):
    # Input: u(t=0) con Δt=t_val → target: u(t=t_val)
    u_target = test_data[:, t_idx, :]  # (128, 128)
    
    delta_t_batch = np.full((n_test, resolution), t_val, dtype=np.float32)
    
    X_test_t = np.stack([x_grid_batch, u_initial, delta_t_batch], axis=-1)
    Y_test_t = u_target[..., np.newaxis]
    
    X_test_t = torch.from_numpy(X_test_t).float().to(device)
    Y_test_t = torch.from_numpy(Y_test_t).float().to(device)
    
    with torch.no_grad():
        Y_pred_t = model(X_test_t)
        error_t = compute_relative_l2_error(Y_pred_t, Y_test_t) / n_test
    
    errors_by_time[t_val] = error_t
    print(f"t={t_val:.2f}: Error = {error_t:.6f}")

print("\nObservation: Error typically increases with time as predictions")
print("become more challenging (longer time evolution).")
print("="*60)

# Save results
results = {
    'test_error_t1': test_error_t1,
    'errors_by_time': errors_by_time
}
np.save('models/results_all2all.npy', results)
print("\nResults saved to models/results_all2all.npy")