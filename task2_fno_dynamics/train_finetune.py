"""
Task 2.4: Finetuning on Unknown Distribution

This script implements all three parts of Task 2.4:
1. Zero-shot testing (5 points)
2. Finetuning (10 points)
3. Training from scratch (10 bonus points)
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

# Hyperparameters for finetuning
batch_size = 8  # Smaller batch size for limited data
learning_rate_finetune = 5e-4  # Lower LR for finetuning
epochs_finetune = 50
learning_rate_scratch = 1e-3  # Higher LR for training from scratch
epochs_scratch = 200  # More epochs needed when training from scratch

# Model parameters (same as Task 3)
modes = 16
width = 64

# Data paths
finetune_train_path = "data/task2/data_finetune_train_unknown_128.npy"
finetune_val_path = "data/task2/data_finetune_val_unknown_128.npy"
test_unknown_path = "data/task2/data_test_unknown_128.npy"
pretrained_model_path = "models/task2_all2all.pt"

# ============================================
# DATASET (Reuse from train_all2all.py)
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
        
        # Timesteps: t = [0.0, 0.25, 0.5, 0.75, 1.0]
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
# UTILITY FUNCTIONS
# ============================================
def compute_relative_l2_error(pred, true):
    """Calcola la SOMMA degli errori relativi L2 per il batch."""
    pred = pred.squeeze(-1)  # (batch, resolution)
    true = true.squeeze(-1)  # (batch, resolution)
    
    errors = torch.norm(pred - true, dim=1) / torch.norm(true, dim=1)
    return errors.sum().item()

def test_at_t1(model, data_path, device):
    """
    Test the model at t=1.0 starting from t=0.
    
    Returns:
        Average relative L2 error
    """
    test_data = np.load(data_path)  # (N, 5, 128)
    n_test = test_data.shape[0]
    resolution = test_data.shape[2]
    x_grid = np.linspace(0, 1, resolution, dtype=np.float32)
    
    # Input: u(t=0) con Δt=1.0 → target: u(t=1.0)
    u_initial = test_data[:, 0, :]  # (N, 128)
    u_final = test_data[:, 4, :]    # (N, 128)
    
    delta_t = np.full((n_test, resolution), 1.0, dtype=np.float32)
    x_grid_batch = np.tile(x_grid, (n_test, 1))
    
    # Stack: [x, u(t=0), Δt=1.0]
    X_test = np.stack([x_grid_batch, u_initial, delta_t], axis=-1)
    Y_test = u_final[..., np.newaxis]
    
    # Convert to tensors
    X_test = torch.from_numpy(X_test).float().to(device)
    Y_test = torch.from_numpy(Y_test).float().to(device)
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        Y_pred = model(X_test)
        error = compute_relative_l2_error(Y_pred, Y_test) / n_test
    
    return error

def test_at_multiple_times(model, data_path, device):
    """
    Test the model at multiple timesteps: 0.25, 0.50, 0.75, 1.0
    starting from t=0.
    """
    test_data = np.load(data_path)
    n_test = test_data.shape[0]
    resolution = test_data.shape[2]
    x_grid = np.linspace(0, 1, resolution, dtype=np.float32)
    x_grid_batch = np.tile(x_grid, (n_test, 1))
    
    u_initial = test_data[:, 0, :]  # (N, 128) - all start from t=0
    
    time_values = np.array([0.25, 0.50, 0.75, 1.0])
    time_indices = [1, 2, 3, 4]  # Indices in data array
    
    errors = {}
    print("  Performance over time:")
    
    for t_val, t_idx in zip(time_values, time_indices):
        # Target u at specific time
        u_target = test_data[:, t_idx, :]
        
        # Calculate Delta t = t_val - 0.0 = t_val
        delta_t_batch = np.full((n_test, resolution), t_val, dtype=np.float32)
        
        X_test = np.stack([x_grid_batch, u_initial, delta_t_batch], axis=-1)
        Y_test = u_target[..., np.newaxis]
        
        X_test = torch.from_numpy(X_test).float().to(device)
        Y_test = torch.from_numpy(Y_test).float().to(device)
        
        model.eval()
        with torch.no_grad():
            Y_pred = model(X_test)
            error = compute_relative_l2_error(Y_pred, Y_test) / n_test
            
        errors[t_val] = error
        print(f"    t={t_val:.2f}: Error = {error:.6f}")
        
    return errors

def train_model(model, train_loader, val_dataset, epochs, lr, device, model_name):
    """
    Train or finetune a model.
    
    Args:
        model: FNO model to train
        train_loader: DataLoader for training
        val_dataset: Dataset for validation (for computing error at t=1.0)
        epochs: Number of epochs
        lr: Learning rate
        device: Device to use
        model_name: Name for saving the model
    
    Returns:
        Trained model
    """
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    print(f"\nTraining {model_name}...")
    
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
        
        # Validation (test at t=1.0)
        if (epoch + 1) % 10 == 0:
            val_error = test_at_t1(model, val_dataset, device)
            print(f'Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.6f}, Val Error (t=1.0) = {val_error:.6f}')
        
        scheduler.step()
    
    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), f'models/{model_name}.pt')
    print(f"Model saved to models/{model_name}.pt")
    
    return model

# ============================================
# TASK 2.4.1: ZERO-SHOT TESTING
# ============================================
print("\n" + "="*70)
print("TASK 2.4.1: Zero-Shot Testing on Unknown Distribution")
print("="*70)

# Load pretrained model from Task 3
model_pretrained = FNO1d(modes=modes, width=width, in_dim=3, out_dim=1).to(device)

if os.path.exists(pretrained_model_path):
    model_pretrained.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    print(f"Loaded pretrained model from {pretrained_model_path}")
else:
    print(f"WARNING: Pretrained model not found at {pretrained_model_path}")
    print("Please run train_all2all.py first to train the model!")
    exit(1)

# Test on unknown distribution (zero-shot)
# Test on unknown distribution (zero-shot)
errors_zeroshot = test_at_multiple_times(model_pretrained, test_unknown_path, device)
error_zeroshot = errors_zeroshot[1.0]
print(f"\nZero-Shot Error on Unknown Distribution (t=1.0): {error_zeroshot:.6f}")
print("This is the model's performance without any finetuning.")

# ============================================
# TASK 2.4.2: FINETUNING
# ============================================
print("\n" + "="*70)
print("TASK 2.4.2: Finetuning on Unknown Distribution")
print("="*70)

# Load finetuning datasets
finetune_train_dataset = All2AllDataset(finetune_train_path)
finetune_train_loader = DataLoader(finetune_train_dataset, batch_size=batch_size, shuffle=True)

# Create a copy of the pretrained model for finetuning
model_finetuned = FNO1d(modes=modes, width=width, in_dim=3, out_dim=1).to(device)
model_finetuned.load_state_dict(model_pretrained.state_dict())

# Finetune the model
model_finetuned = train_model(
    model_finetuned,
    finetune_train_loader,
    finetune_val_path,
    epochs_finetune,
    learning_rate_finetune,
    device,
    "task2_finetuned"
)

# Test finetuned model
# Test finetuned model
errors_finetuned = test_at_multiple_times(model_finetuned, test_unknown_path, device)
error_finetuned = errors_finetuned[1.0]
print(f"\nFinetuned Model Error on Unknown Distribution (t=1.0): {error_finetuned:.6f}")
print(f"Improvement: {error_zeroshot - error_finetuned:.6f} ({((error_zeroshot - error_finetuned) / error_zeroshot * 100):.1f}% reduction)")

# ============================================
# TASK 2.4.3: TRAIN FROM SCRATCH (BONUS)
# ============================================
print("\n" + "="*70)
print("TASK 2.4.3 (BONUS): Training from Scratch on Unknown Distribution")
print("="*70)

# Create new model from scratch
model_scratch = FNO1d(modes=modes, width=width, in_dim=3, out_dim=1).to(device)

# Train from scratch
model_scratch = train_model(
    model_scratch,
    finetune_train_loader,
    finetune_val_path,
    epochs_scratch,
    learning_rate_scratch,
    device,
    "task2_scratch"
)

# Test model trained from scratch
# Test model trained from scratch
errors_scratch = test_at_multiple_times(model_scratch, test_unknown_path, device)
error_scratch = errors_scratch[1.0]
print(f"\nFrom-Scratch Model Error on Unknown Distribution (t=1.0): {error_scratch:.6f}")

# ============================================
# FINAL COMPARISON
# ============================================
print("\n" + "="*70)
print("FINAL RESULTS SUMMARY")
print("="*70)
print(f"Zero-Shot (no finetuning):     {error_zeroshot:.6f}")
print(f"Finetuned (32 trajectories):   {error_finetuned:.6f}")
print(f"From Scratch (32 trajectories): {error_scratch:.6f}")
print("\n" + "="*70)

if error_finetuned < error_scratch:
    improvement = (error_scratch - error_finetuned) / error_scratch * 100
    print(f"✅ Transfer Learning is SUCCESSFUL!")
    print(f"   Finetuning is {improvement:.1f}% better than training from scratch.")
else:
    print(f"❌ Transfer Learning is NOT successful for this case.")
    print(f"   Training from scratch performs better.")
print("="*70)

# Save results
results = {
    'error_zeroshot': error_zeroshot,
    'error_finetuned': error_finetuned,
    'error_scratch': error_scratch
}
np.save('models/results_task4.npy', results)
print("\nResults saved to models/results_task4.npy")
