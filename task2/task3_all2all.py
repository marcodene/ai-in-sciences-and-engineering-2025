"""
Task 3: All-to-All FNO Training
Train a time-dependent FNO using all time snapshots.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
import os

from models.fno_time import FNO1d

from config import device, USE_PRETRAINED, DATA_PATHS, MODEL_PATHS, TASK3_CONFIG
from datasets import All2AllDataset
from utils import compute_relative_l2_error, load_model


print("="*60)
print("TASK 3: All-to-All Training (time-dependent FNO)")
print("="*60)

# ============================================
# LOAD DATASETS
# ============================================
print("\nLoading datasets...")
train_dataset = All2AllDataset(DATA_PATHS['train'])
test_dataset = All2AllDataset(DATA_PATHS['test_128'])

train_loader = DataLoader(train_dataset, batch_size=TASK3_CONFIG['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=TASK3_CONFIG['batch_size'], shuffle=False)

print(f"Train batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")

# ============================================
# CREATE MODEL
# ============================================
model = FNO1d(
    modes=TASK3_CONFIG['modes'], 
    width=TASK3_CONFIG['width'], 
    in_dim=3,  # [x-coords, u(x, t_i), Δt]
    out_dim=1
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"\nModel: {num_params:,} parameters")
print(f"  Modes: {TASK3_CONFIG['modes']}, Width: {TASK3_CONFIG['width']}")

# ============================================
# TRAIN OR LOAD MODEL
# ============================================
if USE_PRETRAINED and os.path.exists(MODEL_PATHS['task3_all2all']):
    model = load_model(model, MODEL_PATHS['task3_all2all'], device)
else:
    if USE_PRETRAINED:
        print(f"\nWARNING: Pretrained model not found at {MODEL_PATHS['task3_all2all']}")
        print("Training new model instead...\n")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = Adam(
        model.parameters(), 
        lr=TASK3_CONFIG['learning_rate'], 
        weight_decay=TASK3_CONFIG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=TASK3_CONFIG['step_size'], 
        gamma=0.5
    )
    
    print("\nStarting training...")
    for epoch in range(TASK3_CONFIG['epochs']):
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
            print(f'Epoch {epoch+1}/{TASK3_CONFIG["epochs"]}: Train Loss = {train_loss:.6f}')
    
    print("\nTraining complete!")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATHS['task3_all2all'])
    print(f"Model saved to {MODEL_PATHS['task3_all2all']}")

# ============================================
# TASK 3.1: TEST AT t=1.0 (compare with Task 1)
# ============================================
print("\n" + "="*60)
print("TASK 3.1: Testing at t=1.0 (compare with one2one)")
print("="*60)

# Prepare input: u(t=0) with Δt=1.0 → target: u(t=1.0)
test_data = np.load(DATA_PATHS['test_128'])  # (128, 5, 128)
n_test = test_data.shape[0]
resolution = test_data.shape[2]
x_grid = np.linspace(0, 1, resolution, dtype=np.float32)

u_initial = test_data[:, 0, :]  # (128, 128) - all start from t=0
u_final = test_data[:, 4, :]    # (128, 128) - target at t=1.0

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
# TASK 3.2: TEST AT MULTIPLE TIMESTEPS
# ============================================
print("\n" + "="*60)
print("TASK 3.2: Testing at Multiple Timesteps")
print("="*60)

time_values = np.array([0.25, 0.50, 0.75, 1.0])
time_indices = [1, 2, 3, 4]  # Indices in data array
errors_by_time = {}

for t_val, t_idx in zip(time_values, time_indices):
    # Input: u(t=0) with Δt=t_val → target: u(t=t_val)
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
