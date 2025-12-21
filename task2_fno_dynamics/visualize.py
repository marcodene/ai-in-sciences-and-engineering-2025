import numpy as np
import torch
import matplotlib.pyplot as plt
from models.fno import FNO1d
import os

# Create plots directory
os.makedirs('plots', exist_ok=True)

# ============================================
# CARICAMENTO DATI
# ============================================
train_data = np.load("data/task2/data_train_128.npy")
test_data = np.load("data/task2/data_test_128.npy")

print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# ============================================
# VISUALIZZA TRAJECTORIES DAL TRAINING SET
# ============================================
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Training Data - Sample Trajectories', fontsize=16)

x = np.linspace(0, 1, 128)
time_steps = [0, 1, 2, 3, 4]
time_values = [0.0, 0.25, 0.50, 0.75, 1.0]

# Mostra 3 diverse traiettorie
for idx in range(2):
    for t_idx, t in enumerate(time_steps):
        ax = axes[idx, min(t_idx, 2)]
        
        if t_idx < 3:
            trajectory_idx = idx * 5  # Prendi traiettorie diverse
            u = train_data[trajectory_idx, t, :]
            ax.plot(x, u, label=f't={time_values[t]}', linewidth=2)
            ax.set_xlabel('x')
            ax.set_ylabel('u(x, t)')
            ax.set_title(f'Trajectory {trajectory_idx}')
            ax.grid(True, alpha=0.3)
            ax.legend()

plt.tight_layout()
plt.savefig('plots/training_data_visualization.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: plots/training_data_visualization.png")
plt.close()

# ============================================
# VISUALIZZA EVOLUZIONE TEMPORALE
# ============================================
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

trajectory_idx = 0
for t_idx, t in enumerate(time_steps):
    u = train_data[trajectory_idx, t, :]
    ax.plot(x, u, label=f't={time_values[t]}', linewidth=2, alpha=0.8)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('u(x, t)', fontsize=12)
ax.set_title(f'Time Evolution of Trajectory {trajectory_idx}', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('plots/trajectory_evolution.png', dpi=150, bbox_inches='tight')
print("✓ Saved: plots/trajectory_evolution.png")
plt.close()

# ============================================
# CARICA MODELLO E FAI PREDIZIONI
# ============================================
print("\n" + "="*50)
print("Loading model and making predictions...")
print("="*50)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FNO1d(modes=16, width=64, in_dim=2, out_dim=1)
model.load_state_dict(torch.load('models/task2_one2one.pt', map_location=device))
model = model.to(device)
model.eval()

# Prepara dati di test
test_data_torch = torch.from_numpy(test_data).type(torch.float32)
x_grid = torch.linspace(0, 1, 128, dtype=torch.float32)

# Prendi alcuni esempi di test
n_examples = 5
test_indices = [0, 32, 64, 96, 127]

fig, axes = plt.subplots(1, n_examples, figsize=(20, 4))
fig.suptitle('Test Set Predictions: Input (t=0) → Prediction vs Ground Truth (t=1)', fontsize=14)

relative_errors = []

with torch.no_grad():
    for idx, test_idx in enumerate(test_indices):
        # Input: t=0
        u_0 = test_data_torch[test_idx, 0, :]
        
        # Ground truth: t=1
        u_true = test_data_torch[test_idx, 4, :]
        
        # Prepara input per il modello
        x_input = torch.stack([x_grid, u_0], dim=-1).unsqueeze(0).to(device)  # (1, 128, 2)
        
        # Predizione
        u_pred = model(x_input).squeeze().cpu()  # (128, 1) -> (128,)
        u_pred = u_pred.squeeze()
        
        # Calcola errore relativo L2
        rel_error = torch.norm(u_pred - u_true) / torch.norm(u_true)
        relative_errors.append(rel_error.item())
        
        # Plot
        ax = axes[idx]
        ax.plot(x, u_0.numpy(), 'k--', label='Input (t=0)', linewidth=2, alpha=0.7)
        ax.plot(x, u_true.numpy(), 'b-', label='True (t=1)', linewidth=2)
        ax.plot(x, u_pred.numpy(), 'r--', label='Pred (t=1)', linewidth=2)
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.set_title(f'Test {test_idx}\nRel L2: {rel_error:.4f}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('plots/test_predictions.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: plots/test_predictions.png")
plt.close()

# ============================================
# STATISTICHE ERRORI
# ============================================
print(f"\nRelative L2 Errors for selected examples:")
for idx, (test_idx, error) in enumerate(zip(test_indices, relative_errors)):
    print(f"  Test sample {test_idx}: {error:.6f}")

print(f"\nMean error (selected samples): {np.mean(relative_errors):.6f}")

# ============================================
# DISTRIBUZIONE ERRORI SU TUTTO IL TEST SET
# ============================================
print("\nComputing errors on full test set...")
all_errors = []

with torch.no_grad():
    for test_idx in range(len(test_data)):
        u_0 = test_data_torch[test_idx, 0, :]
        u_true = test_data_torch[test_idx, 4, :]
        
        x_input = torch.stack([x_grid, u_0], dim=-1).unsqueeze(0).to(device)
        u_pred = model(x_input).squeeze().cpu().squeeze()
        
        rel_error = torch.norm(u_pred - u_true) / torch.norm(u_true)
        all_errors.append(rel_error.item())

all_errors = np.array(all_errors)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(all_errors, bins=30, edgecolor='black', alpha=0.7)
axes[0].axvline(all_errors.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {all_errors.mean():.6f}')
axes[0].set_xlabel('Relative L2 Error')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Errors on Test Set')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Error vs sample index
axes[1].plot(all_errors, 'o-', markersize=3, alpha=0.6)
axes[1].axhline(all_errors.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {all_errors.mean():.6f}')
axes[1].set_xlabel('Test Sample Index')
axes[1].set_ylabel('Relative L2 Error')
axes[1].set_title('Error per Test Sample')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/error_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved: plots/error_analysis.png")
plt.close()

print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)
print(f"Average Relative L2 Error: {all_errors.mean():.6f}")
print(f"Std Dev: {all_errors.std():.6f}")
print(f"Min Error: {all_errors.min():.6f}")
print(f"Max Error: {all_errors.max():.6f}")
print("="*50)
