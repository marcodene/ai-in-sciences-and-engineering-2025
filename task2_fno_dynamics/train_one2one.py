import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from models.fno import FNO1d
import os

# ============================================
# PARAMETRI
# ============================================
n_train = 1024
n_val = 32
n_test = 128
batch_size = 32
resolution = 128

# ============================================
# CARICAMENTO E PREPARAZIONE DATI
# ============================================
train_data_path = "data/task2/data_train_128.npy"
val_data_path = "data/task2/data_val_128.npy"
test_data_path = {
    32 : "data/task2/data_test_32.npy",
    64 : "data/task2/data_test_64.npy",
    96 : "data/task2/data_test_96.npy",
    128 : "data/task2/data_test_128.npy"
}

def prepare_data(data_path, n_samples, resolution):
    """Prepara dati per one2one: t=0 -> t=1"""
    data = torch.from_numpy(np.load(data_path)).type(torch.float32)

    X = data[:n_samples, 0, :]  # (n_samples, resolution) - initial conditions
    Y = data[:n_samples, 4, :]  # (n_samples, resolution) - solution at t=1
    
    # Crea griglia spaziale
    x_grid = torch.linspace(0, 1, resolution, dtype=torch.float32) # (resolution,)
    # Replica la griglia per tutti i samples
    x_grid_rep = x_grid.unsqueeze(0).repeat(n_samples, 1)  # (n_samples, resolution)
    
    # Stack: coordinate + valori -> (n_samples, resolution, 2)
    X = torch.stack([x_grid_rep, X], dim=-1)
    
    # Aggiungi dimensione canale all'output -> (n_samples, resolution, 1)
    Y = Y.unsqueeze(-1)
    
    return X, Y

resolution_train = 128
X_train, Y_train = prepare_data(train_data_path, n_train, resolution_train)
X_val, Y_val = prepare_data(val_data_path, n_val, resolution_train)
X_test, Y_test = prepare_data(test_data_path[resolution_train], n_test, resolution_train)

print(f"X_train shape: {X_train.shape}")  # (1024, 128, 2)
print(f"Y_train shape: {Y_train.shape}")  # (1024, 128, 1)
print(f"X_val shape: {X_val.shape}")      # (32, 128, 2)
print(f"Y_val shape: {Y_val.shape}")      # (32, 128, 1)
print(f"X_test shape: {X_test.shape}")    # (128, 128, 2)
print(f"Y_test shape: {Y_test.shape}")    # (128, 128, 1)

# ============================================
# DATASET E DATALOADER
# ============================================
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_val, Y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = TensorDataset(X_test, Y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"\nTrain batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# ============================================
# MODELLO
# ============================================
modes = 16
width = 64

model = FNO1d(modes=modes, width=width, in_dim=2, out_dim=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print(f"\nDevice: {device}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

# ============================================
# TRAINING
# ============================================
learning_rate = 0.001
epochs = 100
step_size = 50

criterion = torch.nn.MSELoss()
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)

def compute_relative_l2_error(pred, true):
    """Calcola la SOMMA degli errori relativi L2 per il batch.
    
    Args:
        pred: (batch_size, resolution, 1) - predictions
        true: (batch_size, resolution, 1) - ground truth
    
    Returns:
        float: Somma degli errori relativi L2 nel batch
    """
    pred = pred.squeeze(-1)  # (batch_size, resolution)
    true = true.squeeze(-1)  # (batch_size, resolution)
    
    # Calcola norma L2 per ogni sample
    errors = torch.norm(pred - true, dim=1) / torch.norm(true, dim=1)
    return errors.sum().item() 

for epoch in range(epochs):
    # Training
    model.train()
    train_loss = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validation
    model.eval()
    val_error = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_error += compute_relative_l2_error(outputs, targets)  # ← Somma diretta
    
    val_error /= n_val  # ← Dividi per numero totale samples
    # Learning rate scheduling
    scheduler.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.6f}, Val Rel L2 Error = {val_error:.6f}')

os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/task2_one2one.pt')

# ============================================
# TESTING
# ============================================
def test_on_resolution(model, data_path, resolution, device):
    X, Y = prepare_data(data_path, n_test, resolution)

    test_dataset = TensorDataset(X, Y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_predictions = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for(inputs, targets) in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_predictions.append(outputs.cpu())
            all_targets.append(targets)

    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Calcola errore totale e dividi per numero samples
    error = compute_relative_l2_error(predictions, targets) / n_test

    return error


test_error = test_on_resolution(model, test_data_path[128], 128, device)

print(f'\n{"="*50}')
print(f'TASK 1 RESULT: Average Relative L2 Error on Test Set = {test_error:.6f}')
print(f'{"="*50}')


# ============================================
# TASK 2.2: TESTING ON DIFFERENT RESOLUTIONS
# ============================================
print(f'\n\n{"="*50}')
print(f'TASK 2: Testing on Different Resolutions')
print(f'{"="*50}')

resolutions = [32, 64, 96, 128]
errors = {}

for res in resolutions:
    data_path = test_data_path[res]
    errors[res] = test_on_resolution(model, data_path, res, device)
    print(f'Resolution {res}: Error = {errors[res]:.6f}')

print(f'{"="*50}')