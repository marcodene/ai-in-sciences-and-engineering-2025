import torch
from torch.utils.data import DataLoader
from models.fno import FNO1d
import os

from config import (
    device, USE_PRETRAINED, DATA_PATHS, MODEL_PATHS, TASK1_CONFIG
)
from datasets import One2OneDataset
from utils import compute_relative_l2_error, train_model, load_model


print("\nTASK 1: One-to-One Training (t=0 to t=1)")

# ============================================
# LOAD DATASETS
# ============================================
print("\nLoading datasets...")
train_dataset = One2OneDataset(
    DATA_PATHS['train'], 
    TASK1_CONFIG['n_train'], 
    TASK1_CONFIG['resolution']
)
val_dataset = One2OneDataset(
    DATA_PATHS['val'], 
    TASK1_CONFIG['n_val'], 
    TASK1_CONFIG['resolution']
)
test_dataset = One2OneDataset(
    DATA_PATHS['test_128'], 
    TASK1_CONFIG['n_test'], 
    TASK1_CONFIG['resolution']
)

train_loader = DataLoader(train_dataset, batch_size=TASK1_CONFIG['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=TASK1_CONFIG['batch_size'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=TASK1_CONFIG['batch_size'], shuffle=False)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# ============================================
# CREATE MODEL
# ============================================
model = FNO1d(
    modes=TASK1_CONFIG['modes'], 
    width=TASK1_CONFIG['width'], 
    in_dim=2,  # [x-coords, u(x, t=0)]
    out_dim=1
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"\nModel: {num_params:,} parameters")
print(f"  Modes: {TASK1_CONFIG['modes']}, Width: {TASK1_CONFIG['width']}")

# ============================================
# TRAIN OR LOAD MODEL
# ============================================
if USE_PRETRAINED and os.path.exists(MODEL_PATHS['task1_one2one']):
    model = load_model(model, MODEL_PATHS['task1_one2one'], device)
else:
    if USE_PRETRAINED:
        print(f"\nWARNING: Pretrained model not found at {MODEL_PATHS['task1_one2one']}")
        print("Training new model instead...\n")
    
    # Training loop
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TASK1_CONFIG['learning_rate'],
        weight_decay=TASK1_CONFIG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=TASK1_CONFIG['step_size'], 
        gamma=0.5
    )
    
    print("\nStarting training...")
    for epoch in range(TASK1_CONFIG['epochs']):
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
        
        # Validation
        model.eval()
        val_error = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_error += compute_relative_l2_error(outputs, targets)
        
        val_error /= TASK1_CONFIG['n_val']
        
        # Learning rate scheduling
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{TASK1_CONFIG["epochs"]}: '
                  f'Train Loss = {train_loss:.6f}, '
                  f'Val Rel L2 Error = {val_error:.6f}')
    
    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATHS['task1_one2one'])
    print(f"\nModel saved to {MODEL_PATHS['task1_one2one']}")

# ============================================
# TESTING
# ============================================
print("\nTesting on test set (resolution 128)")

model.eval()
test_error = 0

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        test_error += compute_relative_l2_error(outputs, targets)

test_error /= TASK1_CONFIG['n_test']

print(f'\nTASK 1 RESULT: Average Relative L2 Error = {test_error:.6f}')
