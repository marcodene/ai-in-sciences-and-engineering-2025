import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import os

from models.fno_time import FNO1d

from config import device, USE_PRETRAINED, DATA_PATHS, MODEL_PATHS, TASK4_CONFIG
from datasets import All2AllDataset
from utils import (
    compute_relative_l2_error, load_model, test_at_multiple_times
)


print("\nTASK 4: Finetuning on Unknown Distribution")

# ============================================
# LOAD FINETUNING DATASETS
# ============================================
print("\nLoading finetuning datasets...")
finetune_train_dataset = All2AllDataset(DATA_PATHS['finetune_train_unknown'])
finetune_val_dataset = All2AllDataset(DATA_PATHS['finetune_val_unknown'])
finetune_train_loader = DataLoader(
    finetune_train_dataset,
    batch_size=TASK4_CONFIG['batch_size'],
    shuffle=True
)
finetune_val_loader = DataLoader(
    finetune_val_dataset,
    batch_size=TASK4_CONFIG['batch_size'],
    shuffle=False
)

# ============================================
# TASK 4.1: ZERO-SHOT TESTING
# ============================================
print("\nTASK 4.1: Zero-Shot Testing on Unknown Distribution")

# Load pretrained model from Task 3
model_pretrained = FNO1d(
    modes=TASK4_CONFIG['modes'], 
    width=TASK4_CONFIG['width'], 
    in_dim=3, 
    out_dim=1
).to(device)

if os.path.exists(MODEL_PATHS['task3_all2all']):
    model_pretrained = load_model(model_pretrained, MODEL_PATHS['task3_all2all'], device)
else:
    print(f"ERROR: Pretrained model not found at {MODEL_PATHS['task3_all2all']}")
    print("Please run task3_all2all.py first to train the model!")
    exit(1)

# Test on unknown distribution (zero-shot)
errors_zeroshot = test_at_multiple_times(model_pretrained, DATA_PATHS['test_unknown'], device)
error_zeroshot = errors_zeroshot[1.0]
print(f"\nZero-Shot Error on Unknown Distribution (t=1.0): {error_zeroshot:.6f}")

# ============================================
# TASK 4.2: FINETUNING
# ============================================
print("\nTASK 4.2: Finetuning on Unknown Distribution")

# Create a copy of the pretrained model for finetuning
model_finetuned = FNO1d(
    modes=TASK4_CONFIG['modes'], 
    width=TASK4_CONFIG['width'], 
    in_dim=3, 
    out_dim=1
).to(device)
model_finetuned.load_state_dict(model_pretrained.state_dict())

if USE_PRETRAINED and os.path.exists(MODEL_PATHS['task4_finetuned']):
    model_finetuned = load_model(model_finetuned, MODEL_PATHS['task4_finetuned'], device)
else:
    if USE_PRETRAINED:
        print(f"\nWARNING: Pretrained model not found at {MODEL_PATHS['task4_finetuned']}")
        print("Finetuning new model instead...\n")
    
    # Finetune the model
    criterion = nn.MSELoss()
    optimizer = AdamW(
        model_finetuned.parameters(),
        lr=TASK4_CONFIG['learning_rate_finetune'],
        weight_decay=TASK4_CONFIG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=TASK4_CONFIG['step_size'], 
        gamma=0.5
    )
    
    print("\nFinetuning model...")
    for epoch in range(TASK4_CONFIG['epochs_finetune']):
        # Training
        model_finetuned.train()
        train_loss = 0

        for inputs, targets in finetune_train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model_finetuned(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(finetune_train_loader)

        # Validation
        model_finetuned.eval()
        val_error = 0

        with torch.no_grad():
            for inputs, targets in finetune_val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model_finetuned(inputs)
                val_error += compute_relative_l2_error(outputs, targets)

        val_error /= len(finetune_val_dataset)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{TASK4_CONFIG["epochs_finetune"]}: '
                  f'Train Loss = {train_loss:.6f}, '
                  f'Val Rel L2 Error = {val_error:.6f}')

        scheduler.step()
    
    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model_finetuned.state_dict(), MODEL_PATHS['task4_finetuned'])
    print(f"Model saved to {MODEL_PATHS['task4_finetuned']}")

# Test finetuned model
errors_finetuned = test_at_multiple_times(model_finetuned, DATA_PATHS['test_unknown'], device)
error_finetuned = errors_finetuned[1.0]
print(f"\nFinetuned Model Error on Unknown Distribution (t=1.0): {error_finetuned:.6f}")
print(f"Improvement: {error_zeroshot - error_finetuned:.6f} "
      f"({((error_zeroshot - error_finetuned) / error_zeroshot * 100):.1f}% reduction)")

# ============================================
# TASK 4.3: TRAIN FROM SCRATCH (BONUS)
# ============================================
print("\nTASK 4.3 (BONUS): Training from Scratch on Unknown Distribution")

# Create new model from scratch
model_scratch = FNO1d(
    modes=TASK4_CONFIG['modes'], 
    width=TASK4_CONFIG['width'], 
    in_dim=3, 
    out_dim=1
).to(device)

if USE_PRETRAINED and os.path.exists(MODEL_PATHS['task4_scratch']):
    model_scratch = load_model(model_scratch, MODEL_PATHS['task4_scratch'], device)
else:
    if USE_PRETRAINED:
        print(f"\nWARNING: Pretrained model not found at {MODEL_PATHS['task4_scratch']}")
        print("Training from scratch instead...\n")
    
    # Train from scratch
    criterion = nn.MSELoss()
    optimizer = AdamW(
        model_scratch.parameters(),
        lr=TASK4_CONFIG['learning_rate_scratch'],
        weight_decay=TASK4_CONFIG['weight_decay_scratch']
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=TASK4_CONFIG['step_size'], 
        gamma=0.5
    )
    
    print("\nTraining from scratch...")
    for epoch in range(TASK4_CONFIG['epochs_scratch']):
        # Training
        model_scratch.train()
        train_loss = 0

        for inputs, targets in finetune_train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model_scratch(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(finetune_train_loader)

        # Validation
        model_scratch.eval()
        val_error = 0

        with torch.no_grad():
            for inputs, targets in finetune_val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model_scratch(inputs)
                val_error += compute_relative_l2_error(outputs, targets)

        val_error /= len(finetune_val_dataset)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{TASK4_CONFIG["epochs_scratch"]}: '
                  f'Train Loss = {train_loss:.6f}, '
                  f'Val Rel L2 Error = {val_error:.6f}')

        scheduler.step()
    
    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model_scratch.state_dict(), MODEL_PATHS['task4_scratch'])
    print(f"Model saved to {MODEL_PATHS['task4_scratch']}")

# Test model trained from scratch
errors_scratch = test_at_multiple_times(model_scratch, DATA_PATHS['test_unknown'], device)
error_scratch = errors_scratch[1.0]
print(f"\nFrom-Scratch Model Error on Unknown Distribution (t=1.0): {error_scratch:.6f}")

# ============================================
# FINAL COMPARISON
# ============================================
print("\nFINAL RESULTS SUMMARY")
print(f"Zero-Shot (no finetuning):       {error_zeroshot:.6f}")
print(f"Finetuned (32 trajectories):     {error_finetuned:.6f}")
print(f"From Scratch (32 trajectories):  {error_scratch:.6f}")

if error_finetuned < error_scratch:
    improvement = (error_scratch - error_finetuned) / error_scratch * 100
    print(f"\nTransfer Learning: Finetuning is {improvement:.1f}% better than training from scratch.")
else:
    print(f"\nTransfer Learning: Training from scratch performs better.")
