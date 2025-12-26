"""
Task 4: Finetuning on Unknown Distribution

This script implements all three parts of Task 4:
1. Zero-shot testing (5 points)
2. Finetuning (10 points)
3. Training from scratch (10 bonus points)
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import os

from models.fno_time import FNO1d

from config import device, USE_PRETRAINED, DATA_PATHS, MODEL_PATHS, TASK4_CONFIG
from datasets import All2AllDataset
from utils import (
    compute_relative_l2_error, load_model, test_at_multiple_times
)


print("="*70)
print("TASK 4: Finetuning on Unknown Distribution")
print("="*70)

# ============================================
# LOAD FINETUNING DATASETS
# ============================================
print("\nLoading finetuning datasets...")
finetune_train_dataset = All2AllDataset(DATA_PATHS['finetune_train_unknown'])
finetune_train_loader = DataLoader(
    finetune_train_dataset, 
    batch_size=TASK4_CONFIG['batch_size'], 
    shuffle=True
)

# ============================================
# TASK 4.1: ZERO-SHOT TESTING
# ============================================
print("\n" + "="*70)
print("TASK 4.1: Zero-Shot Testing on Unknown Distribution")
print("="*70)

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
print("This is the model's performance without any finetuning.")

# ============================================
# TASK 4.2: FINETUNING
# ============================================
print("\n" + "="*70)
print("TASK 4.2: Finetuning on Unknown Distribution")
print("="*70)

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
    optimizer = Adam(
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
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{TASK4_CONFIG["epochs_finetune"]}: Train Loss = {train_loss:.6f}')
        
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
print("\n" + "="*70)
print("TASK 4.3 (BONUS): Training from Scratch on Unknown Distribution")
print("="*70)

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
    optimizer = Adam(
        model_scratch.parameters(), 
        lr=TASK4_CONFIG['learning_rate_scratch'], 
        weight_decay=TASK4_CONFIG['weight_decay']
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
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{TASK4_CONFIG["epochs_scratch"]}: Train Loss = {train_loss:.6f}')
        
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
print("\n" + "="*70)
print("FINAL RESULTS SUMMARY")
print("="*70)
print(f"Zero-Shot (no finetuning):       {error_zeroshot:.6f}")
print(f"Finetuned (32 trajectories):     {error_finetuned:.6f}")
print(f"From Scratch (32 trajectories):  {error_scratch:.6f}")
print("\n" + "="*70)

if error_finetuned < error_scratch:
    improvement = (error_scratch - error_finetuned) / error_scratch * 100
    print(f"✓ Transfer Learning is SUCCESSFUL!")
    print(f"  Finetuning is {improvement:.1f}% better than training from scratch.")
else:
    print(f"✗ Transfer Learning is NOT successful for this case.")
    print(f"  Training from scratch performs better.")
print("="*70)
