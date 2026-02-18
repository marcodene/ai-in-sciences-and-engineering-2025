import torch

# ============================================
# DEVICE CONFIGURATION
# ============================================
device = torch.device('cpu')
print(f"Using device: {device}")

# ============================================
# TOGGLE: USE PRETRAINED MODELS
# ============================================
USE_PRETRAINED = True  # Set to False to train new models

# ============================================
# DATA PATHS
# ============================================
DATA_PATHS = {
    'train': 'data/task2/data_train_128.npy',
    'val': 'data/task2/data_val_128.npy',
    'test_32': 'data/task2/data_test_32.npy',
    'test_64': 'data/task2/data_test_64.npy',
    'test_96': 'data/task2/data_test_96.npy',
    'test_128': 'data/task2/data_test_128.npy',
    'finetune_train_unknown': 'data/task2/data_finetune_train_unknown_128.npy',
    'finetune_val_unknown': 'data/task2/data_finetune_val_unknown_128.npy',
    'test_unknown': 'data/task2/data_test_unknown_128.npy'
}

# ============================================
# MODEL PATHS
# ============================================
MODEL_PATHS = {
    'task1_one2one': 'models/task1_one2one.pt',
    'task3_all2all': 'models/task3_all2all.pt',
    'task4_finetuned': 'models/task4_finetuned.pt',
    'task4_scratch': 'models/task4_scratch.pt'
}

# ============================================
# TASK 1: ONE2ONE CONFIGURATION
# ============================================
TASK1_CONFIG = {
    'n_train': 1024,
    'n_val': 32,
    'n_test': 128,
    'batch_size': 32,
    'resolution': 128,
    'modes': 16,
    'width': 64,
    'learning_rate': 0.001,
    'epochs': 100,
    'step_size': 50,
    'weight_decay': 1e-5
}

# ============================================
# TASK 3: ALL2ALL CONFIGURATION
# ============================================
TASK3_CONFIG = {
    'batch_size': 32,
    'modes': 16,
    'width': 64,
    'learning_rate': 5e-4,
    'epochs': 250,
    'step_size': 50,
    'weight_decay': 1e-5
}

# ============================================
# TASK 4: FINETUNING CONFIGURATION
# ============================================
TASK4_CONFIG = {
    'batch_size': 8,
    'modes': 16,
    'width': 64,
    'learning_rate_finetune': 5e-5,
    'epochs_finetune': 200,
    'learning_rate_scratch': 1e-3,
    'epochs_scratch': 200,
    'step_size': 30,
    'weight_decay': 1e-5,               # For finetuning
    'weight_decay_scratch': 2e-5 
}
