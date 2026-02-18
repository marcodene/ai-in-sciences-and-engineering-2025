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
# PATHS
# ============================================
PATHS = {
    'checkpoints': 'models',
    'plots_task1_1': 'plots/task1_1',
    'plots_task1_2': 'plots/task1_2',
    'plots_task1_3': 'plots/task1_3',
    'results': 'results'
}

# ============================================
# TASK 1.1: DATA GENERATION
# ============================================
TASK1_1_CONFIG = {
    'N': 64,  # Grid resolution
    'K_values': [1, 4, 8, 16],  # Frequency parameters for visualization
    'n_samples': 3  # Number of samples per K value
}

# ============================================
# TASK 1.2: TRAINING CONFIGURATION
# ============================================
TASK1_2_CONFIG = {
    'N': 64,  # Grid resolution
    'K_values': [1, 4, 16],  # Complexity levels: Low, Medium, High
    'seed': 0,  # Fixed seed for reproducible coefficients
    
    # Network architecture (3-4 hidden layers as per task description)
    'hidden_dim': 256,
    'n_layers': 4,  # Number of hidden layers
    
    # PINN specific
    'n_collocation': 20000,  # Number of collocation points
    'lambda_bc': 400.0,  # Boundary condition weight
    
    # Training - Adam phase
    'epochs_adam': 5000,
    'lr_adam': 1e-3,
    'print_every_adam': 200,
    
    # Training - L-BFGS phase (fine-tuning)
    'max_iter_lbfgs': 10000,
    
    # Data-Driven specific (may need different settings)
    'epochs_adam_dd': 2000,
    'lr_adam_dd': 1e-4,
    'print_every_adam_dd': 400
}

# ============================================
# TASK 1.3: LOSS LANDSCAPE CONFIGURATION
# ============================================
TASK1_3_CONFIG = {
    'N': 64,  # Grid resolution (must match training)
    'K_values': [1, 4, 16],  # Same as Task 1.2
    'model_types': ['PINN', 'DataDriven'],

    # Li et al. 2018 standard range
    'alpha_range': (-1, 1),
    'beta_range': (-1, 1),

    # Grid resolution options:
    'n_points': 64,

    # Visualization options
    'use_log_scale': False,  # True for log scale, False for linear
    'z_min': None,  # Minimum z-axis value (None for auto)
    'z_max': 5000,  # Maximum z-axis value (None for auto)

    # Direction normalization
    'norm': 'filter',  # 'filter' or 'layer'
    'ignore': 'biasbn',  # Ignore bias and batch norm parameters

    # Caching configuration
    'cache_mode': 'use',  # 'use': use cache if available, 'recompute': force recompute, 'disabled': no caching
    'cache_dir': 'plots/task1_3/cache'
}
