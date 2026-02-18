import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np
import os


def compute_relative_l2_error(pred, true):
    """
    Calculate the SUM of relative L2 errors for the batch.
    
    Args:
        pred: (batch_size, resolution, 1) - predictions
        true: (batch_size, resolution, 1) - ground truth
    
    Returns:
        float: Sum of relative L2 errors in the batch
    """
    pred = pred.squeeze(-1)  # (batch_size, resolution)
    true = true.squeeze(-1)  # (batch_size, resolution)
    
    # Calculate L2 norm for each sample
    errors = torch.norm(pred - true, dim=1) / torch.norm(true, dim=1)
    return errors.sum().item()


def train_model(model, train_loader, val_data_path, epochs, lr, device, 
                model_save_path, step_size=50, weight_decay=1e-5, 
                test_function=None):
    """
    Generic training loop for FNO models.
    
    Args:
        model: FNO model to train
        train_loader: DataLoader for training
        val_data_path: Path to validation data (for computing error at t=1.0)
        epochs: Number of epochs
        lr: Learning rate
        device: Device to use
        model_save_path: Path to save the trained model
        step_size: Scheduler step size
        weight_decay: Weight decay for optimizer
        test_function: Function to compute validation error (optional)
    
    Returns:
        Trained model
    """
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
    
    print(f"\nTraining model...")
    
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
        
        # Validation
        if test_function is not None and (epoch + 1) % 10 == 0:
            val_error = test_function(model, val_data_path, device)
            print(f'Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.6f}, Val Error (t=1.0) = {val_error:.6f}')
        elif (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.6f}')
        
        scheduler.step()
    
    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    return model


def test_at_t1(model, data_path, device):
    """
    Test the model at t=1.0 starting from t=0.
    
    Args:
        model: FNO model to test
        data_path: Path to test data
        device: Device to use
    
    Returns:
        Average relative L2 error
    """
    test_data = np.load(data_path)  # (N, 5, 128)
    n_test = test_data.shape[0]
    resolution = test_data.shape[2]
    x_grid = np.linspace(0, 1, resolution, dtype=np.float32)
    
    # Input: u(t=0) with Δt=1.0 → target: u(t=1.0)
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
    
    Args:
        model: FNO model to test
        data_path: Path to test data
        device: Device to use
    
    Returns:
        Dictionary of errors at each time
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


def load_model(model, model_path, device):
    """
    Load a pretrained model from disk.

    Args:
        model: Model instance (with correct architecture)
        model_path: Path to saved model weights
        device: Device to load model to

    Returns:
        Loaded model
    """
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded pretrained model from {model_path}")
        return model
    else:
        raise FileNotFoundError(f"Model not found at {model_path}")
