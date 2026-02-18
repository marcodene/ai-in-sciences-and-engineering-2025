import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from physics import create_grid, generate_coefficients, compute_source_and_solution


# ============================================================================
# MODEL SAVE/LOAD
# ============================================================================

def save_model(solver, method_name, K, save_dir='checkpoints'):
    """
    Save trained model weights and training history

    Args:
        solver: Trained solver (PoissonPINN or DataDrivenSolver)
        method_name: 'PINN' or 'DataDriven'
        K: Frequency parameter
        save_dir: Directory to save checkpoints
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save model weights
    filepath = f"{save_dir}/{method_name}_K{K}.pt"
    torch.save(solver.model.state_dict(), filepath)

    # Save training history
    history_filepath = f"{save_dir}/{method_name}_K{K}_history.pt"
    torch.save(solver.history, history_filepath)

    print(f"Saved: {filepath}")
    print(f"Saved history: {history_filepath}")


def load_model(solver, method_name, K, save_dir='checkpoints'):
    """
    Load trained weights and training history into an existing solver

    Args:
        solver: Solver instance with correct architecture
        method_name: 'PINN' or 'DataDriven'
        K: Frequency parameter
        save_dir: Directory containing checkpoints

    Returns:
        solver: Solver with loaded weights and history
    """
    filepath = f"{save_dir}/{method_name}_K{K}.pt"

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model not found: {filepath}")

    # Load model weights
    solver.model.load_state_dict(torch.load(filepath))
    print(f"Loaded: {filepath}")

    # Load training history if available
    history_filepath = f"{save_dir}/{method_name}_K{K}_history.pt"
    if os.path.exists(history_filepath):
        solver.history = torch.load(history_filepath)
        print(f"Loaded history: {history_filepath} ({len(solver.history)} iterations)")
    else:
        print(f"No history file found (will skip loss plot)")
        solver.history = []

    return solver


# ============================================================================
# PLOTTING FUNCTIONS - TASK 1.1
# ============================================================================

def plot_samples(N, K_values, n_samples, save_dir):
    """
    Generate and save sample visualizations for Task 1.1
    
    Args:
        N: Grid resolution
        K_values: List of K values to visualize
        n_samples: Number of samples per K value
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for K in K_values:
        fig, axes = plt.subplots(n_samples, 2, figsize=(10, 4*n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        fig.suptitle(f'Samples with K = {K}', fontsize=16, fontweight='bold')
        
        for sample_idx in range(n_samples):
            # Generate sample using physics module
            a_ij = generate_coefficients(K, seed=sample_idx)
            X, Y = create_grid(N)
            f, u = compute_source_and_solution(X, Y, a_ij, K)
            
            # Plot source term
            ax_f = axes[sample_idx, 0]
            im_f = ax_f.imshow(f, extent=[0, 1, 0, 1], origin='lower',
                              cmap='plasma', aspect='equal')
            ax_f.set_xlabel('x')
            ax_f.set_ylabel('y')
            ax_f.set_title(f'Source f (sample {sample_idx+1})')
            plt.colorbar(im_f, ax=ax_f, fraction=0.046, pad=0.04)
            
            # Plot solution
            ax_u = axes[sample_idx, 1]
            im_u = ax_u.imshow(u, extent=[0, 1, 0, 1], origin='lower',
                              cmap='plasma', aspect='equal')
            ax_u.set_xlabel('x')
            ax_u.set_ylabel('y')
            ax_u.set_title(f'Solution u (sample {sample_idx+1})')
            plt.colorbar(im_u, ax=ax_u, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # Save with high resolution
        save_path = os.path.join(save_dir, f'samples_K_{K}.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close(fig)


def plot_f_u_comparison(N, K_values, seed=42, save_dir=None, name_suffix=""):
    """
    Generate a plot showing source f and solution u for different K values.
    
    Layout: 2 rows (f and u) x 4 columns (one per K)
    
    Args:
        N: Grid resolution
        K_values: List of K values to visualize
        seed: Random seed for coefficient generation (default: 42)
        save_dir: Directory to save plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    n_cols = len(K_values)
    fig, axes = plt.subplots(2, n_cols, figsize=(4.4*n_cols, 8), gridspec_kw={'hspace': 0.15, 'wspace': 0.2})
    
    for idx, K in enumerate(K_values):
        # Generate data for this K
        a_ij = generate_coefficients(K, seed=seed)
        X, Y = create_grid(N)
        f, u = compute_source_and_solution(X, Y, a_ij, K)
        
        # Plot source term f (first row)
        ax_f = axes[0, idx]
        im_f = ax_f.imshow(f, extent=[0, 1, 0, 1], origin='lower',
                          cmap='plasma', aspect='equal')
        #ax_f.set_xlabel('x')
        #ax_f.set_ylabel('y')
        ax_f.set_title(f'Source f (K={K})', fontsize=18)
        ax_f.set_xticks([])
        ax_f.set_yticks([])
        plt.colorbar(im_f, ax=ax_f, fraction=0.046, pad=0.04)
        
        # Plot solution u (second row)
        ax_u = axes[1, idx]
        im_u = ax_u.imshow(u, extent=[0, 1, 0, 1], origin='lower',
                          cmap='plasma', aspect='equal')
        #ax_u.set_xlabel('x')
        #ax_u.set_ylabel('y')
        ax_u.set_title(f'Solution u (K={K})', fontsize=18)
        ax_u.set_xticks([])
        ax_u.set_yticks([])
        plt.colorbar(im_u, ax=ax_u, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, f'f_u_comparison{name_suffix}.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)


def plot_pinn_vs_datadriven(pinn_solver, dd_solver, K, save_dir):
    """
    Compare PINN and DataDriven predictions on the same input.
    
    Layout: 2 rows x 2 columns
    - Left column: Input (Source f) and Exact solution
    - Right column: PINN prediction and DataDriven prediction
    
    Args:
        pinn_solver: Trained PINN solver
        dd_solver: Trained DataDriven solver
        K: Frequency parameter (for title)
        save_dir: Directory to save plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get predictions
    u_pinn = pinn_solver.predict()
    u_dd = dd_solver.predict()
    u_exact = pinn_solver.u_grid  # Both solvers have the same exact solution
    f_source = pinn_solver.f_grid  # Both solvers have the same source term
    
    # Compute errors
    l2_error_pinn = pinn_solver.compute_l2_error()
    l2_error_dd = dd_solver.compute_l2_error()
    
    # Create figure: 2 rows x 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw={'hspace': 0.1, 'wspace': 0.3})
    
    # Top-left: Source term f
    im0 = axes[0, 0].imshow(f_source, extent=[0, 1, 0, 1], origin='lower', cmap='plasma', aspect='equal')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].set_title('Source f')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # Top-right: PINN prediction
    im1 = axes[0, 1].imshow(u_pinn, extent=[0, 1, 0, 1], origin='lower', cmap='plasma', aspect='equal')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    axes[0, 1].set_title(f'PINN Prediction\n(L2 error: {l2_error_pinn:.6f})')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Bottom-left: Exact solution
    im2 = axes[1, 0].imshow(u_exact, extent=[0, 1, 0, 1], origin='lower', cmap='plasma', aspect='equal')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].set_title('Exact Solution')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Bottom-right: DataDriven prediction
    im3 = axes[1, 1].imshow(u_dd, extent=[0, 1, 0, 1], origin='lower', cmap='plasma', aspect='equal')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    axes[1, 1].set_title(f'DataDriven Prediction\n(L2 error: {l2_error_dd:.6f})')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    #fig.suptitle(f'PINN vs DataDriven Comparison (K={K})', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, f'pinn_vs_datadriven_K{K}.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)


# ============================================================================
# PLOTTING FUNCTIONS - TASK 1.2
# ============================================================================

def plot_solver_results(solver, title, save_dir):
    """
    Plot and save solver prediction results
    
    Args:
        solver: Trained solver
        title: Plot title
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    u_pred = solver.predict()
    error = np.abs(u_pred - solver.u_grid)
    l2_error = solver.compute_l2_error()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Source term
    im0 = axes[0, 0].imshow(solver.f_grid, extent=[0, 1, 0, 1], 
                            origin='lower', cmap='plasma')
    axes[0, 0].set_title('Source f', fontsize=12)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # True solution
    im1 = axes[0, 1].imshow(solver.u_grid, extent=[0, 1, 0, 1], 
                            origin='lower', cmap='plasma')
    axes[0, 1].set_title('True u', fontsize=12)
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Predicted solution
    im2 = axes[1, 0].imshow(u_pred, extent=[0, 1, 0, 1], 
                            origin='lower', cmap='plasma')
    axes[1, 0].set_title(f'Predicted u (L2 error: {l2_error:.6f})', fontsize=12)
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Absolute error
    im3 = axes[1, 1].imshow(error, extent=[0, 1, 0, 1], 
                            origin='lower', cmap='Reds')
    axes[1, 1].set_title('Absolute Error |u_pred - u_true|', fontsize=12)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1, 1])
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save with high resolution
    save_path = os.path.join(save_dir, f'{title.replace(" ", "_")}_prediction.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)


def plot_loss_history(solver, title, save_dir):
    """
    Plot and save training loss history

    Args:
        solver: Trained solver with history
        title: Plot title
        save_dir: Directory to save plots
    """
    # Skip if no history available
    if not solver.history or len(solver.history) == 0:
        print(f"No training history available for {title}, skipping loss plot")
        return

    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.semilogy(solver.history, linewidth=1.5)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Total Loss', fontsize=11)
    ax.set_title(f'{title} - Training Loss', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save with high resolution
    save_path = os.path.join(save_dir, f'{title.replace(" ", "_")}_loss.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)


def plot_all_loss_histories(K_values, method_name, save_dir, checkpoints_dir='models'):
    """
    Plot all loss histories for a given method (PINN or DataDriven) across different K values.
    
    Args:
        K_values: List of K values to plot
        method_name: 'PINN' or 'DataDriven'
        save_dir: Directory to save the plot
        checkpoints_dir: Directory containing the model history files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Colors from plasma colormap at 0%, 50%, 100%
    from matplotlib import cm
    plasma = cm.get_cmap('plasma')
    colors = [plasma(0.2), plasma(0.6), plasma(0.85)]
    
    for idx, K in enumerate(K_values):
        history_filepath = f"{checkpoints_dir}/{method_name}_K{K}_history.pt"
        
        if not os.path.exists(history_filepath):
            print(f"History not found: {history_filepath}")
            continue
        
        # Load history
        import torch
        history = torch.load(history_filepath)
        
        if not history or len(history) == 0:
            print(f"Empty history for {method_name} K={K}")
            continue
        
        # Plot with different color for each K
        color = colors[idx % len(colors)]
        ax.semilogy(history, linewidth=2, label=f'K={K}', color=color)
        print(f"Loaded history for {method_name} K={K}: {len(history)} iterations")
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Total Loss', fontsize=12)
    ax.set_title(f'{method_name} Training Loss Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, f'{method_name}_all_loss_histories.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)


# ============================================================================
# LOSS LANDSCAPE FUNCTIONS - TASK 1.3
# ============================================================================

def get_random_directions(model, norm='filter', ignore='biasbn'):
    """
    Generate two random normalized directions (δ and η) following Li et al. 2018
    
    Args:
        model: Neural network model
        norm: Normalization method ('filter' or 'layer')
        ignore: Which parameters to ignore ('biasbn' to ignore bias and batch norm)
    
    Returns:
        delta, eta: Two random direction vectors
    """
    directions = []
    
    for _ in range(2):
        direction = []
        
        # Generate random direction for each parameter
        for param in model.parameters():
            d = torch.randn_like(param)
            direction.append(d)
        
        # Ignore bias and batch norm
        if ignore == 'biasbn':
            for d, param in zip(direction, model.parameters()):
                if param.dim() <= 1:
                    d.fill_(0)
        
        # Filter normalization
        if norm == 'filter':
            for d, param in zip(direction, model.parameters()):
                if d.numel() == 0 or torch.all(d == 0):
                    continue
                
                if param.dim() == 2:  # Linear layers
                    for i in range(param.shape[0]):
                        neuron_norm = torch.norm(param[i]) + 1e-10
                        d_neuron_norm = torch.norm(d[i]) + 1e-10
                        d[i] = d[i] * (neuron_norm / d_neuron_norm)
                
                elif param.dim() == 4:  # Conv layers
                    for i in range(param.shape[0]):
                        filter_norm = torch.norm(param[i]) + 1e-10
                        d_filter_norm = torch.norm(d[i]) + 1e-10
                        d[i] = d[i] * (filter_norm / d_filter_norm)
        
        directions.append(direction)
    
    return directions[0], directions[1]


def set_weights(model, base_weights, directions=None, alpha=0.0, beta=0.0):
    """
    Set model weights: θ = θ* + α·δ + β·η

    Args:
        model: Neural network model
        base_weights: Base weights θ*
        directions: Tuple of (delta, eta) or None
        alpha: Step size in δ direction
        beta: Step size in η direction
    """
    if directions is None:
        for p, w in zip(model.parameters(), base_weights):
            p.data.copy_(w)
    else:
        delta, eta = directions
        for p, w, d, e in zip(model.parameters(), base_weights, delta, eta):
            p.data = w + alpha * d + beta * e


# ============================================================================
# HDF5 CACHING FOR LOSS LANDSCAPES
# ============================================================================

def get_cache_path(model_type, K, cache_dir):
    """
    Generate consistent cache file path

    Args:
        model_type: 'PINN' or 'DataDriven'
        K: Frequency parameter
        cache_dir: Directory for cache files

    Returns:
        filepath: Path to HDF5 cache file
    """
    return os.path.join(cache_dir, f'{model_type}_K{K}_landscape.h5')


def validate_cache(filepath, config):
    """
    Check if cached data is compatible with current configuration

    Args:
        filepath: Path to HDF5 cache file
        config: Current configuration dictionary

    Returns:
        valid: True if cache is valid, False otherwise
    """
    try:
        import h5py

        with h5py.File(filepath, 'r') as f:
            # Check if required datasets exist
            if not all(key in f for key in ['alpha', 'beta', 'losses']):
                return False

            # Check metadata compatibility
            if 'metadata' in f.attrs:
                metadata = dict(f.attrs['metadata'])

                # Verify grid resolution
                if metadata.get('n_points') != config['n_points']:
                    return False

                # Verify alpha/beta ranges
                if (metadata.get('alpha_range') != config['alpha_range'] or
                    metadata.get('beta_range') != config['beta_range']):
                    return False

            return True

    except Exception as e:
        print(f"      Cache validation failed: {e}")
        return False


def save_landscape_to_h5(filepath, Alpha, Beta, losses, delta, eta,
                         model_type, K, config):
    """
    Save loss landscape data to HDF5 file

    Args:
        filepath: Path to save HDF5 file
        Alpha: 2D meshgrid array for alpha values
        Beta: 2D meshgrid array for beta values
        losses: 2D array of loss values
        delta: List of direction tensors (first random direction)
        eta: List of direction tensors (second random direction)
        model_type: 'PINN' or 'DataDriven'
        K: Frequency parameter
        config: Configuration dictionary
    """
    try:
        import h5py
        from datetime import datetime

        # Create cache directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with h5py.File(filepath, 'w') as f:
            # Save meshgrids and losses
            f.create_dataset('alpha', data=Alpha, compression='gzip')
            f.create_dataset('beta', data=Beta, compression='gzip')
            f.create_dataset('losses', data=losses, compression='gzip')

            # Save random directions
            delta_group = f.create_group('directions/delta')
            eta_group = f.create_group('directions/eta')

            for i, (d_tensor, e_tensor) in enumerate(zip(delta, eta)):
                delta_group.create_dataset(f'param_{i}',
                                          data=d_tensor.detach().cpu().numpy(),
                                          compression='gzip')
                eta_group.create_dataset(f'param_{i}',
                                        data=e_tensor.detach().cpu().numpy(),
                                        compression='gzip')

            # Save metadata as attributes
            metadata = {
                'model_type': model_type,
                'K': K,
                'n_points': config['n_points'],
                'alpha_range': config['alpha_range'],
                'beta_range': config['beta_range'],
                'norm': config['norm'],
                'ignore': config['ignore'],
                'timestamp': datetime.now().isoformat()
            }

            # Store metadata
            for key, value in metadata.items():
                f.attrs[key] = str(value) if not isinstance(value, (int, float)) else value

        print(f"      Cached to: {filepath}")

    except ImportError:
        print(f"       h5py not installed. Install with: pip install h5py")
        print(f"      Skipping cache save...")
    except Exception as e:
        print(f"       Failed to save cache: {e}")


def load_landscape_from_h5(filepath, model):
    """
    Load loss landscape data from HDF5 file

    Args:
        filepath: Path to HDF5 cache file
        model: Neural network model (used to reconstruct direction tensors)

    Returns:
        Alpha: 2D meshgrid array for alpha values
        Beta: 2D meshgrid array for beta values
        losses: 2D array of loss values
        delta: List of direction tensors (first random direction)
        eta: List of direction tensors (second random direction)
    """
    try:
        import h5py

        with h5py.File(filepath, 'r') as f:
            # Load meshgrids and losses
            Alpha = f['alpha'][:]
            Beta = f['beta'][:]
            losses = f['losses'][:]

            # Reconstruct random directions
            delta = []
            eta = []

            delta_group = f['directions/delta']
            eta_group = f['directions/eta']

            # Get parameter count from model
            n_params = len(list(model.parameters()))

            for i in range(n_params):
                d_array = delta_group[f'param_{i}'][:]
                e_array = eta_group[f'param_{i}'][:]

                delta.append(torch.tensor(d_array, dtype=torch.float32))
                eta.append(torch.tensor(e_array, dtype=torch.float32))

        return Alpha, Beta, losses, delta, eta

    except ImportError:
        raise ImportError("h5py not installed. Install with: pip install h5py")
    except Exception as e:
        raise RuntimeError(f"Failed to load cache: {e}")


# ============================================================================
# COMBINED LANDSCAPE PLOTTING FUNCTIONS
# ============================================================================

def plot_all_contours_combined(landscapes_data, K_values, save_path, use_log=False, z_min=None, z_max=None):
    """
    Create combined 2D contour plot for all models and K values
    
    Layout: 2 rows x 3 columns
    - Row 1: PINN for K=1, K=4, K=16
    - Row 2: DataDriven for K=1, K=4, K=16
    
    Args:
        landscapes_data: Dict with structure {model_type: {K: (Alpha, Beta, losses)}}
        K_values: List of K values (e.g., [1, 4, 16])
        save_path: Path to save the figure
        use_log: Boolean for logarithmic scale
        z_min, z_max: Optional limits for z-axis (None = auto)
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    # Create figure with 2 rows x 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    model_types = ['PINN', 'DataDriven']
    
    for row, model_type in enumerate(model_types):
        for col, K in enumerate(K_values):
            ax = axes[row, col]
            
            if model_type in landscapes_data and K in landscapes_data[model_type]:
                Alpha, Beta, losses = landscapes_data[model_type][K]
                
                # Handle NaN/Inf values
                finite_mask = np.isfinite(losses)
                if not finite_mask.any():
                    ax.text(0.5, 0.5, 'All NaN/Inf', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12)
                    continue
                
                finite_losses = losses[finite_mask]
                
                # Choose scale
                if use_log:
                    # Log scale
                    losses_plot = np.log10(finite_losses.min() + losses)
                    losses_plot[~finite_mask] = np.log10(finite_losses.max()) * 1.2
                    zlabel = 'log₁₀(Loss)'
                    
                    # Apply z-bounds if specified
                    if z_min is not None or z_max is not None:
                        actual_min = z_min if z_min is not None else np.log10(finite_losses.min())
                        actual_max = z_max if z_max is not None else np.log10(finite_losses.max())
                        losses_plot = np.clip(losses_plot, actual_min, actual_max)
                else:
                    # Linear scale
                    losses_plot = losses.copy()
                    max_finite = finite_losses.max()
                    losses_plot[~finite_mask] = max_finite * 1.5
                    
                    # Apply z-bounds if specified, otherwise use percentiles
                    if z_min is not None or z_max is not None:
                        vmin = z_min if z_min is not None else np.percentile(finite_losses, 1)
                        vmax = z_max if z_max is not None else np.percentile(finite_losses, 99)
                    else:
                        vmin = np.percentile(finite_losses, 1)
                        vmax = np.percentile(finite_losses, 99)
                    
                    losses_plot = np.clip(losses_plot, vmin, vmax)
                    zlabel = 'Loss'
                
                # Create contour plot
                contour = ax.contourf(Alpha, Beta, losses_plot, levels=20, cmap='plasma')
                
                # Mark optimal point at origin
                ax.plot(0, 0, 'r*', markersize=15, markeredgecolor='white', 
                       markeredgewidth=1.5, label='θ*')
                
                ax.set_xlabel('α', fontsize=11)
                ax.set_ylabel('β', fontsize=11)
                ax.set_title(f'{model_type} (K={K})', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.set_aspect('equal')
                
                # Add individual colorbar for each subplot
                cbar = plt.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
                #cbar.set_label(zlabel, fontsize=10)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_xlabel('α')
                ax.set_ylabel('β')
                ax.set_title(f'{model_type} (K={K})', fontsize=12)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined contour plot: {save_path}")
    plt.close(fig)


def plot_all_surfaces_combined(landscapes_data, K_values, save_path, use_log=False, z_min=None, z_max=None):
    """
    Create combined 3D surface plot for all models and K values
    
    Layout: 2 rows x 3 columns
    - Row 1: PINN for K=1, K=4, K=16
    - Row 2: DataDriven for K=1, K=4, K=16
    
    Args:
        landscapes_data: Dict with structure {model_type: {K: (Alpha, Beta, losses)}}
        K_values: List of K values (e.g., [1, 4, 16])
        save_path: Path to save the figure
        use_log: Boolean for logarithmic scale
        z_min, z_max: Optional limits for z-axis (None = auto)
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create figure with 2 rows x 3 columns
    fig = plt.figure(figsize=(21, 12))
    
    model_types = ['PINN', 'DataDriven']
    
    for row, model_type in enumerate(model_types):
        for col, K in enumerate(K_values):
            ax = fig.add_subplot(2, 3, row * 3 + col + 1, projection='3d')
            
            if model_type in landscapes_data and K in landscapes_data[model_type]:
                Alpha, Beta, losses = landscapes_data[model_type][K]
                
                # Handle NaN/Inf values
                finite_mask = np.isfinite(losses)
                if not finite_mask.any():
                    ax.text2D(0.5, 0.5, 'All NaN/Inf', ha='center', va='center', 
                             transform=ax.transAxes, fontsize=12)
                    continue
                
                finite_losses = losses[finite_mask]
                
                # Choose scale
                if use_log:
                    # Log scale
                    losses_plot = np.log10(finite_losses.min() + losses)
                    losses_plot[~finite_mask] = np.log10(finite_losses.max()) * 1.2
                    zlabel = 'log₁₀(Loss)'
                    
                    # Apply z-bounds if specified
                    if z_min is not None or z_max is not None:
                        actual_min = z_min if z_min is not None else np.log10(finite_losses.min())
                        actual_max = z_max if z_max is not None else np.log10(finite_losses.max())
                        losses_plot = np.clip(losses_plot, actual_min, actual_max)
                else:
                    # Linear scale
                    losses_plot = losses.copy()
                    max_finite = finite_losses.max()
                    losses_plot[~finite_mask] = max_finite * 1.5
                    
                    # Apply z-bounds if specified, otherwise use percentiles
                    if z_min is not None or z_max is not None:
                        vmin = z_min if z_min is not None else np.percentile(finite_losses, 1)
                        vmax = z_max if z_max is not None else np.percentile(finite_losses, 99)
                    else:
                        vmin = np.percentile(finite_losses, 1)
                        vmax = np.percentile(finite_losses, 99)
                    
                    losses_plot = np.clip(losses_plot, vmin, vmax)
                    zlabel = 'Loss'
                
                # Plot surface
                surf = ax.plot_surface(Alpha, Beta, losses_plot, cmap='plasma',
                                      linewidth=0, antialiased=True, alpha=0.9)
                
                # Mark optimal point
                center_idx = losses_plot.shape[0] // 2
                center_loss = losses_plot[center_idx, center_idx]
                ax.scatter([0], [0], [center_loss], color='red', s=200, marker='*',
                          edgecolors='white', linewidths=2, zorder=10)
                
                ax.set_xlabel('α', fontsize=10)
                ax.set_ylabel('β', fontsize=10)
                ax.set_zlabel(zlabel, fontsize=10)
                ax.zaxis.set_rotate_label(False)
                ax.zaxis.label.set_rotation(90)
                
                ax.text2D(0.5, 0.93, f'{model_type} (K={K})', transform=ax.transAxes, ha='center', va='bottom', fontsize=12)
                ax.view_init(elev=25, azim=45)
                
                # Add individual colorbar for each subplot
                cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.02)
                #cbar.set_label(zlabel, fontsize=9)
            else:
                ax.text2D(0.5, 0.5, 'No data', ha='center', va='center', 
                         transform=ax.transAxes, fontsize=12)
                ax.set_xlabel('α')
                ax.set_ylabel('β')
                ax.set_zlabel('Loss')
                ax.set_title(f'{model_type} (K={K})', fontsize=12)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined 3D surface plot: {save_path}")
    plt.close(fig)
