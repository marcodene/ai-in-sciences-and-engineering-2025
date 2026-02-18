import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os
from tqdm import tqdm

from config_task1 import TASK1_3_CONFIG, TASK1_2_CONFIG, PATHS
from physics import generate_coefficients
from models import PoissonPINN, DataDrivenSolver
from utils_task1 import (
    get_random_directions, set_weights,
    get_cache_path, validate_cache, save_landscape_to_h5, load_landscape_from_h5,
    plot_all_contours_combined, plot_all_surfaces_combined
)


def compute_loss_on_grid(solver, base_weights, delta, eta, alpha_range, beta_range, n_points=41):
    """
    Compute loss on 2D grid: L(θ* + α·δ + β·η)
    
    Args:
        solver: Trained solver
        base_weights: Base weights θ*
        delta, eta: Random direction vectors
        alpha_range: (min, max) for α
        beta_range: (min, max) for β
        n_points: Grid resolution
    
    Returns:
        Alpha, Beta, losses: Meshgrid and loss values
    """
    alphas = np.linspace(alpha_range[0], alpha_range[1], n_points)
    betas = np.linspace(beta_range[0], beta_range[1], n_points)
    losses = np.zeros((len(betas), len(alphas)))
    
    solver.model.eval()
    
    total_points = len(alphas) * len(betas)
    pbar = tqdm(total=total_points, desc="Computing loss landscape")
    
    for i, beta in enumerate(betas):
        for j, alpha in enumerate(alphas):
            set_weights(solver.model, base_weights, (delta, eta), alpha, beta)
            
            with torch.set_grad_enabled(True):
                try:
                    loss_dict = solver.compute_loss()
                    loss_value = loss_dict['total'].item()
                    losses[i, j] = loss_value if np.isfinite(loss_value) else np.nan
                except Exception as e:
                    losses[i, j] = np.nan
            
            solver.model.zero_grad()
            pbar.update(1)
    
    pbar.close()
    set_weights(solver.model, base_weights)
    
    Alpha, Beta = np.meshgrid(alphas, betas)
    return Alpha, Beta, losses


def plot_contour(Alpha, Beta, losses, title, save_path, use_log=False, z_min=None, z_max=None):
    """
    Create 2D contour plot
    
    Args:
        Alpha, Beta: Meshgrid coordinates
        losses: Loss values
        title: Plot title
        save_path: Where to save the plot
        use_log: Use logarithmic scale
        z_min, z_max: Optional z-axis bounds
    """
    # Handle NaN/Inf values
    finite_mask = np.isfinite(losses)
    if not finite_mask.any():
        print(f"\nERROR: All losses are NaN/Inf!")
        return
    
    finite_losses = losses[finite_mask]
    
    # Choose scale
    if use_log:
        # Log scale (better for wide range)
        losses_plot = np.log10(finite_losses.min() + losses)
        losses_plot[~finite_mask] = np.log10(finite_losses.max()) * 1.2
        zlabel = 'log₁₀(Loss)'
        
        # Apply z-bounds if specified
        if z_min is not None or z_max is not None:
            actual_min = z_min if z_min is not None else np.log10(finite_losses.min())
            actual_max = z_max if z_max is not None else np.log10(finite_losses.max())
            losses_plot = np.clip(losses_plot, actual_min, actual_max)
            print(f"      Log scale with bounds: [{actual_min:.2f}, {actual_max:.2f}]")
        else:
            print(f"      Log scale: [{np.log10(finite_losses.min()):.2f}, {np.log10(finite_losses.max()):.2f}]")
    else:
        # Linear scale with clipping
        losses_plot = losses.copy()
        max_finite = finite_losses.max()
        losses_plot[~finite_mask] = max_finite * 1.5
        
        # Apply z-bounds if specified, otherwise use percentiles
        if z_min is not None or z_max is not None:
            vmin = z_min if z_min is not None else np.percentile(finite_losses, 1)
            vmax = z_max if z_max is not None else np.percentile(finite_losses, 99)
            print(f"      Linear scale with bounds: [{vmin:.2e}, {vmax:.2e}]")
        else:
            vmin = np.percentile(finite_losses, 1)
            vmax = np.percentile(finite_losses, 99)
            print(f"      Linear scale: [{finite_losses.min():.2e}, {finite_losses.max():.2e}]")
        
        losses_plot = np.clip(losses_plot, vmin, vmax)
        zlabel = 'Loss'
    
    # Plot
    fig, ax = plt.subplots(figsize=(11, 9))
    
    levels = 30
    contour = ax.contourf(Alpha, Beta, losses_plot, levels=levels, cmap='plasma')
    ax.contour(Alpha, Beta, losses_plot, levels=levels, colors='black', 
               alpha=0.15, linewidths=0.5)
    
    # Mark θ*
    ax.plot(0, 0, 'r*', markersize=25, markeredgecolor='white', 
            markeredgewidth=2.5, label='θ* (trained)', zorder=10)
    
    ax.set_xlabel('α (direction δ)', fontsize=14, fontweight='bold')
    ax.set_ylabel('β (direction η)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_aspect('equal')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.2)
    
    cbar = plt.colorbar(contour, ax=ax, pad=0.02)
    cbar.set_label(zlabel, fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"      Saved: {save_path}")
    plt.close()


def plot_surface_3d(Alpha, Beta, losses, title, save_path, use_log=False, z_min=None, z_max=None):
    """
    Create 3D surface plot
    
    Args:
        Alpha, Beta: Meshgrid coordinates
        losses: Loss values
        title: Plot title
        save_path: Where to save the plot
        use_log: Use logarithmic scale
        z_min, z_max: Optional z-axis bounds
    """
    # Handle NaN/Inf values
    finite_mask = np.isfinite(losses)
    if not finite_mask.any():
        print(f"\nERROR: All losses are NaN/Inf!")
        return
    
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
        # Linear scale with clipping
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
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Surface plot
    surf = ax.plot_surface(Alpha, Beta, losses_plot, cmap='plasma',
                          linewidth=0, antialiased=True, alpha=0.9)
    
    # Mark θ*
    center_idx = losses_plot.shape[0] // 2
    center_loss = losses_plot[center_idx, center_idx]
    ax.scatter([0], [0], [center_loss], color='red', s=200, marker='*',
              edgecolors='white', linewidths=2, label='θ* (trained)', zorder=10)
    
    # Labels and title
    ax.set_xlabel('α (direction δ)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('β (direction η)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_zlabel(zlabel, fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label(zlabel, fontsize=12, fontweight='bold')
    
    # View angle
    ax.view_init(elev=30, azim=45)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"      Saved: {save_path}")
    plt.close()


def process_model(model_type, K, N, a_ij, config, output_dir):
    """
    Process one model: load, compute/load landscape, create plots

    Supports caching via cache_mode:
    - 'use': Load from cache if valid, compute and save otherwise
    - 'recompute': Force recompute and overwrite cache
    - 'disabled': Always compute, never save/load

    Args:
        model_type: 'PINN' or 'DataDriven'
        K: Frequency parameter
        N: Grid resolution
        a_ij: Coefficient matrix
        config: Configuration dictionary
        output_dir: Directory to save plots

    Returns:
        True if successful, False if model not found
    """
    print(f"\nProcessing {model_type} - K={K}")

    # Check if model exists
    model_path = f"{PATHS['checkpoints']}/{model_type}_K{K}.pt"
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print(f"Skipping {model_type} K={K}...")
        return False

    # Create solver and load model
    print(f"Loading {model_type} model...")
    if model_type == 'PINN':
        solver = PoissonPINN(
            N=N, K=K, a_ij=a_ij,
            n_collocation=config['n_collocation'],
            hidden_dim=config['hidden_dim'],
            n_layers=config['n_layers'],
            lambda_bc=config['lambda_bc']
        )
    else:  # DataDriven
        solver = DataDrivenSolver(
            N=N, K=K, a_ij=a_ij,
            hidden_dim=config['hidden_dim'],
            n_layers=config['n_layers']
        )

    solver.model.load_state_dict(torch.load(model_path))
    base_weights = [p.data.clone() for p in solver.model.parameters()]
    print(f"      Loaded from {model_path}")

    # Verify model
    loss = solver.compute_loss()
    print(f"      Loss at θ*: {loss['total'].item():.6e}")

    # Handle caching - compute or load landscape data
    cache_mode = config.get('cache_mode', 'use')
    cache_path = get_cache_path(model_type, K, config['cache_dir'])

    # Decide whether to load from cache or compute
    should_load_cache = False
    should_compute = True
    should_save_cache = False

    if cache_mode == 'use':
        # Try to load from cache
        if os.path.exists(cache_path) and validate_cache(cache_path, config):
            should_load_cache = True
            should_compute = False
        else:
            # Cache miss or invalid - compute and save
            should_compute = True
            should_save_cache = True

    elif cache_mode == 'recompute':
        # Force recompute and update cache
        should_compute = True
        should_save_cache = True
        if os.path.exists(cache_path):
            print(f"Cache mode: recompute (will overwrite existing cache)")

    elif cache_mode == 'disabled':
        # No caching
        should_compute = True
        should_save_cache = False
        print(f"Cache mode: disabled (no caching)")

    else:
        print(f" Unknown cache_mode: {cache_mode}. Using 'disabled'.")
        should_compute = True
        should_save_cache = False

    # Load from cache or compute
    if should_load_cache:
        print(f"Loading from cache...")
        try:
            Alpha, Beta, losses, delta, eta = load_landscape_from_h5(cache_path, solver.model)
            print(f"      Loaded from: {cache_path}")
        except Exception as e:
            print(f"       Cache load failed: {e}")
            print(f"      Falling back to computation...")
            should_compute = True
            should_save_cache = (cache_mode == 'use')

    if should_compute:
        # Generate random directions
        print(f"Generating random directions...")
        delta, eta = get_random_directions(
            solver.model,
            norm=config['norm'],
            ignore=config['ignore']
        )
        print(f"      Generated δ and η")

        # Compute landscape
        print(f"Computing loss landscape ({config['n_points']}×{config['n_points']})...")
        Alpha, Beta, losses = compute_loss_on_grid(
            solver, base_weights, delta, eta,
            config['alpha_range'], config['beta_range'], config['n_points']
        )

        # Save to cache if needed
        if should_save_cache:
            save_landscape_to_h5(
                cache_path, Alpha, Beta, losses, delta, eta,
                model_type, K, config
            )
    else:
        # Skipped computation (loaded from cache)
        print(f"Skipped computation (loaded from cache)")

    # Statistics
    finite_mask = np.isfinite(losses)
    if finite_mask.any():
        fl = losses[finite_mask]
        print(f"      Min: {fl.min():.6e}, Max: {fl.max():.6e}")
        center_loss = losses[config['n_points']//2, config['n_points']//2]
        if np.isfinite(center_loss):
            print(f"      At θ*: {center_loss:.6e}")
            print(f"      Dynamic range: {fl.max() / fl.min():.2e}×")

    # Create plots
    print(f"Creating plots...")
    base_filename = f'{output_dir}/{model_type}_K{K}'

    # 2D Contour
    contour_path = f'{base_filename}_contour.png'
    plot_contour(Alpha, Beta, losses,
                f'{model_type} Loss Landscape (K={K})',
                contour_path,
                use_log=config['use_log_scale'],
                z_min=config['z_min'],
                z_max=config['z_max'])

    # 3D Surface
    surface_path = f'{base_filename}_surface.png'
    plot_surface_3d(Alpha, Beta, losses,
                   f'{model_type} Loss Landscape (K={K})',
                   surface_path,
                   use_log=config['use_log_scale'],
                   z_min=config['z_min'],
                   z_max=config['z_max'])

    return True


def main():
    """
    Generate all loss landscape plots with both contour and 3D views
    """
    
    # Extract configuration (merge both configs - TASK1_2 has training params, TASK1_3 has viz params)
    config = TASK1_2_CONFIG | TASK1_3_CONFIG
    N = config['N']
    K_values = config['K_values']
    model_types = config['model_types']
    output_dir = PATHS['plots_task1_3']
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Header
    print("TASK 1.3: LOSS LANDSCAPE VISUALIZATION")
    print(f"Models: {model_types}")
    print(f"K values: {K_values}")
    print(f"Grid: {config['n_points']}×{config['n_points']} ({config['n_points']**2} points)")
    print(f"Range: α ∈ {config['alpha_range']}, β ∈ {config['beta_range']}")
    print(f"Scale: {'Logarithmic' if config['use_log_scale'] else 'Linear'}")
    if config['z_min'] is not None or config['z_max'] is not None:
        z_min_str = f"{config['z_min']}" if config['z_min'] is not None else "auto"
        z_max_str = f"{config['z_max']}" if config['z_max'] is not None else "auto"
        print(f"Z-bounds: [{z_min_str}, {z_max_str}]")
    else:
        print(f"Z-bounds: Auto-scale")
    print(f"Cache mode: {config.get('cache_mode', 'use')}")
    
    # Process all models
    completed = []
    skipped = []
    
    for K in K_values:
        print(f"\nProcessing K = {K}")
        
        # Use same coefficients as training (seed=0)
        a_ij = generate_coefficients(K, seed=TASK1_2_CONFIG['seed'])
        
        for model_type in model_types:
            success = process_model(
                model_type=model_type,
                K=K,
                N=N,
                a_ij=a_ij,
                config=config,
                output_dir=output_dir
            )
            
            if success:
                completed.append(f"{model_type}_K{K}")
            else:
                skipped.append(f"{model_type}_K{K}")
    
    # Summary
    print(f"\nSuccessfully generated: {len(completed) * 2} plots ({len(completed)} models × 2 views)")
    for name in completed:
        print(f"   - {name}_contour.png")
        print(f"   - {name}_surface.png")
    
    if skipped:
        print(f"\nSkipped: {len(skipped)} models (not found)")
        for name in skipped:
            print(f"   - {name}")
        print(f"\nRun task1_2_train.py first to train the models")
    
    print(f"\nAll plots saved to: {output_dir}/")
    
    # Generate combined landscape plots
    if len(completed) >= 2:
        print("\nGENERATING COMBINED LANDSCAPE PLOTS")
        
        # Load all landscapes from cache
        print("\nLoading landscape data from cache...")
        landscapes_data = {}
        
        for model_type in model_types:
            landscapes_data[model_type] = {}
            
            for K in K_values:
                if f"{model_type}_K{K}" in completed:
                    cache_path = get_cache_path(model_type, K, config['cache_dir'])
                    
                    try:
                        # Create dummy model to load directions
                        a_ij = generate_coefficients(K, seed=TASK1_2_CONFIG['seed'])
                        
                        if model_type == 'PINN':
                            dummy_solver = PoissonPINN(
                                N=N,
                                K=K,
                                a_ij=a_ij,
                                n_collocation=TASK1_2_CONFIG['n_collocation'],
                                hidden_dim=TASK1_2_CONFIG['hidden_dim'],
                                n_layers=TASK1_2_CONFIG['n_layers']
                            )
                            dummy_model = dummy_solver.model
                        else:
                            dummy_solver = DataDrivenSolver(
                                N=N,
                                K=K,
                                a_ij=a_ij,
                                hidden_dim=TASK1_2_CONFIG['hidden_dim'],
                                n_layers=TASK1_2_CONFIG['n_layers']
                            )
                            dummy_model = dummy_solver.model
                        
                        # Load landscape data
                        Alpha, Beta, losses, delta, eta = load_landscape_from_h5(cache_path, dummy_model)
                        landscapes_data[model_type][K] = (Alpha, Beta, losses)
                        
                        print(f"Loaded landscape: {model_type} K={K}")
                        
                    except Exception as e:
                        print(f"Failed to load {model_type} K={K}: {e}")
        
        # Generate combined contour plot
        if landscapes_data:
            print("\nGenerating combined 2D contour plot...")
            contour_path = os.path.join(output_dir, 'all_landscapes_contour.png')
            plot_all_contours_combined(
                landscapes_data=landscapes_data,
                K_values=K_values,
                save_path=contour_path,
                use_log=config['use_log_scale'],
                z_min=config['z_min'],
                z_max=config['z_max']
            )
            
            # Generate combined 3D surface plot
            print("\nGenerating combined 3D surface plot...")
            surface_path = os.path.join(output_dir, 'all_landscapes_surface.png')
            plot_all_surfaces_combined(
                landscapes_data=landscapes_data,
                K_values=K_values,
                save_path=surface_path,
                use_log=config['use_log_scale'],
                z_min=config['z_min'],
                z_max=config['z_max']
            )
            
            print("\nGenerated files:")
            print(f"  - all_landscapes_contour.png")
            print(f"  - all_landscapes_surface.png")


if __name__ == '__main__':
    main()
