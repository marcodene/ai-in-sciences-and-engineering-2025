"""
Loss Landscape Visualization - Complete Analysis with 3D Plots
All PINNs and Data-Driven models for K=1,4,16
Following Li et al. 2018 methodology
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from new_task import PoissonPINN, DataDrivenSolver, generate_coefficients
import os
from tqdm import tqdm


def get_random_directions(model, norm='filter', ignore='biasbn'):
    """Generate two random normalized directions (δ and η)"""
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
    """Set model weights: θ = θ* + α·δ + β·η"""
    if directions is None:
        for p, w in zip(model.parameters(), base_weights):
            p.data.copy_(w)
    else:
        delta, eta = directions
        for p, w, d, e in zip(model.parameters(), base_weights, delta, eta):
            p.data = w + alpha * d + beta * e


def compute_loss_on_grid(solver, base_weights, delta, eta, alpha_range, beta_range, n_points=41):
    """Compute loss on 2D grid: L(θ* + α·δ + β·η)"""
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
    """Create 2D contour plot

    Args:
        z_min: Optional minimum z value for color scale
        z_max: Optional maximum z value for color scale
    """

    # Handle NaN/Inf values
    finite_mask = np.isfinite(losses)
    if not finite_mask.any():
        print(f"\n❌ ERROR: All losses are NaN/Inf!")
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
    contour = ax.contourf(Alpha, Beta, losses_plot, levels=levels, cmap='viridis')
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
    print(f"      ✓ Saved: {save_path}")
    plt.close()


def plot_surface_3d(Alpha, Beta, losses, title, save_path, use_log=False, z_min=None, z_max=None):
    """Create 3D surface plot

    Args:
        z_min: Optional minimum z value for color scale
        z_max: Optional maximum z value for color scale
    """

    # Handle NaN/Inf values
    finite_mask = np.isfinite(losses)
    if not finite_mask.any():
        print(f"\n❌ ERROR: All losses are NaN/Inf!")
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
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Surface plot
    surf = ax.plot_surface(Alpha, Beta, losses_plot, cmap='viridis',
                          linewidth=0, antialiased=True, alpha=0.9,
                          rstride=1, cstride=1)
    
    # Mark θ* at center
    n_beta, n_alpha = losses_plot.shape
    center_i, center_j = n_beta // 2, n_alpha // 2
    ax.scatter([0], [0], [losses_plot[center_i, center_j]], 
              color='red', s=200, marker='*',
              edgecolors='white', linewidths=2.5,
              label='θ* (trained)', zorder=1000)
    
    # Labels and styling
    ax.set_xlabel('α (direction δ)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('β (direction η)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_zlabel(zlabel, fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Better viewing angle
    ax.view_init(elev=30, azim=225)
    
    # Color bar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=8, pad=0.1)
    cbar.set_label(zlabel, fontsize=11, fontweight='bold')
    
    ax.legend(fontsize=10, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"      ✓ Saved: {save_path}")
    plt.close()


def process_model(model_type, K, N, a_ij, alpha_range, beta_range, n_points,
                 output_dir, use_log_scale=False, z_min=None, z_max=None):
    """
    Process a single model and generate both contour and 3D plots

    Args:
        model_type: 'PINN' or 'DataDriven'
        K: coefficient complexity
        N: grid size
        a_ij: coefficients
        alpha_range, beta_range: perturbation ranges
        n_points: grid resolution
        output_dir: where to save plots
        use_log_scale: whether to use logarithmic scale
        z_min: optional minimum z value for plots (in log10 if use_log_scale=True)
        z_max: optional maximum z value for plots (in log10 if use_log_scale=True)
    """
    print(f"\n{'='*70}")
    print(f"{model_type} Model (K={K})")
    print(f"{'='*70}")
    
    # Check if model exists
    model_path = f"checkpoints/{model_type}_K{K}.pt"
    if not os.path.exists(model_path):
        print(f"⚠️  Model not found: {model_path}")
        print(f"Skipping {model_type} K={K}...")
        return False
    
    # Create solver
    print(f"[1/5] Loading {model_type} model...")
    if model_type == 'PINN':
        solver = PoissonPINN(
            N=N, K=K, a_ij=a_ij,
            n_collocation=10000,
            hidden_dim=256, n_layers=6,
            lambda_bc=10.0
        )
    else:  # DataDriven
        solver = DataDrivenSolver(
            N=N, K=K, a_ij=a_ij,
            hidden_dim=256, n_layers=6
        )
    
    solver.model.load_state_dict(torch.load(model_path))
    base_weights = [p.data.clone() for p in solver.model.parameters()]
    print(f"      ✓ Loaded from {model_path}")
    
    # Verify
    print(f"[2/5] Verifying model...")
    loss = solver.compute_loss()
    print(f"      Loss at θ*: {loss['total'].item():.6e}")
    
    # Generate directions
    print(f"[3/5] Generating random directions...")
    delta, eta = get_random_directions(solver.model, norm='filter', ignore='biasbn')
    print(f"      ✓ Generated δ and η")
    
    # Compute landscape
    print(f"[4/5] Computing loss landscape ({n_points}×{n_points})...")
    Alpha, Beta, losses = compute_loss_on_grid(
        solver, base_weights, delta, eta,
        alpha_range, beta_range, n_points
    )
    
    # Statistics
    finite_mask = np.isfinite(losses)
    if finite_mask.any():
        fl = losses[finite_mask]
        print(f"      Min: {fl.min():.6e}, Max: {fl.max():.6e}")
        center_loss = losses[n_points//2, n_points//2]
        if np.isfinite(center_loss):
            print(f"      At θ*: {center_loss:.6e}")
            print(f"      Dynamic range: {fl.max() / fl.min():.2e}×")
    
    # Create plots
    print(f"[5/5] Creating plots...")
    base_filename = f'{output_dir}/{model_type}_K{K}'

    # 2D Contour
    contour_path = f'{base_filename}_contour.png'
    plot_contour(Alpha, Beta, losses,
                f'{model_type} Loss Landscape (K={K})',
                contour_path, use_log=use_log_scale, z_min=z_min, z_max=z_max)

    # 3D Surface
    surface_path = f'{base_filename}_surface.png'
    plot_surface_3d(Alpha, Beta, losses,
                   f'{model_type} Loss Landscape (K={K})',
                   surface_path, use_log=use_log_scale, z_min=z_min, z_max=z_max)
    
    return True


def main():
    """
    Generate all loss landscape plots with both contour and 3D views
    """
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    N = 64
    K_values = [1, 4, 16]
    model_types = ['PINN', 'DataDriven']
    
    # Li et al. 2018 standard range
    alpha_range = (-0.5, 0.5)
    beta_range = (-0.5, 0.5)
    
    # Grid resolution options:
    # - 21×21 (441 points): Fast, ~2 min per model
    # - 41×41 (1681 points): Standard, ~4 min per model ← DEFAULT
    # - 51×51 (2601 points): High res, ~6 min per model
    # - 101×101 (10201 points): Very high res, ~25 min per model
    n_points = 41
    
    # Logarithmic scale options:
    # - False: Linear scale (good when losses don't vary too much)
    # - True: Log scale (good when losses span many orders of magnitude)
    #         Use this if loss at θ* is ~1e-6 but boundaries are ~100
    use_log_scale = False  # Change to True for better visualization of wide ranges

    # Z-axis bounds (optional):
    # - None, None: Auto-scale based on data
    # - Set specific values to control the vertical scale
    # - If use_log_scale=True, specify values in log10 scale (e.g., -6 for 10^-6)
    # - If use_log_scale=False, specify actual loss values
    # Examples:
    #   z_min, z_max = None, None  # Auto-scale (default)
    #   z_min, z_max = -6, 2       # For log scale: 10^-6 to 10^2
    #   z_min, z_max = 0, 100      # For linear scale: 0 to 100
    z_min, z_max = 0, 30

    output_dir = 'plots/task_loss_landscape'
    os.makedirs(output_dir, exist_ok=True)
    
    # ========================================================================
    # HEADER
    # ========================================================================
    print("="*70)
    print("COMPLETE LOSS LANDSCAPE ANALYSIS")
    print("Following Li et al. 2018 Methodology")
    print("="*70)
    print(f"Models: {model_types}")
    print(f"K values: {K_values}")
    print(f"Grid: {n_points}×{n_points} ({n_points**2} points)")
    print(f"Range: α ∈ [{alpha_range[0]}, {alpha_range[1]}], β ∈ [{beta_range[0]}, {beta_range[1]}]")
    print(f"Scale: {'Logarithmic' if use_log_scale else 'Linear'}")
    if z_min is not None or z_max is not None:
        z_min_str = f"{z_min}" if z_min is not None else "auto"
        z_max_str = f"{z_max}" if z_max is not None else "auto"
        print(f"Z-bounds: [{z_min_str}, {z_max_str}]")
    else:
        print(f"Z-bounds: Auto-scale")
    print(f"Total plots to generate: {len(model_types) * len(K_values) * 2} (contour + 3D)")
    print("="*70)
    
    # ========================================================================
    # PROCESS ALL MODELS
    # ========================================================================
    completed = []
    skipped = []
    
    for K in K_values:
        print(f"\n{'#'*70}")
        print(f"# Processing K = {K}")
        print(f"{'#'*70}")
        
        a_ij = generate_coefficients(K, seed=0)
        
        for model_type in model_types:
            success = process_model(
                model_type=model_type,
                K=K,
                N=N,
                a_ij=a_ij,
                alpha_range=alpha_range,
                beta_range=beta_range,
                n_points=n_points,
                output_dir=output_dir,
                use_log_scale=use_log_scale,
                z_min=z_min,
                z_max=z_max
            )
            
            if success:
                completed.append(f"{model_type}_K{K}")
            else:
                skipped.append(f"{model_type}_K{K}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\n✓ Successfully generated: {len(completed) * 2} plots ({len(completed)} models × 2 views)")
    for name in completed:
        print(f"   - {name}_contour.png")
        print(f"   - {name}_surface.png")
    
    if skipped:
        print(f"\n⚠️  Skipped: {len(skipped)} models (not found)")
        for name in skipped:
            print(f"   - {name}")
    
    print(f"\nAll plots saved to: {output_dir}/")
    print("="*70)
    
    # Analysis tips
    if len(completed) >= 2:
        print("\n" + "="*70)
        print("VISUALIZATION GUIDE")
        print("="*70)
        print("\n📊 What to Look For:")
        print("  • Bowl shape: Convex, well-conditioned landscape")
        print("  • Valley shape: Flat manifold, ill-conditioned")
        print("  • Chaotic: Multiple minima, very hard to optimize")
        print("\n🔍 Comparisons to Make:")
        print("  1. PINN vs DataDriven (same K):")
        print("     → Physics regularization effect")
        print("     → PINN should be smoother but more anisotropic")
        print("  2. K=1 vs K=4 vs K=16 (same type):")
        print("     → Complexity effect on landscape")
        print("     → Higher K might be harder to optimize")
        print("  3. Contour vs 3D:")
        print("     → Contours show curvature clearly")
        print("     → 3D shows overall shape intuitively")
        print("\n📐 Quantitative Metrics:")
        print("  • Dynamic range: Max loss / Min loss")
        print("  • Anisotropy: Compare different directions")
        print("  • Sharpness: How fast does loss increase?")
        print("="*70)


if __name__ == '__main__':
    main()