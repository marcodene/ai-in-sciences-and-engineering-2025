"""
Loss Landscape Visualization - Following Li et al. 2018 Methodology
Uses in-place parameter modification (not load_state_dict)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
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
    """
    Set model weights following Li et al. 2018:
    θ = θ* + α·δ + β·η
    
    Modifies parameters IN-PLACE (like Goldstein's code)
    """
    if directions is None:
        # Just restore base weights
        for p, w in zip(model.parameters(), base_weights):
            p.data.copy_(w)
    else:
        # Perturb: θ = θ* + α·δ + β·η
        delta, eta = directions
        for p, w, d, e in zip(model.parameters(), base_weights, delta, eta):
            p.data = w + alpha * d + beta * e


def compute_loss_on_grid(solver, base_weights, delta, eta, alpha_range, beta_range, n_points=41):
    """
    Compute loss on 2D grid with proper BN handling
    Following Li et al. 2018 Section 2.3
    """
    alphas = np.linspace(alpha_range[0], alpha_range[1], n_points)
    betas = np.linspace(beta_range[0], beta_range[1], n_points)
    losses = np.zeros((len(betas), len(alphas)))
    
    # CRITICAL: Freeze BatchNorm statistics (if any BN layers exist)
    # This ensures running_mean and running_var stay constant
    solver.model.eval()  # Put in eval mode (freezes BN stats)
    
    # But we still need gradients for PINN's PDE residual computation
    # Solution: Use eval mode + explicit gradient enabling
    
    total_points = len(alphas) * len(betas)
    pbar = tqdm(total=total_points, desc="Computing loss landscape")
    
    for i, beta in enumerate(betas):
        for j, alpha in enumerate(alphas):
            # Set perturbed weights: θ = θ* + α·δ + β·η
            set_weights(solver.model, base_weights, (delta, eta), alpha, beta)
            
            # Compute loss WITH gradients but WITHOUT updating BN stats
            with torch.set_grad_enabled(True):  # Enable gradients for autograd
                try:
                    loss_dict = solver.compute_loss()
                    loss_value = loss_dict['total'].item()
                    
                    if not np.isfinite(loss_value):
                        loss_value = np.nan
                    
                    losses[i, j] = loss_value
                    
                except Exception as e:
                    print(f"\nError at α={alpha:.3f}, β={beta:.3f}: {str(e)}")
                    losses[i, j] = np.nan
            
            # Clear gradients
            solver.model.zero_grad()
            pbar.update(1)
    
    pbar.close()
    
    # Restore original weights
    set_weights(solver.model, base_weights)
    
    Alpha, Beta = np.meshgrid(alphas, betas)
    return Alpha, Beta, losses


def plot_contour(Alpha, Beta, losses, title, save_path):
    """Create contour plot with NaN handling"""
    
    # Handle NaN/Inf values
    finite_mask = np.isfinite(losses)
    if not finite_mask.any():
        print(f"\n❌ ERROR: All losses are NaN/Inf!")
        return
    
    finite_losses = losses[finite_mask]
    print(f"\n      Loss range: [{finite_losses.min():.2e}, {finite_losses.max():.2e}]")
    print(f"      Finite points: {finite_mask.sum()} / {losses.size}")
    
    # Replace NaN with high value
    losses_plot = losses.copy()
    max_finite = finite_losses.max()
    losses_plot[~finite_mask] = max_finite * 1.5
    
    # Clip outliers
    vmin = np.percentile(finite_losses, 1)
    vmax = np.percentile(finite_losses, 99)
    losses_plot = np.clip(losses_plot, vmin, vmax)
    
    # Plot
    fig, ax = plt.subplots(figsize=(11, 9))
    
    levels = 30
    contour = ax.contourf(Alpha, Beta, losses_plot, levels=levels, cmap='viridis')
    ax.contour(Alpha, Beta, losses_plot, levels=levels, colors='black', 
               alpha=0.15, linewidths=0.5)
    
    ax.plot(0, 0, 'r*', markersize=25, markeredgecolor='white', 
            markeredgewidth=2.5, label='θ* (trained)', zorder=10)
    
    ax.set_xlabel('α (direction δ)', fontsize=14, fontweight='bold')
    ax.set_ylabel('β (direction η)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_aspect('equal')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.2)
    
    cbar = plt.colorbar(contour, ax=ax, pad=0.02)
    cbar.set_label('Loss', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {save_path}")
    plt.close()


def process_model(model_type, K, N, a_ij, alpha_range, beta_range, n_points, output_dir):
    """
    Process a single model (PINN or DataDriven)
    
    Args:
        model_type: 'PINN' or 'DataDriven'
        K: coefficient complexity
        N: grid size
        a_ij: coefficients
        alpha_range, beta_range: perturbation ranges
        n_points: grid resolution
        output_dir: where to save plots
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
    print(f"[1/4] Loading {model_type} model...")
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
    print(f"[2/4] Verifying model...")
    loss = solver.compute_loss()
    print(f"      Loss at θ*: {loss['total'].item():.6e}")
    
    # Generate directions
    print(f"[3/4] Generating random directions...")
    delta, eta = get_random_directions(solver.model, norm='filter', ignore='biasbn')
    print(f"      ✓ Generated δ and η")
    
    # Compute landscape
    print(f"[4/4] Computing loss landscape ({n_points}×{n_points})...")
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
    
    # Plot
    save_path = f'{output_dir}/{model_type}_K{K}_contour.png'
    plot_contour(Alpha, Beta, losses, f'{model_type} Loss Landscape (K={K})', save_path)
    
    return True


def main():
    """
    Generate all loss landscape plots:
    - PINN: K=1, 4, 16
    - DataDriven: K=1, 4, 16
    Total: 6 plots
    """
    
    # Configuration
    N = 64
    K_values = [1, 4, 16]
    model_types = ['PINN', 'DataDriven']
    
    # Li et al. 2018 standard range
    alpha_range = (-1.0, 1.0)
    beta_range = (-1.0, 1.0)
    n_points = 41  # 41×41 grid
    
    output_dir = 'plots/task_loss_landscape'
    os.makedirs(output_dir, exist_ok=True)
    
    # Print header
    print("="*70)
    print("COMPLETE LOSS LANDSCAPE ANALYSIS")
    print("Following Li et al. 2018 Methodology")
    print("="*70)
    print(f"Models: {model_types}")
    print(f"K values: {K_values}")
    print(f"Grid: {n_points}×{n_points}, Range: [{alpha_range[0]}, {alpha_range[1]}]")
    print(f"Total plots to generate: {len(model_types) * len(K_values)}")
    print("="*70)
    
    # Track results
    completed = []
    skipped = []
    
    # Loop over all combinations
    for K in K_values:
        print(f"\n{'#'*70}")
        print(f"# Processing K = {K}")
        print(f"{'#'*70}")
        
        # Generate coefficients (same for both models)
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
                output_dir=output_dir
            )
            
            if success:
                completed.append(f"{model_type}_K{K}")
            else:
                skipped.append(f"{model_type}_K{K}")
    
    # Final summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\n✓ Successfully generated: {len(completed)} plots")
    for name in completed:
        print(f"   - {name}_contour.png")
    
    if skipped:
        print(f"\n⚠️  Skipped: {len(skipped)} models (not found)")
        for name in skipped:
            print(f"   - {name}")
    
    print(f"\nAll plots saved to: {output_dir}/")
    print("="*70)
    
    # Create comparison summary
    if len(completed) >= 2:
        print("\n" + "="*70)
        print("ANALYSIS TIPS")
        print("="*70)
        print("Compare the plots to analyze:")
        print("  1. PINN vs DataDriven (same K): Physics regularization effect")
        print("  2. K=1 vs K=4 vs K=16 (same type): Complexity effect")
        print("  3. Landscape shape: Bowl vs valley vs chaotic")
        print("  4. Anisotropy: Circular vs elliptical contours")
        print("  5. Sharpness: Loss increase rate from center")
        print("="*70)


if __name__ == '__main__':
    main()