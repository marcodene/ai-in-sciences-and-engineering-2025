import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from PoissonPINN import PoissonPINN


def compute_l2_relative_error(model, test_points, exact_solution_func):
    """
    Compute L2 relative error: ||u_pred - u_exact||_L2 / ||u_exact||_L2
    
    Args:
        model: Trained neural network
        test_points: Test points [N, 2]
        exact_solution_func: Function to compute exact solution
    
    Returns:
        L2 relative error as percentage
    """
    with torch.no_grad():
        u_pred = model(test_points).reshape(-1)
        u_exact = exact_solution_func(test_points).reshape(-1)
        
        numerator = torch.sqrt(torch.mean((u_pred - u_exact) ** 2))
        denominator = torch.sqrt(torch.mean(u_exact ** 2))
        
        relative_error = (numerator / denominator) * 100
    
    return relative_error.item()


def plot_training_history(history, mode, K, save_dir='plots/task1_2'):
    """
    Plot and save training loss curve
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    
    ax.plot(history, linewidth=1.5)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Training History - {mode.upper()} - K={K}', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'loss_curve_{mode}_K{K}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"  → Saved loss curve: {save_path}")
    plt.close(fig)


def plot_predictions(pinn_model, K, coefficients, mode, save_dir='plots/task1_2'):
    """
    Plot exact solution vs prediction vs error
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create dense grid for visualization
    N = 100
    x = torch.linspace(0, 1, N)
    y = torch.linspace(0, 1, N)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Flatten for network input
    xy_flat = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    
    # Get predictions
    with torch.no_grad():
        u_pred = pinn_model.approximate_solution(xy_flat).reshape(N, N)
        u_exact = pinn_model.exact_solution(xy_flat).reshape(N, N)
    
    error = torch.abs(u_pred - u_exact)
    
    # Create figure
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), dpi=100)
    
    # Exact solution
    im1 = axs[0].contourf(X.numpy(), Y.numpy(), u_exact.numpy(), 
                          levels=50, cmap='RdBu_r')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_title(f'Exact Solution - K={K}')
    axs[0].set_aspect('equal')
    plt.colorbar(im1, ax=axs[0])
    
    # Predicted solution
    im2 = axs[1].contourf(X.numpy(), Y.numpy(), u_pred.numpy(), 
                          levels=50, cmap='RdBu_r')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    axs[1].set_title(f'Predicted Solution ({mode.upper()}) - K={K}')
    axs[1].set_aspect('equal')
    plt.colorbar(im2, ax=axs[1])
    
    # Absolute error
    im3 = axs[2].contourf(X.numpy(), Y.numpy(), error.numpy(), 
                          levels=50, cmap='hot')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('y')
    axs[2].set_title(f'Absolute Error - K={K}')
    axs[2].set_aspect('equal')
    plt.colorbar(im3, ax=axs[2])
    
    plt.suptitle(f'Results: {mode.upper()} - K={K}', fontsize=16, y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'predictions_{mode}_K{K}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"  → Saved predictions: {save_path}")
    plt.close(fig)


def train_single_case(K, mode, n_int=5000, n_sb=500, 
                     hidden_dim=128, n_layers=4,
                     num_epochs_adam=1000, num_epochs_lbfgs=100,
                     lr_adam=0.001, lr_lbfgs=0.1,
                     seed=42, save_dir='plots/task1_2'):
    """
    Train a single PINN or Data-Driven model for given K
    
    Args:
        K: Frequency parameter (complexity)
        mode: 'pinn' or 'data'
        n_int: Number of interior points
        n_sb: Number of boundary points per edge
        hidden_dim: MLP hidden dimension
        n_layers: Number of hidden layers
        num_epochs_adam: Epochs for Adam phase
        num_epochs_lbfgs: Epochs for L-BFGS phase
        lr_adam: Adam learning rate
        lr_lbfgs: L-BFGS learning rate
        seed: Random seed
        save_dir: Directory to save results
    
    Returns:
        results: Dictionary with training history, final error, model
    """
    print(f"\n{'='*70}")
    print(f"Training {mode.upper()} - K={K}")
    print(f"{'='*70}")
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate coefficients (fixed for this case)
    coefficients = torch.randn(K, K)
    
    # Initialize model
    pinn = PoissonPINN(
        n_int_=n_int,
        n_sb_=n_sb,
        K=K,
        coefficients=coefficients,
        mode=mode,
        hidden_dim=hidden_dim,
        n_layers=n_layers
    )
    
    # Create optimizers
    optimizer_adam = torch.optim.Adam(
        pinn.approximate_solution.parameters(),
        lr=lr_adam
    )
    
    optimizer_lbfgs = torch.optim.LBFGS(
        pinn.approximate_solution.parameters(),
        lr=lr_lbfgs,
        max_iter=50,
        max_eval=50,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=100,
        line_search_fn="strong_wolfe"
    )
    
    # Train
    print(f"\nTraining configuration:")
    print(f"  - Interior points: {n_int}")
    print(f"  - Boundary points: {4 * n_sb}")
    print(f"  - Network: {n_layers} layers × {hidden_dim} neurons")
    print(f"  - Adam epochs: {num_epochs_adam}")
    print(f"  - L-BFGS epochs: {num_epochs_lbfgs}\n")
    
    history = pinn.fit(
        num_epochs_adam=num_epochs_adam,
        num_epochs_lbfgs=num_epochs_lbfgs,
        optimizer_adam=optimizer_adam,
        optimizer_lbfgs=optimizer_lbfgs,
        verbose=True  # Set to True for detailed output
    )
    
    # Compute final L2 error on test points
    test_sobol = torch.quasirandom.SobolEngine(dimension=2)
    test_points = test_sobol.draw(10000)  # 10k test points
    test_points = test_points * (pinn.domain_extrema[:, 1] - pinn.domain_extrema[:, 0]) + pinn.domain_extrema[:, 0]
    
    l2_error = compute_l2_relative_error(
        pinn.approximate_solution, 
        test_points, 
        pinn.exact_solution
    )
    
    print(f"\n{'='*70}")
    print(f"✓ Training complete for {mode.upper()} - K={K}")
    print(f"  Final Loss: {history[-1]:.6e}")
    print(f"  L2 Relative Error: {l2_error:.4f}%")
    print(f"{'='*70}\n")
    
    # Save results
    plot_training_history(history, mode, K, save_dir)
    plot_predictions(pinn, K, coefficients, mode, save_dir)
    
    # Return results
    results = {
        'mode': mode,
        'K': K,
        'history': history,
        'final_loss': history[-1],
        'l2_error': l2_error,
        'coefficients': coefficients,
        'model': pinn
    }
    
    return results


def task_1_2_full_training(K_values=[1, 4, 16], 
                           save_dir='plots/task1_2',
                           results_file='results/task1_2_results.txt'):
    """
    Complete Task 1.2: Train both PINN and Data-Driven for all K values
    
    Args:
        K_values: List of K values to train (Low, Medium, High complexity)
        save_dir: Directory to save plots
        results_file: File to save numerical results
    """
    print("\n" + "="*70)
    print("TASK 1.2: PINN vs Data-Driven Training")
    print("="*70)
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    all_results = {}
    
    for K in K_values:
        complexity = {1: "Low", 4: "Medium", 16: "High"}[K]
        print(f"\n{'#'*70}")
        print(f"# COMPLEXITY: {complexity} (K={K})")
        print(f"{'#'*70}")
        
        # Train PINN
        pinn_results = train_single_case(
            K=K, 
            mode='pinn',
            n_int=5000,
            n_sb=500,
            num_epochs_adam=400,
            num_epochs_lbfgs=100,
            seed=42,
            save_dir=save_dir
        )
        
        # Train Data-Driven
        data_results = train_single_case(
            K=K, 
            mode='data',
            n_int=5000,
            n_sb=500,
            num_epochs_adam=400,
            num_epochs_lbfgs=100,
            seed=42,
            save_dir=save_dir
        )
        
        all_results[K] = {
            'pinn': pinn_results,
            'data': data_results
        }
    
    # Save summary results to file
    with open(results_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("TASK 1.2 RESULTS SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for K in K_values:
            complexity = {1: "Low", 4: "Medium", 16: "High"}[K]
            f.write(f"\n{'='*70}\n")
            f.write(f"Complexity: {complexity} (K={K})\n")
            f.write(f"{'='*70}\n\n")
            
            pinn_res = all_results[K]['pinn']
            data_res = all_results[K]['data']
            
            f.write("PINN Results:\n")
            f.write(f"  - Final Loss: {pinn_res['final_loss']:.6e}\n")
            f.write(f"  - L2 Relative Error: {pinn_res['l2_error']:.4f}%\n\n")
            
            f.write("Data-Driven Results:\n")
            f.write(f"  - Final Loss: {data_res['final_loss']:.6e}\n")
            f.write(f"  - L2 Relative Error: {data_res['l2_error']:.4f}%\n\n")
    
    print(f"\n{'='*70}")
    print(f"✓ All training complete!")
    print(f"  - Plots saved to: {save_dir}/")
    print(f"  - Results saved to: {results_file}")
    print(f"{'='*70}\n")
    
    return all_results


def create_comparison_table(all_results, save_path='results/task1_2_table.txt'):
    """
    Create a formatted comparison table of all results
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("\n" + "="*100 + "\n")
        f.write("TASK 1.2: COMPARISON TABLE\n")
        f.write("="*100 + "\n\n")
        
        # Header
        f.write(f"{'Complexity':<15} {'K':<5} {'Mode':<12} {'Final Loss':<15} {'L2 Error (%)':<15}\n")
        f.write("-"*100 + "\n")
        
        # Data rows
        for K in sorted(all_results.keys()):
            complexity = {1: "Low", 4: "Medium", 16: "High"}[K]
            
            pinn_res = all_results[K]['pinn']
            data_res = all_results[K]['data']
            
            f.write(f"{complexity:<15} {K:<5} {'PINN':<12} {pinn_res['final_loss']:<15.6e} {pinn_res['l2_error']:<15.4f}\n")
            f.write(f"{'':<15} {'':<5} {'Data-Driven':<12} {data_res['final_loss']:<15.6e} {data_res['l2_error']:<15.4f}\n")
            f.write("\n")
        
        f.write("="*100 + "\n")
    
    print(f"✓ Comparison table saved to: {save_path}\n")


# Main execution
if __name__ == "__main__":
    # Run complete Task 1.2
    all_results = task_1_2_full_training(
        K_values=[1, 4, 16],
        save_dir='plots/task1_2',
        results_file='results/task1_2_results.txt'
    )
    
    # Create comparison table
    create_comparison_table(all_results, save_path='results/task1_2_table.txt')
    
    print("\n🎉 Task 1.2 Complete! 🎉\n")