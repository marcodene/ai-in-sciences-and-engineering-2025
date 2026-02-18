import os

from config_task1 import USE_PRETRAINED, TASK1_2_CONFIG, PATHS
from physics import generate_coefficients
from models import PoissonPINN, DataDrivenSolver
from utils_task1 import (
    save_model, load_model, plot_solver_results, plot_loss_history,
    plot_pinn_vs_datadriven, plot_all_loss_histories
)


def train_or_load_model(model_type, K, N, a_ij, config):
    """
    Train a new model or load pretrained one based on USE_PRETRAINED flag
    
    Args:
        model_type: 'PINN' or 'DataDriven'
        K: Frequency parameter
        N: Grid resolution
        a_ij: Coefficient matrix
        config: Configuration dictionary
    
    Returns:
        solver: Trained or loaded solver
    """
    print(f"\n{model_type} Approach (K={K})")
    
    # Create solver
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
    
    # Check if we should load pretrained model
    model_path = f"{PATHS['checkpoints']}/{model_type}_K{K}.pt"
    
    if USE_PRETRAINED and os.path.exists(model_path):
        # Load pretrained model
        solver = load_model(solver, model_type, K, save_dir=PATHS['checkpoints'])
        
        # Verify loaded model
        print(f"Verifying loaded model...")
        l2_error_grid = solver.compute_l2_error()
        l2_error_test = solver.compute_test_error()
        print(f"L2 Error (grid): {l2_error_grid:.6f}")
        print(f"L2 Error (test set): {l2_error_test:.6f}")
        
    else:
        # Train new model
        if USE_PRETRAINED:
            print(f"WARNING: Pretrained model not found at {model_path}")
            print("Training new model instead...\n")
        
        # Print training info
        if model_type == 'PINN':
            print(f"Collocation points: {solver.xy_collocation.shape[0]}")
            print(f"Boundary points: {solver.xy_boundary.shape[0]}")
        else:
            print(f"Training points: {solver.xy_train.shape[0]}")
            if hasattr(solver, 'u_mean'):
                print(f"Data normalization: mean={solver.u_mean:.6f}, std={solver.u_std:.6f}")
        
        # Phase 1: Adam optimizer
        print("Training with Adam optimizer...")
        if model_type == 'PINN':
            solver.fit(
                epochs=config['epochs_adam'],
                lr=config['lr_adam'],
                print_every=config['print_every_adam']
            )
        else:
            solver.fit(
                epochs=config['epochs_adam_dd'],
                lr=config['lr_adam_dd'],
                print_every=config['print_every_adam_dd']
            )
        
        # Phase 2: L-BFGS fine-tuning
        print("Fine-tuning with L-BFGS optimizer...")
        solver.fit_lbfgs(max_iter=config['max_iter_lbfgs'])
        
        # Final evaluation
        l2_error_grid = solver.compute_l2_error()
        l2_error_test = solver.compute_test_error()
        print(f"\n{model_type} L2 Error (grid): {l2_error_grid:.6f}")
        print(f"{model_type} L2 Error (test set): {l2_error_test:.6f}")
        
        # Save model
        save_model(solver, model_type, K, save_dir=PATHS['checkpoints'])
    
    return solver


def main():
    """Execute Task 1.2: Train PINN and Data-Driven for all K values"""
    
    print("\nTASK 1.2: TRAINING PINN AND DATA-DRIVEN SOLVERS")
    
    # Extract configuration
    N = TASK1_2_CONFIG['N']
    K_values = TASK1_2_CONFIG['K_values']
    seed = TASK1_2_CONFIG['seed']
    
    print(f"\nConfiguration:")
    print(f"  Grid resolution: {N}×{N}")
    print(f"  K values: {K_values} (Low, Medium, High complexity)")
    print(f"  Network: {TASK1_2_CONFIG['n_layers']} layers × {TASK1_2_CONFIG['hidden_dim']} neurons")
    print(f"  USE_PRETRAINED: {USE_PRETRAINED}")
    print(f"  Checkpoints: {PATHS['checkpoints']}/")
    print(f"  Plots: {PATHS['plots_task1_2']}/")
    
    # Track results
    results = {}
    
    # Train for different complexity levels
    for K in K_values:
        complexity = {1: "Low", 4: "Medium", 16: "High"}[K]
        
        print(f"\nCOMPLEXITY LEVEL: {complexity} (K = {K})")
        
        # Generate one sample for this K (using fixed seed for reproducibility)
        a_ij = generate_coefficients(K, seed=seed)
        
        # Train/Load PINN
        pinn = train_or_load_model('PINN', K, N, a_ij, TASK1_2_CONFIG)
        
        # Train/Load Data-Driven
        dd = train_or_load_model('DataDriven', K, N, a_ij, TASK1_2_CONFIG)
        
        # Save results
        results[K] = {
            'PINN': {
                'l2_error_grid': pinn.compute_l2_error(),
                'l2_error_test': pinn.compute_test_error()
            },
            'DataDriven': {
                'l2_error_grid': dd.compute_l2_error(),
                'l2_error_test': dd.compute_test_error()
            }
        }
        
        # Generate plots
        print(f"\nGenerating plots...")
        plot_solver_results(pinn, f'PINN_K{K}', save_dir=PATHS['plots_task1_2'])
        plot_loss_history(pinn, f'PINN_K{K}', save_dir=PATHS['plots_task1_2'])
        
        plot_solver_results(dd, f'DataDriven_K{K}', save_dir=PATHS['plots_task1_2'])
        plot_loss_history(dd, f'DataDriven_K{K}', save_dir=PATHS['plots_task1_2'])
        
        # Comparison
        print(f"\nCOMPARISON (K={K})")
        print(f"PINN L2 Error:        {results[K]['PINN']['l2_error_grid']:.6f}")
        print(f"Data-Driven L2 Error: {results[K]['DataDriven']['l2_error_grid']:.6f}")
        
        if results[K]['PINN']['l2_error_grid'] < results[K]['DataDriven']['l2_error_grid']:
            print("PINN: lower error")
        else:
            print("Data-Driven: lower error")
    
    # Print summary table
    print("\nSUMMARY TABLE:")
    print(f"{'Complexity':<12} {'K':<5} {'Method':<15} {'L2 Error (grid)':<18}")
    print("-"*70)
    
    for K in K_values:
        complexity = {1: "Low", 4: "Medium", 16: "High"}[K]
        print(f"{complexity:<12} {K:<5} {'PINN':<15} {results[K]['PINN']['l2_error_grid']:<18.6f}")
        print(f"{'':12} {'':5} {'Data-Driven':<15} {results[K]['DataDriven']['l2_error_grid']:<18.6f}")
        print()
    
    # Generate comparison plots
    print("\nGENERATING MODEL COMPARISON PLOTS")
    
    for K in K_values:
        print(f"\nLoading models for K={K} comparison...")
        
        # Generate coefficients (same as used in training)
        a_ij = generate_coefficients(K, seed=seed)
        
        # Create and load PINN solver
        pinn_solver = PoissonPINN(
            N=N, K=K, a_ij=a_ij,
            n_collocation=TASK1_2_CONFIG['n_collocation'],
            hidden_dim=TASK1_2_CONFIG['hidden_dim'],
            n_layers=TASK1_2_CONFIG['n_layers'],
            lambda_bc=TASK1_2_CONFIG['lambda_bc']
        )
        pinn_solver = load_model(pinn_solver, 'PINN', K, save_dir=PATHS['checkpoints'])
        
        # Create and load DataDriven solver
        dd_solver = DataDrivenSolver(
            N=N, K=K, a_ij=a_ij,
            hidden_dim=TASK1_2_CONFIG['hidden_dim'],
            n_layers=TASK1_2_CONFIG['n_layers']
        )
        dd_solver = load_model(dd_solver, 'DataDriven', K, save_dir=PATHS['checkpoints'])
        
        # Generate comparison plot
        print(f"Generating comparison plot for K={K}...")
        plot_pinn_vs_datadriven(pinn_solver, dd_solver, K, save_dir=PATHS['plots_task1_2'])
    
    # Generate loss history comparison plots
    print("\nGENERATING LOSS HISTORY COMPARISON PLOTS")
    
    print("\nPlotting PINN loss histories...")
    plot_all_loss_histories(K_values, 'PINN', save_dir=PATHS['plots_task1_2'], 
                            checkpoints_dir=PATHS['checkpoints'])
    
    print("\nPlotting DataDriven loss histories...")
    plot_all_loss_histories(K_values, 'DataDriven', save_dir=PATHS['plots_task1_2'],
                            checkpoints_dir=PATHS['checkpoints'])
    
    print("\nResults saved in:")
    print(f"  - {PATHS['checkpoints']}/  (trained models)")
    print(f"  - {PATHS['plots_task1_2']}/  (visualizations)")
    print("\nGenerated comparison files:")
    for K in K_values:
        print(f"  - pinn_vs_datadriven_K{K}.png")
    print(f"  - PINN_all_loss_histories.png")
    print(f"  - DataDriven_all_loss_histories.png")


if __name__ == '__main__':
    main()
