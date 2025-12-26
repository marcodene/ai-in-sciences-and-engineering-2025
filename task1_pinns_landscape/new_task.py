import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from abc import ABC, abstractmethod


# ============================================================================
# PHYSICS MODULE - Single Source of Truth
# ============================================================================

def create_grid(N):
    """Create spatial grid [0,1] x [0,1]"""
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    return X, Y


def generate_coefficients(K, seed=None):
    """Generate random coefficients from N(0,1)"""
    if seed is not None:
        np.random.seed(seed)
    return np.random.randn(K, K)


def compute_source_and_solution(X, Y, a_ij, K):
    """
    Compute source term f and solution u for the Poisson equation.
    
    f(x,y) = (π/K²) Σ a_ij · (i²+j²)^0.5 · sin(πix)sin(πjy)
    u(x,y) = (1/πK²) Σ a_ij · (i²+j²)^(-0.5) · sin(πix)sin(πjy)
    """
    f = np.zeros_like(X)
    u = np.zeros_like(X)
    
    for i in range(1, K + 1):
        for j in range(1, K + 1):
            coeff = a_ij[i-1, j-1]
            freq_weight_f = (i**2 + j**2)**0.5
            freq_weight_u = (i**2 + j**2)**(-0.5)
            spatial = np.sin(np.pi * i * X) * np.sin(np.pi * j * Y)
            
            f += coeff * freq_weight_f * spatial
            u += coeff * freq_weight_u * spatial
    
    f *= (np.pi / K**2)
    u *= (1.0 / (np.pi * K**2))
    
    return f, u


def compute_solution_at_points(xy_points, a_ij, K):
    """
    Compute exact solution u at arbitrary points (for test set evaluation).
    
    Args:
        xy_points: numpy array of shape (N, 2) with (x, y) coordinates
        a_ij: coefficient matrix (K x K)
        K: frequency parameter
    
    Returns:
        u: numpy array of shape (N,) with solution values
    """
    x = xy_points[:, 0:1]
    y = xy_points[:, 1:2]
    u = np.zeros((len(xy_points), 1))
    
    for i in range(1, K + 1):
        for j in range(1, K + 1):
            coeff = a_ij[i-1, j-1]
            freq_weight = (i**2 + j**2)**(-0.5)
            spatial = np.sin(np.pi * i * x) * np.sin(np.pi * j * y)
            u += coeff * freq_weight * spatial
    
    u *= (1.0 / (np.pi * K**2))
    return u.flatten()


def compute_source_torch(xy, a_ij_tensor, K):
    """
    Compute source term f at arbitrary points using PyTorch.
    
    Args:
        xy: torch.Tensor of shape (N, 2) with coordinates
        a_ij_tensor: torch.Tensor of shape (K, K) with coefficients
        K: frequency parameter
    """
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    f = torch.zeros(len(x), 1)
    
    for i in range(1, K + 1):
        for j in range(1, K + 1):
            coeff = a_ij_tensor[i-1, j-1]
            freq_weight = (i**2 + j**2)**0.5
            spatial = torch.sin(np.pi * i * x) * torch.sin(np.pi * j * y)
            f += coeff * freq_weight * spatial
    
    f *= (np.pi / K**2)
    return f


# ============================================================================
# NEURAL NETWORK
# ============================================================================

class MLP(nn.Module):
    """Multi-Layer Perceptron for function approximation"""
    
    def __init__(self, hidden_dim=128, num_hidden_layers=4):
        super(MLP, self).__init__()
        
        layers = []
        layers.append(nn.Linear(2, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, xy):
        return self.network(xy)


# ============================================================================
# BASE SOLVER
# ============================================================================

class BaseSolver(ABC):
    """Base class for Poisson equation solvers"""
    
    def __init__(self, N, K, a_ij, hidden_dim=128, n_layers=4):
        """
        Args:
            N: Grid resolution
            K: Frequency parameter
            a_ij: Coefficient matrix (K x K)
            hidden_dim: Hidden dimension of MLP
            n_layers: Number of hidden layers
        """
        self.N = N
        self.K = K
        self.a_ij = a_ij
        
        # Generate grid and exact solution using physics module
        self.X_grid, self.Y_grid = create_grid(N)
        self.f_grid, self.u_grid = compute_source_and_solution(
            self.X_grid, self.Y_grid, a_ij, K
        )
        
        # Initialize model
        self.model = MLP(hidden_dim=hidden_dim, num_hidden_layers=n_layers)
        
        # Training history (simple list of total losses)
        self.history = []
        
        # Pre-compute test set for online error monitoring (using Sobol sequences)
        sobol_engine_test = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=123)
        xy_test_np = sobol_engine_test.draw(8000).numpy()
        
        # Compute exact solution at test points
        u_test_exact_np = compute_solution_at_points(xy_test_np, a_ij, K)
        
        self.xy_test = torch.tensor(xy_test_np, dtype=torch.float32)
        self.u_test_exact = torch.tensor(u_test_exact_np, dtype=torch.float32)
        
        # Prepare training data (implemented by subclasses)
        self.prepare_training_data()
    
    @abstractmethod
    def prepare_training_data(self):
        """Prepare training data (implemented by subclasses)"""
        pass
    
    @abstractmethod
    def compute_loss(self):
        """Compute loss (implemented by subclasses)"""
        pass
    
    def compute_test_error(self):
        """Compute L2 relative error on test set"""
        with torch.no_grad():
            u_pred_test = self.model(self.xy_test).squeeze()
            
            # Denormalize if needed (for DataDrivenSolver)
            if hasattr(self, 'u_mean') and hasattr(self, 'u_std'):
                u_pred_test = u_pred_test * self.u_std + self.u_mean
            
            error = torch.sqrt(torch.mean((u_pred_test - self.u_test_exact)**2))
            norm = torch.sqrt(torch.mean(self.u_test_exact**2))
            l2_error = error / (norm + 1e-10)
        
        return l2_error.item()
    
    def fit(self, epochs=1000, lr=1e-3, print_every=100):
        """Train with Adam optimizer"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss_dict = self.compute_loss()
            loss = loss_dict['total']
            loss.backward()
            optimizer.step()
            
            # Record total loss only
            self.history.append(loss.item())
            
            # Print progress (including components and L2 error for monitoring)
            if (epoch + 1) % print_every == 0:
                l2_error_test = self.compute_test_error()
                
                loss_str = f"Epoch {epoch+1:5d} | Loss: {loss.item():.6f}"
                for key, value in loss_dict.items():
                    if key != 'total':
                        loss_str += f" | {key}: {value.item():.6f}"
                loss_str += f" | L2 Error: {l2_error_test:.6f}"
                print(loss_str)
        
        return self.history
    
    def fit_lbfgs(self, max_iter=500):
        """Fine-tune with L-BFGS optimizer"""
        optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            max_iter=max_iter,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            history_size=100,
            line_search_fn='strong_wolfe'
        )
        
        def closure():
            optimizer.zero_grad()
            loss_dict = self.compute_loss()
            loss = loss_dict['total']
            loss.backward()
            
            # Record total loss only
            self.history.append(loss.item())
            
            return loss
        
        optimizer.step(closure)
        
        # Compute final L2 error on test set
        l2_error_test = self.compute_test_error()
        
        print(f"L-BFGS | Final Loss: {self.history[-1]:.6f} | L2 Error: {l2_error_test:.6f}")
        
        return self.history
    
    def predict(self, X_test=None, Y_test=None):
        """Predict solution on grid"""
        if X_test is None or Y_test is None:
            X_test = self.X_grid
            Y_test = self.Y_grid
        
        xy_test = torch.tensor(
            np.stack([X_test.flatten(), Y_test.flatten()], axis=1),
            dtype=torch.float32
        )
        
        with torch.no_grad():
            u_pred = self.model(xy_test).numpy()
        
        # Denormalize if needed (only for DataDrivenSolver)
        if hasattr(self, 'u_mean') and hasattr(self, 'u_std'):
            u_pred = u_pred * self.u_std + self.u_mean
        
        u_pred = u_pred.reshape(X_test.shape)
        return u_pred
    
    def compute_l2_error(self):
        """Compute relative L2 error on grid (for final evaluation)"""
        u_pred = self.predict()
        error = np.sqrt(np.mean((u_pred - self.u_grid)**2))
        norm = np.sqrt(np.mean(self.u_grid**2))
        return error / (norm + 1e-10)


# ============================================================================
# PINN SOLVER
# ============================================================================

class PoissonPINN(BaseSolver):
    """
    Physics-Informed Neural Network solver
    
    Note on normalization: We don't normalize u in PINNs because:
    1. The PDE physics naturally constrains the solution scale
    2. Normalization would require scaling derivatives: ∂u_norm/∂x = (1/std)·∂u/∂x
    3. This would require careful tracking through all derivative computations
    4. The loss (PDE residual + BC) is already well-scaled by physics
    """
    
    def __init__(self, N, K, a_ij, n_collocation=10000, 
                 hidden_dim=128, n_layers=4, lambda_bc=10.0):
        self.n_collocation = n_collocation
        self.lambda_bc = lambda_bc
        super().__init__(N, K, a_ij, hidden_dim, n_layers)
    
    def prepare_training_data(self):
        """Prepare collocation and boundary points"""
        # Use Sobol sequence for collocation points (better space coverage than random)
        sobol_engine = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=42)
        self.xy_collocation = sobol_engine.draw(self.n_collocation).float()
        
        # Compute source term at collocation points using physics module
        a_ij_tensor = torch.tensor(self.a_ij, dtype=torch.float32)
        self.f_collocation = compute_source_torch(self.xy_collocation, a_ij_tensor, self.K)
        
        # Boundary points (same as before)
        bottom = np.stack([self.X_grid[0, :], self.Y_grid[0, :]], axis=1)
        top = np.stack([self.X_grid[-1, :], self.Y_grid[-1, :]], axis=1)
        left = np.stack([self.X_grid[1:-1, 0], self.Y_grid[1:-1, 0]], axis=1)
        right = np.stack([self.X_grid[1:-1, -1], self.Y_grid[1:-1, -1]], axis=1)
        
        xy_boundary = np.vstack([bottom, top, left, right])
        self.xy_boundary = torch.tensor(xy_boundary, dtype=torch.float32)
    
    def compute_laplacian(self, xy):
        """Compute Laplacian using automatic differentiation"""
        xy = xy.clone().detach().requires_grad_(True)
        u = self.model(xy)
        
        # First derivatives
        u_grad = torch.autograd.grad(
            u, xy, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        
        u_x = u_grad[:, 0:1]
        u_y = u_grad[:, 1:2]
        
        # Second derivatives
        u_xx = torch.autograd.grad(
            u_x, xy, grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True
        )[0][:, 0:1]
        
        u_yy = torch.autograd.grad(
            u_y, xy, grad_outputs=torch.ones_like(u_y),
            create_graph=True, retain_graph=True
        )[0][:, 1:2]
        
        laplacian = u_xx + u_yy
        return laplacian
    
    def compute_loss(self):
        """Compute PINN loss: PDE residual + boundary condition"""
        # PDE loss: -Δu = f
        laplacian = self.compute_laplacian(self.xy_collocation)
        residual = -laplacian - self.f_collocation
        loss_pde = torch.mean(residual**2)
        
        # Boundary loss: u = 0 on boundary
        u_boundary = self.model(self.xy_boundary)
        loss_bc = torch.mean(u_boundary**2)
        
        # Total loss
        loss_total = loss_pde + self.lambda_bc * loss_bc
        
        return {
            'total': loss_total,
            'PDE': loss_pde,
            'BC': loss_bc
        }


# ============================================================================
# DATA-DRIVEN SOLVER
# ============================================================================

class DataDrivenSolver(BaseSolver):
    """
    Supervised learning solver with normalization
    
    Note on normalization: Essential for DataDriven because:
    1. We're learning u directly via MSE loss
    2. Small u values (~0.01) lead to tiny gradients
    3. Normalization (mean=0, std=1) greatly improves training
    """
    
    def prepare_training_data(self):
        """Prepare supervised training data with normalization"""
        # Create input-output pairs from grid
        xy_train = np.stack([
            self.X_grid.flatten(),
            self.Y_grid.flatten()
        ], axis=1)
        
        u_train = self.u_grid.flatten().reshape(-1, 1)
        
        # Normalization
        self.u_mean = np.mean(u_train)
        self.u_std = np.std(u_train)
        u_train_scaled = (u_train - self.u_mean) / (self.u_std + 1e-8)
        
        # Convert to tensors
        self.xy_train = torch.tensor(xy_train, dtype=torch.float32)
        self.u_train = torch.tensor(u_train_scaled, dtype=torch.float32)
    
    def compute_loss(self):
        """Compute supervised MSE loss"""
        u_pred = self.model(self.xy_train)
        loss_mse = torch.mean((u_pred - self.u_train)**2)
        
        return {
            'total': loss_mse,
            'MSE': loss_mse
        }


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_samples(N, K_values, n_samples, save_dir):
    """Generate and save sample visualizations for Task 1.1"""
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
                              cmap='RdBu_r', aspect='equal')
            ax_f.set_xlabel('x')
            ax_f.set_ylabel('y')
            ax_f.set_title(f'Source f (sample {sample_idx+1})')
            plt.colorbar(im_f, ax=ax_f, fraction=0.046, pad=0.04)
            
            # Plot solution
            ax_u = axes[sample_idx, 1]
            im_u = ax_u.imshow(u, extent=[0, 1, 0, 1], origin='lower',
                              cmap='RdBu_r', aspect='equal')
            ax_u.set_xlabel('x')
            ax_u.set_ylabel('y')
            ax_u.set_title(f'Solution u (sample {sample_idx+1})')
            plt.colorbar(im_u, ax=ax_u, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # Save with high resolution
        save_path = os.path.join(save_dir, f'samples_K_{K}.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close(fig)


def plot_solver_results(solver, title, save_dir):
    """Plot and save solver prediction results"""
    os.makedirs(save_dir, exist_ok=True)
    
    u_pred = solver.predict()
    error = np.abs(u_pred - solver.u_grid)
    l2_error = solver.compute_l2_error()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Source term
    im0 = axes[0, 0].imshow(solver.f_grid, extent=[0, 1, 0, 1], 
                            origin='lower', cmap='RdBu_r')
    axes[0, 0].set_title('Source f', fontsize=12)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # True solution
    im1 = axes[0, 1].imshow(solver.u_grid, extent=[0, 1, 0, 1], 
                            origin='lower', cmap='RdBu_r')
    axes[0, 1].set_title('True u', fontsize=12)
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Predicted solution
    im2 = axes[1, 0].imshow(u_pred, extent=[0, 1, 0, 1], 
                            origin='lower', cmap='RdBu_r')
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
    print(f"✓ Saved: {save_path}")
    plt.close(fig)


def plot_loss_history(solver, title, save_dir):
    """Plot and save training loss history"""
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
    print(f"✓ Saved: {save_path}")
    plt.close(fig)


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def save_model(solver, method_name, K, save_dir='checkpoints'):
    """Save only the trained model weights"""
    os.makedirs(save_dir, exist_ok=True)
    
    filepath = f"{save_dir}/{method_name}_K{K}.pt"
    torch.save(solver.model.state_dict(), filepath)
    print(f"✓ Saved: {filepath}")


def load_model(solver, method_name, K, save_dir='checkpoints'):
    """Load trained weights into an existing solver"""
    filepath = f"{save_dir}/{method_name}_K{K}.pt"
    solver.model.load_state_dict(torch.load(filepath))
    print(f"✓ Loaded: {filepath}")
    return solver


def main():
    """Execute all tasks sequentially"""
    
    # Configuration
    N = 64  # Grid resolution
    K_values_task1 = [1, 4, 8, 16]  # For Task 1.1 visualization
    K_values_task2 = [1, 4, 16]  # For Task 1.2 training
    n_samples = 3  # Number of samples per K value
    
    print("="*70)
    print("TASK 1.1: DATA GENERATION AND VISUALIZATION")
    print("="*70)
    
    # Generate and save sample visualizations
    plot_samples(N, K_values_task1, n_samples, save_dir='plots/task1_1')
    
    print("\n" + "="*70)
    print("TASK 1.2: TRAINING PINN AND DATA-DRIVEN SOLVERS")
    print("="*70)
    
    # Train for different complexity levels
    for K in K_values_task2:
        print(f"\n{'='*70}")
        print(f"COMPLEXITY LEVEL: K = {K}")
        print(f"{'='*70}")
        
        # Generate one sample for this K (using seed for reproducibility)
        a_ij = generate_coefficients(K, seed=0)
        
        # ====================================================================
        # PINN APPROACH
        # ====================================================================
        print(f"\n{'-'*70}")
        print(f"PINN Approach (K={K})")
        print(f"{'-'*70}")
        
        pinn = PoissonPINN(
            N=N, K=K, a_ij=a_ij,
            n_collocation=10000,
            hidden_dim=256, n_layers=6,
            lambda_bc=10.0
        )
        
        print(f"Collocation points: {pinn.xy_collocation.shape[0]}")
        print(f"Boundary points: {pinn.xy_boundary.shape[0]}")
        print("Training with Adam optimizer...")
        
        pinn.fit(epochs=2000, lr=1e-3, print_every=200)
        
        print("Fine-tuning with L-BFGS optimizer...")
        pinn.fit_lbfgs(max_iter=2000)
        
        print(f"\n✓ PINN L2 Error (grid): {pinn.compute_l2_error():.6f}")
        print(f"✓ PINN L2 Error (test set): {pinn.compute_test_error():.6f}")
        
        # Save results
        plot_solver_results(pinn, f'PINN_K{K}', save_dir='plots/task1_2')
        plot_loss_history(pinn, f'PINN_K{K}', save_dir='plots/task1_2')
        save_model(pinn, 'PINN', K)

        # ====================================================================
        # DATA-DRIVEN APPROACH
        # ====================================================================
        print(f"\n{'-'*70}")
        print(f"Data-Driven Approach (K={K})")
        print(f"{'-'*70}")
        
        dd = DataDrivenSolver(
            N=N, K=K, a_ij=a_ij,
            hidden_dim=256, n_layers=6
        )
        
        print(f"Training points: {dd.xy_train.shape[0]}")
        print(f"Data normalization: mean={dd.u_mean:.6f}, std={dd.u_std:.6f}")
        print("Training with Adam optimizer...")
        
        dd.fit(epochs=4000, lr=1e-4, print_every=400)
        
        print("Fine-tuning with L-BFGS optimizer...")
        dd.fit_lbfgs(max_iter=2000)
        
        print(f"\n✓ Data-Driven L2 Error (grid): {dd.compute_l2_error():.6f}")
        print(f"✓ Data-Driven L2 Error (test set): {dd.compute_test_error():.6f}")
        
        # Save results
        plot_solver_results(dd, f'DataDriven_K{K}', save_dir='plots/task1_2')
        plot_loss_history(dd, f'DataDriven_K{K}', save_dir='plots/task1_2')
        save_model(dd, 'DataDriven', K) 

        # ====================================================================
        # COMPARISON
        # ====================================================================
        print(f"\n{'-'*70}")
        print(f"COMPARISON (K={K})")
        print(f"{'-'*70}")
        print(f"PINN L2 Error:        {pinn.compute_l2_error():.6f}")
        print(f"Data-Driven L2 Error: {dd.compute_l2_error():.6f}")
        
        if pinn.compute_l2_error() < dd.compute_l2_error():
            print("→ PINN performed better")
        else:
            print("→ Data-Driven performed better")
    
    print("\n" + "="*70)
    print("✓ ALL TASKS COMPLETED SUCCESSFULLY")
    print("="*70)
    print("Results saved in:")
    print("  - plots/task1_1/  (sample visualizations for K=1,4,8,16)")
    print("  - plots/task1_2/  (training results for K=1,4,16)")
    print("="*70)


if __name__ == '__main__':
    main()