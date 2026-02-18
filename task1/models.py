import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod

from physics import (
    create_grid, 
    compute_source_and_solution,
    compute_solution_at_points,
    compute_source_torch
)


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
        
        # Boundary points
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
