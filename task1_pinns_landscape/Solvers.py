import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from generate_data import generate_sample, create_grid

class MLP(nn.Module):
    """Multi-Layer Perceptron for function approximation"""
    def __init__(self, hidden_dim=128, num_hidden_layers=4):
        super(MLP, self).__init__()
        
        self.input_layer = nn.Linear(2, hidden_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) 
            for _ in range(num_hidden_layers - 1)
        ])
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.activation = nn.Tanh()
    
    def forward(self, xy):
        x = self.activation(self.input_layer(xy))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        u = self.output_layer(x)
        return u


class BaseSolver(ABC):
    """
    Base class for Poisson equation solvers
    Contains all common functionality
    """

    def __init__(self, f_grid, u_grid, X_grid, Y_grid, hidden_dim=128, n_layers=4):
        self.f_grid = f_grid
        self.u_grid = u_grid
        self.X_grid = X_grid
        self.Y_grid = Y_grid

        self.model = MLP(hidden_dim=hidden_dim, num_hidden_layers=n_layers)

        self.history = {'loss':[], 'loss_components':{}}

        self.prepare_training_data()

    @abstractmethod
    def prepare_training_data(self):
        pass

    @abstractmethod
    def compute_loss(self):
        pass

    def fit(self, epochs=1000, lr=1e-3):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            loss_dict = self.compute_loss()
            loss = loss_dict['total']
            loss.backward()
            optimizer.step()
            
            self.history['loss'].append(loss.item())
            for key, value in loss_dict.items():
                if key != 'total':
                    if key not in self.history['loss_components']:
                        self.history['loss_components'][key] = []
                    self.history['loss_components'][key].append(value.item())

            if (epoch + 1)% 10 == 0:
                loss_str = f"Epoch {epoch:5d} | Loss: {loss.item():.6f}"
                for key, value in loss_dict.items():
                    if key != 'total':
                        loss_str += f" | {key}: {value.item():.6f}"
                print(loss_str)

        return self.history

    def fit_lbfgs(self, max_iter=500):
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
            
            self.history['loss'].append(loss.item())
            for key, value in loss_dict.items():
                if key != 'total':
                    if key not in self.history['loss_components']:
                        self.history['loss_components'][key] = []
                    self.history['loss_components'][key].append(value.item())
            
            return loss
            
        optimizer.step(closure)

        print(f"L-BFGS | Final Loss: {self.history['loss'][-1]:.6f}")

        return self.history

    def predict(self, X_test=None, Y_test=None):
        if X_test is None or Y_test is None:
            X_test = self.X_grid
            Y_test = self.Y_grid
        
        xy_test = torch.tensor(
            np.stack([X_test.flatten(), Y_test.flatten()], axis=1),
            dtype=torch.float32
        )

        with torch.no_grad():
            u_pred = self.model(xy_test)

        u_pred = u_pred.numpy().reshape(X_test.shape)

        return u_pred

    def compute_l2_error(self):
        u_pred = self.predict()
        error = np.sqrt(np.mean((u_pred - self.u_grid)**2)) / np.sqrt(np.mean(self.u_grid**2))
        return error

    def plot_results(self):
        """Plot f, true u, predicted u, and error"""
        u_pred = self.predict()
        error = np.abs(u_pred - self.u_grid)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Source term
        im0 = axes[0, 0].imshow(self.f_grid, extent=[0,1,0,1], origin='lower', cmap='RdBu_r')
        axes[0, 0].set_title('Source f')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y')
        plt.colorbar(im0, ax=axes[0, 0])
        
        # True solution
        im1 = axes[0, 1].imshow(self.u_grid, extent=[0,1,0,1], origin='lower', cmap='RdBu_r')
        axes[0, 1].set_title('True u')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Predicted solution
        im2 = axes[1, 0].imshow(u_pred, extent=[0,1,0,1], origin='lower', cmap='RdBu_r')
        axes[1, 0].set_title(f'Predicted u (L2 error: {self.compute_l2_error():.4f})')
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('y')
        plt.colorbar(im2, ax=axes[1, 0])
        
        # Absolute error
        im3 = axes[1, 1].imshow(error, extent=[0,1,0,1], origin='lower', cmap='Reds')
        axes[1, 1].set_title('Absolute Error |u_pred - u_true|')
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('y')
        plt.colorbar(im3, ax=axes[1, 1])
        
        plt.tight_layout()
        return fig
    
    def plot_loss_history(self):
        """Plot training loss curves"""
        n_components = len(self.history['loss_components'])
        
        if n_components == 0:
            # Only total loss
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.semilogy(self.history['loss'])
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Total Loss')
            ax.set_title('Training Loss')
            ax.grid(True)
        else:
            # Total + components
            fig, axes = plt.subplots(1, n_components + 1, figsize=(5*(n_components+1), 4))
            
            # Total loss
            axes[0].semilogy(self.history['loss'])
            axes[0].set_xlabel('Iteration')
            axes[0].set_ylabel('Total Loss')
            axes[0].set_title('Total Loss')
            axes[0].grid(True)
            
            # Component losses
            for idx, (name, values) in enumerate(self.history['loss_components'].items()):
                axes[idx + 1].semilogy(values)
                axes[idx + 1].set_xlabel('Iteration')
                axes[idx + 1].set_ylabel(f'{name} Loss')
                axes[idx + 1].set_title(f'{name} Loss')
                axes[idx + 1].grid(True)
        
        plt.tight_layout()
        return fig


class PoissonPINN(BaseSolver):
    def __init__(self, f_grid, u_grid, X_grid, Y_grid, hidden_dim=128, n_layers=4, lambda_bc=10.0):
        super().__init__(f_grid, u_grid, X_grid, Y_grid, hidden_dim, n_layers)
        self.lambda_bc = lambda_bc

    def prepare_training_data(self):
        X_interior = self.X_grid[1:-1, 1:-1]
        Y_interior = self.Y_grid[1:-1, 1:-1]
        f_interior = self.f_grid[1:-1, 1:-1]

        xy_collocation = np.stack([
            X_interior.flatten(),
            Y_interior.flatten()
        ], axis=1)

        f_collocation = f_interior.flatten().reshape(-1, 1)

        self.xy_collocation = torch.tensor(xy_collocation, dtype=torch.float32, requires_grad=True)
        self.f_collocation = torch.tensor(f_collocation, dtype=torch.float32)
            
        # Boundary points
        N = self.X_grid.shape[0]
        bottom = np.stack([self.X_grid[0, :], self.Y_grid[0, :]], axis=1)
        top = np.stack([self.X_grid[-1, :], self.Y_grid[-1, :]], axis=1)
        left = np.stack([self.X_grid[1:-1, 0], self.Y_grid[1:-1, 0]], axis=1)
        right = np.stack([self.X_grid[1:-1, -1], self.Y_grid[1:-1, -1]], axis=1)

        xy_boundary = np.vstack([bottom, top, left, right])
        self.xy_boundary = torch.tensor(xy_boundary, dtype=torch.float32, requires_grad=True)


    def compute_laplacian(self, xy):
        xy.requires_grad_(True)
        u = self.model(xy)
        
        u_grad = torch.autograd.grad(
            u, xy, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]

        u_x = u_grad[:, 0:1]
        u_y = u_grad[:, 1:2]

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
        laplacian = self.compute_laplacian(self.xy_collocation)
        residual_pde = -laplacian - self.f_collocation
        loss_pde = torch.mean(residual_pde**2)

        u_boundary = self.model(self.xy_boundary)
        loss_bc = torch.mean(u_boundary**2)

        loss_total = loss_pde + self.lambda_bc * loss_bc
        
        #print(f"Loss: {loss_total.item():.6f} | PDE: {loss_pde.item():.6f} | BC: {loss_bc.item():.6f}")
        
        return {
            'total': loss_total,
            'PDE': loss_pde,
            'BC': loss_bc
        }


class DataDrivenSolver(BaseSolver):
    def __init__(self, f_grid, u_grid, X_grid, Y_grid, hidden_dim=128, n_layers=4):
        super().__init__(f_grid, u_grid, X_grid, Y_grid, hidden_dim, n_layers)

    def prepare_training_data(self):
        xy_train = np.stack([
            self.X_grid.flatten(),
            self.Y_grid.flatten()
        ], axis=1)
        
        u_train = self.u_grid.flatten().reshape(-1, 1)
        
        self.xy_train = torch.tensor(xy_train, dtype=torch.float32)
        self.u_train = torch.tensor(u_train, dtype=torch.float32)

    def compute_loss(self):
        u_pred = self.model(self.xy_train)
        loss_mse = torch.mean((u_pred - self.u_train)**2)

        #print(f"Loss: {loss_mse.item():.6f}")

        return {
            'total': loss_mse,
            'MSE': loss_mse
        }
        
    

# Generate data
N = 64
K = 16
f, u = generate_sample(N, K)
X, Y = create_grid(N)

print("="*60)
print("PINN Approach")
print("="*60)

# Train PINN
pinn = PoissonPINN(
    f_grid=f, u_grid=u, X_grid=X, Y_grid=Y,
    hidden_dim=128, n_layers=4, lambda_bc=10.0
)

print(f"Collocation points: {pinn.xy_collocation.shape[0]}")
print(f"Boundary points: {pinn.xy_boundary.shape[0]}")

pinn.fit(epochs=2000, lr=1e-3)
pinn.fit_lbfgs(max_iter=500)

print(f"PINN L2 Error: {pinn.compute_l2_error():.6f}")

print("\n" + "="*60)
print("Data-Driven Approach")
print("="*60)

# Train Data-Driven (same network architecture!)
dd = DataDrivenSolver(
    f_grid=f, u_grid=u, X_grid=X, Y_grid=Y,
    hidden_dim=128, n_layers=4
)

print(f"Training points: {dd.xy_train.shape[0]}")

dd.fit(epochs=2000, lr=1e-3)
dd.fit_lbfgs(max_iter=500)

print(f"Data-Driven L2 Error: {dd.compute_l2_error():.6f}")

print("\n" + "="*60)
print("Comparison")
print("="*60)
print(f"PINN:        {pinn.compute_l2_error():.6f}")
print(f"Data-Driven: {dd.compute_l2_error():.6f}")

# Visualize
fig1 = pinn.plot_results()
fig1.suptitle('PINN Results', fontsize=16)

fig2 = dd.plot_results()
fig2.suptitle('Data-Driven Results', fontsize=16)

fig3 = pinn.plot_loss_history()
fig3.suptitle('PINN Loss History', fontsize=16)

fig4 = dd.plot_loss_history()
fig4.suptitle('Data-Driven Loss History', fontsize=16)

plt.show()