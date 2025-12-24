import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

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


class PoissonPINN:
    def __init__(self, n_int_, n_sb_, K, coefficients, mode, hidden_dim=128, n_layers=4):
        self.n_int = n_int_
        self.n_sb = n_sb_
        self.K = K
        self.coefficients = coefficients
        self.mode = mode

        self.domain_extrema = torch.tensor([[0, 1],  
                                            [0, 1]]) 
        self.space_dimensions = 2

        # Parameter to balance role of data and PDE
        self.lambda_u = 10

        self.approximate_solution = MLP(hidden_dim=hidden_dim, num_hidden_layers=n_layers)

        # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        # Training sets S_sb, S_int as torch dataloader
        self.training_set_sb, self.training_set_int = self.assemble_datasets()

    ################################################################################################
    # Function to linearly transform a tensor whose value are between 0 and 1
    # to a tensor whose values are between the domain extrema
    def convert(self, tens):
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]

    def source_term(self, inputs):
        x = inputs[:, 0]
        y = inputs[:, 1]

        f = torch.zeros_like(x)
        r = 0.5

        for i in range(1, self.K + 1):
            for j in range(1, self.K + 1):
                a_ij = self.coefficients[i-1, j-1]

                freq_weight = (i**2 + j**2)**r
                spatial_pattern = torch.sin(np.pi * i * x) * torch.sin(np.pi * j * y)

                f += a_ij * freq_weight * spatial_pattern
        
        f = (np.pi / (self.K**2)) * f

        return f.reshape(-1, 1)

    def exact_solution(self, inputs):
        x = inputs[:, 0]
        y = inputs[:, 1]

        u = torch.zeros_like(x)  # Renamed from 'f' to 'u' for clarity
        r = 0.5

        for i in range(1, self.K + 1):
            for j in range(1, self.K + 1):
                a_ij = self.coefficients[i-1, j-1]

                freq_weight = (i**2 + j**2) ** (r - 1)
                spatial_pattern = torch.sin(np.pi * i * x) * torch.sin(np.pi * j * y)

                u += a_ij * freq_weight * spatial_pattern
        
        u = (1 / (np.pi * self.K**2)) * u

        return u.reshape(-1, 1)

    # Function returning the input-output tensor required to assemble the training set S_sb corresponding to the spatial boundary
    def add_spatial_boundary_points(self):
        x0 = self.domain_extrema[0, 0] # x_min = 0
        x1 = self.domain_extrema[0, 1] # x_max = 1
        y0 = self.domain_extrema[1, 0] # y_min = 0
        y1 = self.domain_extrema[1, 1] # y_max = 1

        # Generate base points using Sobol sampling
        # We'll need n_sb points for each of the 4 edges
        input_sb = self.convert(self.soboleng.draw(self.n_sb))

        # ============== LEFT EDGE: x = 0, y ∈ [0, 1] ==============
        input_sb_left = torch.clone(input_sb)
        input_sb_left[:, 0] = torch.full(input_sb_left[:, 0].shape, x0)
        # y coordinate is already randomized from Sobol
        
        # ============== RIGHT EDGE: x = 1, y ∈ [0, 1] ==============
        input_sb_right = torch.clone(input_sb)
        input_sb_right[:, 0] = torch.full(input_sb_right[:, 0].shape, x1)
        
        # ============== BOTTOM EDGE: x ∈ [0, 1], y = 0 ==============
        input_sb_bottom = torch.clone(input_sb)
        input_sb_bottom[:, 1] = torch.full(input_sb_bottom[:, 1].shape, y0)
        # x coordinate is already randomized from Sobol
        
        # ============== TOP EDGE: x ∈ [0, 1], y = 1 ==============
        input_sb_top = torch.clone(input_sb)
        input_sb_top[:, 1] = torch.full(input_sb_top[:, 1].shape, y1)

        # All boundary points should have u = 0
        output_sb_left = torch.zeros((input_sb.shape[0], 1))
        output_sb_right = torch.zeros((input_sb.shape[0], 1))
        output_sb_bottom = torch.zeros((input_sb.shape[0], 1))
        output_sb_top = torch.zeros((input_sb.shape[0], 1))

        # Concatenate all 4 edges
        input_sb_all = torch.cat([input_sb_left, input_sb_right, 
                                input_sb_bottom, input_sb_top], dim=0)
        output_sb_all = torch.cat([output_sb_left, output_sb_right, 
                                output_sb_bottom, output_sb_top], dim=0)
        
        return input_sb_all, output_sb_all

    #  Function returning the input-output tensor required to assemble the training set S_int corresponding to the interior domain where the PDE is enforced
    def add_interior_points(self):
        input_int = self.convert(self.soboleng.draw(self.n_int))
        output_int = torch.zeros((input_int.shape[0], 1))
        return input_int, output_int

    # Function returning the training sets S_sb, S_int as dataloader
    def assemble_datasets(self):
        """
        Assemble training datasets for spatial boundaries and interior points
        """
        input_sb, output_sb = self.add_spatial_boundary_points()  # 4 edges
        input_int, output_int = self.add_interior_points()        # Interior
        
        training_set_sb = DataLoader(
            torch.utils.data.TensorDataset(input_sb, output_sb), 
            batch_size=4*self.n_sb,  # 4 edges
            shuffle=False
        )
        
        training_set_int = DataLoader(
            torch.utils.data.TensorDataset(input_int, output_int), 
            batch_size=self.n_int, 
            shuffle=False
        )
        
        return training_set_sb, training_set_int

    ################################################################################################
    # Function to compute the terms required in the definition of the SPATIAL boundary residual
    def apply_boundary_conditions(self, input_sb):
        u_pred_sb = self.approximate_solution(input_sb)

        return u_pred_sb

    # Function to compute the PDE residuals
    def compute_pde_residual(self, input_int):
        input_int.requires_grad = True

        # Forward pass: get network prediction u(x,y)
        u = self.approximate_solution(input_int)

        # ============== FIRST DERIVATIVES ==============
        # Compute ∇u = [∂u/∂x, ∂u/∂y]
        grad_u = torch.autograd.grad(
            outputs=u.sum(), 
            inputs=input_int, 
            create_graph=True
        )[0]
        
        grad_u_x = grad_u[:, 0]  # ∂u/∂x
        grad_u_y = grad_u[:, 1]  # ∂u/∂y
        
        # ============== SECOND DERIVATIVES ==============
        # Compute ∂²u/∂x²
        grad_u_xx = torch.autograd.grad(
            outputs=grad_u_x.sum(), 
            inputs=input_int, 
            create_graph=True
        )[0][:, 0]
        
        # Compute ∂²u/∂y²
        grad_u_yy = torch.autograd.grad(
            outputs=grad_u_y.sum(), 
            inputs=input_int, 
            create_graph=True
        )[0][:, 1]
        
        # ============== LAPLACIAN ==============
        # Δu = ∂²u/∂x² + ∂²u/∂y²
        laplacian_u = grad_u_xx + grad_u_yy
        
        # ============== SOURCE TERM ==============
        # Compute f(x,y) at the interior points
        f = self.source_term(input_int).squeeze()  # [n] to match laplacian_u shape
        
        # ============== PDE RESIDUAL ==============
        # Poisson equation: -Δu = f
        # Residual: r = -Δu - f (should be ≈ 0)
        residual = -laplacian_u - f  # Both tensors are [n], no broadcasting
        
        return residual

    def compute_data_driven_loss(self, input_points):
        u_pred = self.approximate_solution(input_points)
        u_exact = self.exact_solution(input_points)
        loss = torch.mean((u_pred - u_exact)**2)
        return loss

    # Function to compute the total loss (weighted sum of spatial boundary loss, temporal boundary loss and interior loss)
    def compute_loss(self, inp_train_sb, u_train_sb, inp_train_int, epoch, verbose=True, optimizer_type='adam'):
        """
        Compute the total loss for either PINN or Data-Driven mode
        
        PINN mode: L = λ_u * L_boundary + L_pde
        Data-Driven mode: L = L_supervised
        
        Args:
            optimizer_type: 'adam' or 'lbfgs' - controls logging frequency
        """
        if self.mode == 'pinn':
            # ============== PINN LOSS ==============
        
            # 1. BOUNDARY LOSS: enforce u = 0 on boundaries
            u_pred_sb = self.apply_boundary_conditions(inp_train_sb)

            assert u_pred_sb.shape[1] == u_train_sb.shape[1]

            r_sb = u_train_sb - u_pred_sb
            loss_sb = torch.mean(abs(r_sb)**2)

            # 2. PDE RESIDUAL LOSS: enforce -Δu = f in the interior
            r_int = self.compute_pde_residual(inp_train_int)
            loss_int = torch.mean(abs(r_int) ** 2)

            # 3. COMBINED LOSS
            loss = self.lambda_u * loss_sb + loss_int

            # Only print for Adam optimizer or at epoch boundaries for L-BFGS
            if verbose and (epoch + 1) % 10 == 0 and optimizer_type == 'adam':
                print(f"Total loss: {loss.item():.6e} | "
                    f"Boundary Loss: {loss_sb.item():.6e} | "
                    f"PDE Loss: {loss_int.item():.6e}")

            return loss
        
        elif self.mode == 'data':
            # ============== DATA-DRIVEN LOSS ==============
            
            # Pure supervised learning: compare predictions to exact solution
            u_pred_int = self.approximate_solution(inp_train_int)
            u_exact_int = self.exact_solution(inp_train_int)
            
            # Simple MSE loss
            loss = torch.mean((u_pred_int - u_exact_int) ** 2)
            
            # Only print for Adam optimizer or at epoch boundaries for L-BFGS
            if verbose and (epoch + 1) % 10 == 0 and optimizer_type == 'adam':   
                print(f"Data-Driven Loss: {loss.item():.6f}")
            
            return loss
            
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'pinn' or 'data'")

    ################################################################################################
    def fit(self, num_epochs_adam, num_epochs_lbfgs, optimizer_adam, optimizer_lbfgs, verbose=True):
        """
        Train the network using a two-phase optimization strategy:
        1. Phase 1: Adam optimizer for fast initial convergence
        2. Phase 2: L-BFGS optimizer for fine-tuning
        
        Args:
            num_epochs_adam: Number of epochs for Adam phase
            num_epochs_lbfgs: Number of epochs for L-BFGS phase
            optimizer_adam: Adam optimizer instance
            optimizer_lbfgs: L-BFGS optimizer instance
            verbose: Whether to print training progress
        """
        history = list()

        # ============== PHASE 1: ADAM OPTIMIZER ==============
        print("=" * 60)
        print("PHASE 1: Training with Adam optimizer")
        print("=" * 60)

        # Loop over epochs
        for epoch in range(num_epochs_adam):
            if verbose and (epoch + 1) % 10 == 0: print(f"\n{'='*20} Epoch {epoch+1}/{num_epochs_adam} {'='*20}")

            for j, ((inp_train_sb, u_train_sb), (inp_train_int, u_train_int)) in enumerate(zip(self.training_set_sb, self.training_set_int)):
                def closure():
                    optimizer_adam.zero_grad()
                    loss = self.compute_loss(
                        inp_train_sb, u_train_sb, 
                        inp_train_int, 
                        epoch=epoch,
                        verbose=verbose,
                        optimizer_type='adam'
                    )
                    loss.backward()
                    return loss

                loss = optimizer_adam.step(closure)
                history.append(loss.item())

        print(f"\nAdam Phase Complete. Final Loss: {history[-1]:.6f}")

        # ============== PHASE 2: L-BFGS OPTIMIZER ==============
        print("\n" + "=" * 60)
        print("PHASE 2: Fine-tuning with L-BFGS optimizer")
        print("=" * 60)
        
        for epoch in range(num_epochs_lbfgs):
            if verbose and (epoch + 1) % 10 == 0:
                print(f"\n{'='*20} Epoch {epoch+1}/{num_epochs_lbfgs} {'='*20}")
            
            # Loop over batches
            for j, ((inp_train_sb, u_train_sb), (inp_train_int, u_train_int)) in enumerate(
                zip(self.training_set_sb, self.training_set_int)
            ):
                def closure():
                    optimizer_lbfgs.zero_grad()
                    loss = self.compute_loss(
                        inp_train_sb, u_train_sb, 
                        inp_train_int, 
                        epoch=epoch,
                        verbose=False,  # Disable per-closure printing
                        optimizer_type='lbfgs'
                    )
                    loss.backward()
                    return loss
                
                loss = optimizer_lbfgs.step(closure)
                history.append(loss.item())
            
            # Print summary after epoch completes
            if verbose and (epoch + 1) % 10 == 0:
                # Simply print the last computed loss from history
                print(f"Loss: {history[-1]:.6e}")
        
        print("\n" + "=" * 60)
        print(f"Training Complete! Final Loss: {history[-1]:.6f}")
        print("=" * 60)
        
        return history

    ################################################################################################
    def plotting(self):
        inputs = self.soboleng.draw(100000)
        inputs = self.convert(inputs)

        output = self.approximate_solution(inputs).reshape(-1, )
        exact_output = self.exact_solution(inputs).reshape(-1, )

        fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
        
        im1 = axs[0].scatter(inputs[:, 0].detach(), inputs[:, 1].detach(), 
                            c=exact_output.detach(), cmap="jet")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y") 
        plt.colorbar(im1, ax=axs[0])
        axs[0].grid(True, which="both", ls=":")
        
        im2 = axs[1].scatter(inputs[:, 0].detach(), inputs[:, 1].detach(), 
                            c=output.detach(), cmap="jet")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("y")
        plt.colorbar(im2, ax=axs[1])
        axs[1].grid(True, which="both", ls=":")
        
        axs[0].set_title("Exact Solution")
        axs[1].set_title("Approximate Solution")

        plt.show()

        err = (torch.mean((output - exact_output) ** 2) / torch.mean(exact_output ** 2)) ** 0.5 * 100
        print("L2 Relative Error Norm: ", err.item(), "%")
