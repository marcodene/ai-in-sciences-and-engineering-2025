import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, hidden_dim=128, num_hidden_layers=4):
        super(MLP, self).__init__()

        self.input_layer = nn.Linear(2, hidden_dim)

        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim),
            for _ in range(num_hidden_layers-1)
        ])

        self.output_layer = nn.Linear(hidden_dim, 1)

        self.activation = nn.Tanh()
        
    def forward(self, xy):
        x = self.activation(self.input_layer(xy))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x)

def pinn_loss(model, xy_collocation, f_values, lambda_bc=10.0):
    """
    Compute PINN loss: residual + boundary conditions
    """
    pass

def data_driven_loss(model, xy_points, u_true):
    """
    Compute supervised loss: MSE
    """

def compute_l2_error(model, xy_grid, u_true):
    """
    Compute L2 relative error
    """