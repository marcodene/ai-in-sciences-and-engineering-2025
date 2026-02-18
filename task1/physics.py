import numpy as np
import torch


def create_grid(N):
    """
    Create spatial grid [0,1] x [0,1]
    
    Args:
        N: Grid resolution
    
    Returns:
        X, Y: Meshgrid arrays of shape (N, N)
    """
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    return X, Y


def generate_coefficients(K, seed=None):
    """
    Generate random coefficients from N(0,1)
    
    Args:
        K: Frequency parameter
        seed: Random seed for reproducibility
    
    Returns:
        a_ij: Coefficient matrix of shape (K, K)
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.randn(K, K)


def compute_source_and_solution(X, Y, a_ij, K):
    """
    Compute source term f and solution u for the Poisson equation.
    
    -Δu = f in D = [0,1]²
    u = 0 on ∂D
    
    f(x,y) = (π/K²) Σ a_ij · (i²+j²)^r · sin(πix)sin(πjy)
    u(x,y) = (1/πK²) Σ a_ij · (i²+j²)^(r-1) · sin(πix)sin(πjy)
    
    where r = 0.5
    
    Args:
        X: X-coordinates meshgrid
        Y: Y-coordinates meshgrid
        a_ij: Coefficient matrix (K, K)
        K: Frequency parameter
    
    Returns:
        f: Source term, same shape as X
        u: Analytical solution, same shape as X
    """
    f = np.zeros_like(X)
    u = np.zeros_like(X)
    
    r = 0.5  # Exponent parameter
    
    for i in range(1, K + 1):
        for j in range(1, K + 1):
            coeff = a_ij[i-1, j-1]
            freq_weight_f = (i**2 + j**2)**r
            freq_weight_u = (i**2 + j**2)**(r - 1)
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
    
    r = 0.5  # Exponent parameter
    
    for i in range(1, K + 1):
        for j in range(1, K + 1):
            coeff = a_ij[i-1, j-1]
            freq_weight = (i**2 + j**2)**(r - 1)
            spatial = np.sin(np.pi * i * x) * np.sin(np.pi * j * y)
            u += coeff * freq_weight * spatial
    
    u *= (1.0 / (np.pi * K**2))
    return u.flatten()


def compute_source_torch(xy, a_ij_tensor, K):
    """
    Compute source term f at arbitrary points using PyTorch.
    
    Used for PINN training where we need gradients.
    
    Args:
        xy: torch.Tensor of shape (N, 2) with coordinates
        a_ij_tensor: torch.Tensor of shape (K, K) with coefficients
        K: frequency parameter
    
    Returns:
        f: torch.Tensor of shape (N, 1) with source values
    """
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    f = torch.zeros(len(x), 1)
    
    r = 0.5  # Exponent parameter
    
    for i in range(1, K + 1):
        for j in range(1, K + 1):
            coeff = a_ij_tensor[i-1, j-1]
            freq_weight = (i**2 + j**2)**r
            spatial = torch.sin(np.pi * i * x) * torch.sin(np.pi * j * y)
            f += coeff * freq_weight * spatial
    
    f *= (np.pi / K**2)
    return f
