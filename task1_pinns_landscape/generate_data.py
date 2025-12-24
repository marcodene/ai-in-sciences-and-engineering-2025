import numpy as np
import matplotlib.pyplot as plt

def create_grid(N):
    """Create spatial grid"""
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    return X, Y

def generate_coefficients(K):
    """Generate random coefficients from N(0,1)"""
    return np.random.randn(K, K)

def compute_fields(X, Y, a_ij, K, r):
    field = np.zeros_like(X)

    for i in range(1, K + 1): # i = 1, 2, ..., K
        for j in range(1, K + 1): # j = 1, 2, ..., K
            # Get the coefficient (array is 0-indexed, formula is 1-indexed)
            coeff = a_ij[i-1, j-1]

            freq_weights = (i**2 + j**2)**r

            spatial_pattern = np.sin(np.pi * i * X) * np.sin(np.pi * j * Y)

            field += coeff * freq_weights * spatial_pattern

    field /= (K**2)

    return field

def generate_sample(N, K):
    """Generate (f, u) pair"""
    X, Y = create_grid(N)
    a_ij = generate_coefficients(K)
    f = np.pi * compute_fields(X, Y, a_ij, K, 0.5)
    u = (1 / np.pi) * compute_fields(X, Y, a_ij, K, -0.5)

    return f, u

def visualize_samples(N=64, K_values=[1, 4, 8, 16], n_samples=3):
    """Generate and visualize samples for different complexity levels"""
    figures = []
    
    for K in K_values:
        # Create figure for this K value
        # Layout: n_samples rows × 2 columns (f and u)
        fig, axes = plt.subplots(n_samples, 2, figsize=(10, 4*n_samples))
        fig.suptitle(f'Samples with K = {K}', fontsize=16, fontweight='bold')
        
        # Generate n_samples for this K
        for sample_idx in range(n_samples):
            # Generate one sample
            f, u = generate_sample(N, K)
            
            # Plot source term f (left column)
            ax_f = axes[sample_idx, 0] if n_samples > 1 else axes[0]
            im_f = ax_f.imshow(f, extent=[0, 1, 0, 1], origin='lower', 
                              cmap='RdBu_r', aspect='equal')
            ax_f.set_xlabel('x')
            ax_f.set_ylabel('y')
            ax_f.set_title(f'Source f (sample {sample_idx+1})')
            plt.colorbar(im_f, ax=ax_f, fraction=0.046, pad=0.04)
            
            # Plot solution u (right column)
            ax_u = axes[sample_idx, 1] if n_samples > 1 else axes[1]
            im_u = ax_u.imshow(u, extent=[0, 1, 0, 1], origin='lower', 
                              cmap='RdBu_r', aspect='equal')
            ax_u.set_xlabel('x')
            ax_u.set_ylabel('y')
            ax_u.set_title(f'Solution u (sample {sample_idx+1})')
            plt.colorbar(im_u, ax=ax_u, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        figures.append(fig)
    
    return figures



figures = visualize_samples(N=64, K_values=[1, 4, 8, 16], n_samples=3)

# Show all figures
plt.show()

# Optionally save them for your report
for idx, fig in enumerate(figures):
    K_val = [1, 4, 8, 16][idx]
    fig.savefig(f'plots/task1/samples_K_{K_val}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: samples_K_{K_val}.png")