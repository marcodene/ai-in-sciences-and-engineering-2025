import torch
import numpy as np
import matplotlib.pyplot as plt
import os


class PoissonDataGenerator:
    def __init__(self, K, N=64):
        """
        Args:
            K: Frequency parameter (complexity indicator)
            N: Grid resolution (N×N grid)
        """
        self.K = K
        self.N = N
        self.r = 0.5
    
    def generate_sample(self, seed=None):
        """
        Generate one sample with random coefficients
        
        Returns:
            X: x-coordinates grid [N, N]
            Y: y-coordinates grid [N, N]
            f_grid: source term on grid [N, N]
            u_grid: exact solution on grid [N, N]
            coefficients: random a_ij coefficients [K, K]
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Generate random coefficients a_ij ~ N(0, 1)
        coefficients = torch.randn(self.K, self.K)

        # Create structured grid [0, 1] × [0, 1]
        x = torch.linspace(0, 1, self.N)
        y = torch.linspace(0, 1, self.N)
        X, Y = torch.meshgrid(x, y, indexing='ij')

        # Compute f and u on the grid
        f_grid = self.compute_source_term(X, Y, coefficients)
        u_grid = self.compute_exact_solution(X, Y, coefficients)

        return X, Y, f_grid, u_grid, coefficients

    def compute_source_term(self, X, Y, coefficients):
        """
        Compute source term f(x, y) on grid
        
        f(x,y) = (π/K²) Σ_{i,j=1}^K a_ij · (i² + j²)^r · sin(πix) · sin(πjy)
        """
        f = torch.zeros_like(X)
        
        for i in range(1, self.K + 1):
            for j in range(1, self.K + 1):
                a_ij = coefficients[i-1, j-1]
                
                # Frequency weight: (i² + j²)^r
                freq_weight = (i**2 + j**2)**self.r
                
                # Spatial pattern: sin(πix) sin(πjy)
                spatial_pattern = torch.sin(np.pi * i * X) * torch.sin(np.pi * j * Y)
                
                f += a_ij * freq_weight * spatial_pattern
        
        # Scale by π/K²
        f = (np.pi / self.K**2) * f
        
        return f
    
    def compute_exact_solution(self, X, Y, coefficients):
        """
        Compute exact solution u(x, y) on grid
        
        u(x,y) = (1/πK²) Σ_{i,j=1}^K a_ij · (i² + j²)^(r-1) · sin(πix) · sin(πjy)
        """
        u = torch.zeros_like(X)
        
        for i in range(1, self.K + 1):
            for j in range(1, self.K + 1):
                a_ij = coefficients[i-1, j-1]
                
                # Frequency weight: (i² + j²)^(r-1)
                freq_weight = (i**2 + j**2)**(self.r - 1)
                
                # Spatial pattern: sin(πix) sin(πjy)
                spatial_pattern = torch.sin(np.pi * i * X) * torch.sin(np.pi * j * Y)
                
                u += a_ij * freq_weight * spatial_pattern
        
        # Scale by 1/(πK²)
        u = (1.0 / (np.pi * self.K**2)) * u
        
        return u

    def plot_sample(self, X, Y, f_grid, u_grid, title_suffix="", save_path=None):
        """
        Plot one sample: source term f and solution u side by side
        
        Args:
            save_path: If provided, save the plot to this path
        """
        fig, axs = plt.subplots(1, 2, figsize=(14, 5), dpi=100)
        
        # Plot source term f
        im1 = axs[0].contourf(X.numpy(), Y.numpy(), f_grid.numpy(), 
                              levels=50, cmap='RdBu_r')
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('y')
        axs[0].set_title(f'Source Term f(x,y) - K={self.K}{title_suffix}')
        axs[0].set_aspect('equal')
        plt.colorbar(im1, ax=axs[0])
        
        # Plot exact solution u
        im2 = axs[1].contourf(X.numpy(), Y.numpy(), u_grid.numpy(), 
                              levels=50, cmap='RdBu_r')
        axs[1].set_xlabel('x')
        axs[1].set_ylabel('y')
        axs[1].set_title(f'Exact Solution u(x,y) - K={self.K}{title_suffix}')
        axs[1].set_aspect('equal')
        plt.colorbar(im2, ax=axs[1])
        
        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"  → Saved: {save_path}")
            plt.close(fig)
        else:
            plt.show()
    
    def generate_and_plot_samples(self, n_samples=3, save_dir=None):
        """
        Generate and plot multiple samples for this K value
        
        Args:
            n_samples: Number of samples to generate
            save_dir: If provided, save plots to this directory
        """
        print(f"\nGenerating {n_samples} samples for K={self.K}")
        
        samples = []
        for i in range(n_samples):
            X, Y, f_grid, u_grid, coeffs = self.generate_sample(seed=42+i)
            samples.append((X, Y, f_grid, u_grid, coeffs))
            
            # Determine save path
            if save_dir is not None:
                save_path = os.path.join(save_dir, f'K{self.K}_sample{i+1}.png')
            else:
                save_path = None
            
            # Plot
            self.plot_sample(X, Y, f_grid, u_grid, 
                           title_suffix=f" (Sample {i+1})", 
                           save_path=save_path)
        
        return samples



def task_1_1_generate_and_visualize(save_plots=True, output_dir='plots/task1'):
    """
    Complete Task 1.1: Generate and visualize samples for different K values
    
    Args:
        save_plots: If True, save plots to output_dir
        output_dir: Directory where to save plots
    """
    K_values = [1, 4, 8, 16]
    N = 64  # Grid resolution
    n_samples = 3  # Number of samples per K
    
    print("="*70)
    print("TASK 1.1: Data Generation and Visualization")
    print("="*70)

    # Create output directory ONCE if saving
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nSaving plots to: {output_dir}/\n")
    
    all_samples = {}
    
    for K in K_values:
        # Create generator for this K value
        generator = PoissonDataGenerator(K=K, N=N)

        # Pass the directory (already created) or None
        save_dir = output_dir if save_plots else None
        
        # Generate and plot samples
        samples = generator.generate_and_plot_samples(
            n_samples=n_samples,
            save_dir=save_dir
        )
        all_samples[K] = samples
    
    if save_plots:
        print(f"\n{'='*70}")
        print(f"✓ All plots saved to: {output_dir}/")
        print(f"{'='*70}\n")
    
    return all_samples


def plot_comparison_across_K(save_plot=True, output_dir='plots/task1'):
    """
    Create a comparison plot showing one sample for each K value
    
    Args:
        save_plot: If True, save plot to output_dir
        output_dir: Directory where to save plot
    """
    K_values = [1, 4, 8, 16]
    N = 64
    
    fig, axs = plt.subplots(2, 4, figsize=(20, 10), dpi=100)
    
    for idx, K in enumerate(K_values):
        generator = PoissonDataGenerator(K=K, N=N)
        X, Y, f_grid, u_grid, _ = generator.generate_sample(seed=42)
        
        # Plot source term
        im1 = axs[0, idx].contourf(X.numpy(), Y.numpy(), f_grid.numpy(), 
                                    levels=50, cmap='RdBu_r')
        axs[0, idx].set_title(f'f(x,y) - K={K}')
        axs[0, idx].set_xlabel('x')
        axs[0, idx].set_ylabel('y')
        axs[0, idx].set_aspect('equal')
        plt.colorbar(im1, ax=axs[0, idx])
        
        # Plot solution
        im2 = axs[1, idx].contourf(X.numpy(), Y.numpy(), u_grid.numpy(), 
                                    levels=50, cmap='RdBu_r')
        axs[1, idx].set_title(f'u(x,y) - K={K}')
        axs[1, idx].set_xlabel('x')
        axs[1, idx].set_ylabel('y')
        axs[1, idx].set_aspect('equal')
        plt.colorbar(im2, ax=axs[1, idx])
    
    plt.suptitle('Comparison Across Different Complexity Levels (K)', 
                 fontsize=16, y=0.995)
    plt.tight_layout()
    
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)  # Solo qui!
        save_path = os.path.join(output_dir, 'comparison_all_K.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"  → Saved: {save_path}\n")
        plt.close(fig)
    else:
        plt.show()


task_1_1_generate_and_visualize(save_plots=True, output_dir='plots/task1_1')
plot_comparison_across_K(save_plot=True, output_dir='plots/task1_1')