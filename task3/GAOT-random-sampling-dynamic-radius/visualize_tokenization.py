import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from omegaconf import OmegaConf

# Add current directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.datasets.data_processor import DataProcessor
from src.datasets.dataset import DATASET_METADATA
from src.datasets.graph_builder import GraphBuilder
from src.utils.scaling import CoordinateScaler

def load_config(config_path):
    """Load configuration from file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return OmegaConf.load(config_path)

def plot_tokenization(tokens, physical_points, radius, title, domain, filename=None, circle_alpha=0.05):
    """
    Plot physical points, tokens, and their coverage radius.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Handle radius statistics and per-token radius
    if np.ndim(radius) == 0:
        # Scalar case
        radii = np.full(len(tokens), radius)
        r_mean = float(radius)
        r_var = 0.0
    else:
        # Array case
        radii = np.array(radius)
        if len(radii) != len(tokens):
             if len(radii) == 1:
                r_mean = float(radii[0])
                r_var = 0.0
                radii = np.full(len(tokens), radii[0])
             else:
                raise ValueError(f"Radius array length {len(radii)} does not match tokens length {len(tokens)}")
        else:
             r_mean = np.mean(radii)
             r_var = np.var(radii)

    x_min, y_min = domain[0]
    x_max, y_max = domain[1]
    
    # 0. Plot Domain Boundary
    width = x_max - x_min
    height = y_max - y_min
    rect = Rectangle((x_min, y_min), width, height, 
                     linewidth=2, edgecolor='black', facecolor='none', 
                     label='Domain', zorder=5)
    ax.add_patch(rect)

    # 1. Plot physical points (mesh) - Make them more visible
    if physical_points is not None:
        ax.scatter(physical_points[:, 0], physical_points[:, 1], 
                   s=10, alpha=0.4, c='gray', label='Physical Points', zorder=1)
    
    # 2. Plot lines from tokens to radius visualization (optional/implicit)
    
    # 3. Plot coverage radius (Filled)
    # We plot this before tokens so tokens appear on top, or use zorder
    for i in range(len(tokens)):
        # Use a single label for legend
        label = 'Coverage Radius' if i == 0 else None
        circle = Circle(xy=(tokens[i, 0], tokens[i, 1]), 
                        radius=radii[i],
                        color='blue', fill=True, alpha=circle_alpha, zorder=2, label=label)
        ax.add_patch(circle)

    # 4. Plot tokens (Red Points)
    ax.scatter(tokens[:, 0], tokens[:, 1], 
               s=20, c='red', marker='o', label='Latent Tokens', zorder=3)
        
    stats_text = f"Radius Mean: {r_mean:.4f} | Radius Var: {r_var:.6f}"
    ax.set_title(f"{title}\n{stats_text}", fontsize=14)
    ax.set_xlim(x_min - 0.1, x_max + 0.1)
    ax.set_ylim(y_min - 0.1, y_max + 0.1)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    if filename:
        os.makedirs("plots", exist_ok=True)
        save_path = os.path.join("plots", filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()

def main():
    # Configuration
    # We use the elasticity config as a base
    config_path = os.path.join("config", "examples", "time_indep", "elasticity_random_fixed.json")
    
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        print(f"Config file {config_path} not found. Please ensure you are running from the project root.")
        return

    # Basic setup
    dataset_config = config.dataset
    model_config = config.model

    # Inject missing config keys required by DataProcessor
    if 'rand_dataset' not in dataset_config:
        dataset_config.rand_dataset = False
    if 'coord_scaling' not in dataset_config:
        dataset_config.coord_scaling = 'per_dim_scaling'
    
    # Get metadata
    if dataset_config.metaname not in DATASET_METADATA:
        raise ValueError(f"Metadata for {dataset_config.metaname} not found")
    metadata = DATASET_METADATA[dataset_config.metaname]
    
    # Initialize DataProcessor
    print("Initializing DataProcessor...")
    # Fix dataset path if necessary (assuming running from GAOT-random-sampling-dynamic-radius root)
    if not os.path.exists(dataset_config.base_path):
        # Try finding it in the parent context if needed, but for now assume standard structure
        print(f"Warning: Dataset path {dataset_config.base_path} check failed. Ensure data is present.")

    data_processor = DataProcessor(
        dataset_config=dataset_config,
        metadata=metadata
    )
    
    # Load data to get physical points
    # We need to perform load_and_process_data to get x_train
    try:
        data_splits, is_variable_coords = data_processor.load_and_process_data()
        
        # Extract physical points
        x_train = data_splits['train']['x']
        if is_variable_coords:
            physical_points = x_train[0] # Take first sample
        else:
            physical_points = x_train # Fixed coordinates
            
        # Ensure it's on CPU and numpy for plotting
        if isinstance(physical_points, torch.Tensor):
            physical_points_np = physical_points.detach().cpu().numpy()
            physical_points_tensor = physical_points # Keep tensor for generator
        else:
            physical_points_np = physical_points
            physical_points_tensor = torch.tensor(physical_points, dtype=torch.float32)

    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using dummy physical points for visualization logic check (if data is missing).")
        print("WARNING: THE VISUALIZATION WILL NOT REPRESENT THE TRUE GEOMETRY.")
        # Dummy data for robustness if real data is missing during dev
        x = np.linspace(metadata.domain_x[0][0], metadata.domain_x[1][0], 50)
        y = np.linspace(metadata.domain_x[0][1], metadata.domain_x[1][1], 50)
        xv, yv = np.meshgrid(x, y)
        physical_points_np = np.stack([xv.flatten(), yv.flatten()], axis=1)
        physical_points_tensor = torch.tensor(physical_points_np, dtype=torch.float32)
    
    # Initialize and fit CoordinateScaler on physical points
    if data_processor.coord_scaler is None:
        data_processor.coord_scaler = CoordinateScaler(
            target_range=(-1, 1),
            mode=dataset_config.coord_scaling
        )
        data_processor.coord_scaler.fit(physical_points_tensor)
        print("Explicitly fitted CoordinateScaler on physical points.")

    domain = metadata.domain_x
    #radius = model_config.args.magno.radius
    radius = 0.125
    
    # Calculate radius scaling factor from scaled space [-1, 1] to physical space
    domain_width = domain[1][0] - domain[0][0]
    scaled_width = 2.0 # range (-1, 1) has width 2
    radius_scaling_factor = domain_width / scaled_width
    viz_radius = radius * radius_scaling_factor
    print(f"Radius correction: Model Radius ({radius}) -> Visualization Radius ({viz_radius:.4f})")

    # --- Initial Grid (Baseline) ---
    print("\nVisualizing Strategy I: Structured Grid...")
    # Assume 256 tokens -> 16x16 grid
    grid_size = (16, 16)
    tokens_grid = data_processor.generate_latent_queries(
        token_size=grid_size,
        strategy='grid'
    )
    
    tokens_grid = data_processor.coord_scaler.inverse_transform(tokens_grid)
        
    plot_tokenization(
        tokens=tokens_grid.detach().cpu().numpy(),
        physical_points=physical_points_np,
        radius=viz_radius, 
        title="Strategy I: Structured Grid Sampling",
        domain=domain,
        filename="strategy_1_grid.png"
    )

    # --- Random Sampling with Fixed Radius (Uniform) ---
    print("\nVisualizing Strategy II: Random Sampling (Uniform)...")
    n_tokens = 256
    tokens_random_uniform = data_processor.generate_latent_queries(
        token_size=[n_tokens],
        strategy='random',
        geometry_aware=False,
        seed=42
    )
    
    tokens_random_uniform = data_processor.coord_scaler.inverse_transform(tokens_random_uniform)
        
    plot_tokenization(
        tokens=tokens_random_uniform.detach().cpu().numpy(),
        physical_points=physical_points_np,
        radius=viz_radius, 
        title="Strategy II: Random Sampling (Uniform, Fixed Radius)",
        domain=domain,
        filename="strategy_2_random_uniform.png"
    )

    # --- Random Sampling Geometry Aware ---
    print("\nVisualizing Strategy III: Random Sampling (Geometry Aware)...")
    tokens_random_geo = data_processor.generate_latent_queries(
        token_size=[n_tokens],
        strategy='random',
        physical_points=physical_points_tensor,
        geometry_aware=True,
        seed=42
    )
    
    tokens_random_geo = data_processor.coord_scaler.inverse_transform(tokens_random_geo)
        
    plot_tokenization(
        tokens=tokens_random_geo.detach().cpu().numpy(),
        physical_points=physical_points_np,
        radius=viz_radius,
        title="Strategy III: Random Sampling (Geometry-Aware, Fixed Radius)",
        domain=domain,
        filename="strategy_3_random_geo.png"
    )

    # --- Random Sampling + Dynamic Radius ---
    print("\nVisualizing Strategy IV: Random Sampling + Dynamic Radius...")
    
    # 1. Generate tokens
    tokens_dynamic = data_processor.generate_latent_queries(
        token_size=[n_tokens],
        strategy='random',
        physical_points=physical_points_tensor,
        geometry_aware=True,
        seed=42
    )

    # 2. Compute dynamic radii
    # We need to instantiate GraphBuilder
    graph_builder = GraphBuilder(neighbor_search_method='native') # Use native for safety locally
    
    # Params from config or defaults
    alpha = 1.2
    k = 3
    
    # Compute in latent space [-1, 1]
    radii_latent = graph_builder.compute_dynamic_radii(
        latent_coords=tokens_dynamic, 
        k=k, 
        alpha=alpha
    )
    
    # 3. Scale back to physical space for visualization
    # Tokens to physical
    tokens_dynamic_phys = data_processor.coord_scaler.inverse_transform(tokens_dynamic)
    
    # Radii to physical (Assuming isotropic scaling)
    # radius_phys = radius_latent * scaling_factor
    radii_phys = radii_latent * radius_scaling_factor

    # Convert to numpy
    tokens_dynamic_phys_np = tokens_dynamic_phys.detach().cpu().numpy()
    radii_phys_np = radii_phys.detach().cpu().numpy()

    plot_tokenization(
        tokens=tokens_dynamic_phys_np,
        physical_points=physical_points_np,
        radius=radii_phys_np, # Valid array of radii
        title=f"Strategy IV: Random Sampling + Dynamic Radius (k={k}, α={alpha})",
        domain=domain,
        filename="strategy_4_dynamic_radius.png",
        circle_alpha=0.1 # Slightly higher alpha to see density differences
    )
    
    print("\nDone! Plots saved to 'plots/' directory.")

if __name__ == "__main__":
    main()
