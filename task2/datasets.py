"""
Dataset classes for all tasks
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class One2OneDataset(Dataset):
    """
    Dataset for Task 1: One-to-One mapping from t=0 to t=1
    
    Input:  [x-coords, u(x, t=0)]
    Output: u(x, t=1)
    """
    def __init__(self, data_path, n_samples, resolution):
        """
        Args:
            data_path: Path to .npy file with shape (N, 5, resolution)
            n_samples: Number of samples to use
            resolution: Spatial resolution
        """
        data = torch.from_numpy(np.load(data_path)).type(torch.float32)
        
        self.X = data[:n_samples, 0, :]  # Initial conditions at t=0
        self.Y = data[:n_samples, 4, :]  # Solution at t=1
        
        # Create spatial grid
        x_grid = torch.linspace(0, 1, resolution, dtype=torch.float32)
        x_grid_rep = x_grid.unsqueeze(0).repeat(n_samples, 1)
        
        # Stack: [x-coords, u(x, t=0)] -> (n_samples, resolution, 2)
        self.X = torch.stack([x_grid_rep, self.X], dim=-1)
        
        # Add channel dimension to output -> (n_samples, resolution, 1)
        self.Y = self.Y.unsqueeze(-1)
        
        print(f"One2OneDataset: {n_samples} samples, resolution {resolution}")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class All2AllDataset(Dataset):
    """
    Dataset for all2all training: creates pairs (t_i, t_j) with t_i < t_j.
    
    Input:  [x-coords, u(x, t_i), Δt]
    Output: u(x, t_j)
    
    Where Δt = t_j - t_i
    """
    def __init__(self, data_path):
        # Load data
        self.data = np.load(data_path)  # (N_traj, 5, 128)
        self.n_trajectories = self.data.shape[0]
        self.n_times = self.data.shape[1]
        self.resolution = self.data.shape[2]
        
        # Timesteps: t = [0.0, 0.25, 0.5, 0.75, 1.0]
        self.time_values = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        
        # Spatial grid (fixed for all)
        self.x_grid = np.linspace(0, 1, self.resolution, dtype=np.float32)
        
        # Create all pairs (i, j) with i < j (only forward in time)
        self.time_pairs = [
            (i, j) for i in range(self.n_times) 
            for j in range(i + 1, self.n_times)
        ]
        self.n_pairs = len(self.time_pairs)  # 4+3+2+1 = 10 pairs
        
        # Total number of samples
        self.length = self.n_trajectories * self.n_pairs
        
        print(f"All2AllDataset: {self.length} samples ({self.n_trajectories} traj × {self.n_pairs} pairs)")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        """
        Returns:
            x_input: (128, 3) - [x-coords, u(x, t_i), Δt]
            y_target: (128, 1) - u(x, t_j)
        """
        # Map linear index -> (trajectory, time_pair)
        traj_idx = index // self.n_pairs
        pair_idx = index % self.n_pairs
        
        # Extract time indices
        i, j = self.time_pairs[pair_idx]
        
        # Extract u(x) at the two times
        u_in = self.data[traj_idx, i, :]   # (128,) - u(x, t_i)
        u_out = self.data[traj_idx, j, :]  # (128,) - u(x, t_j)
        
        # Calculate Δt = t_j - t_i
        delta_t = self.time_values[j] - self.time_values[i]
        
        # Repeat Δt for each spatial point
        delta_t_repeated = np.full(self.resolution, delta_t, dtype=np.float32)
        
        # Create input: [x-coords, u(x, t_i), Δt]
        x_input = np.stack([
            self.x_grid,
            u_in,
            delta_t_repeated
        ], axis=-1)  # (128, 3)
        
        # Target: u(x, t_j)
        y_target = u_out[..., np.newaxis]  # (128, 1)
        
        return (
            torch.from_numpy(x_input).float(),
            torch.from_numpy(y_target).float()
        )
