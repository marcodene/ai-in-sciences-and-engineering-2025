from config_task1 import TASK1_1_CONFIG, PATHS
from utils_task1 import plot_samples, plot_f_u_comparison


print("\nTASK 1.1: DATA GENERATION AND VISUALIZATION")

# Extract configuration
N = TASK1_1_CONFIG['N']
K_values = TASK1_1_CONFIG['K_values']
n_samples = TASK1_1_CONFIG['n_samples']

print(f"\nConfiguration:")
print(f"  Grid resolution: {N}x{N}")
print(f"  K values: {K_values}")
print(f"  Samples per K: {n_samples}")
print(f"  Save directory: {PATHS['plots_task1_1']}/")

# Generate and save sample visualizations
print(f"\nGenerating samples...")
plot_samples(N, K_values, n_samples, save_dir=PATHS['plots_task1_1'])

# Generate f and u comparison plot
print(f"\nGenerating f vs u comparison plot...")
plot_f_u_comparison(N=N, K_values=K_values, seed=42, save_dir=PATHS['plots_task1_1'], name_suffix="1")
plot_f_u_comparison(N=N, K_values=K_values, seed=41, save_dir=PATHS['plots_task1_1'], name_suffix="2")
plot_f_u_comparison(N=N, K_values=K_values, seed=40, save_dir=PATHS['plots_task1_1'], name_suffix="3")

print(f"\nResults saved to: {PATHS['plots_task1_1']}/")
