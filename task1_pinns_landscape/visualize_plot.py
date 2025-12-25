from new_task import PoissonPINN, DataDrivenSolver, generate_coefficients
import os

K=1
N=64
a_ij = generate_coefficients(K, seed=0)
pretrained_model_path = "models/"


model_pretrained = PoissonPINN(
            N=N, K=K, a_ij=a_ij,
            n_collocation=10000,
            hidden_dim=256, n_layers=6,
            lambda_bc=10.0
        )

if os.path.exists(pretrained_model_path):
    model_pretrained.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    print(f"Loaded pretrained model from {pretrained_model_path}")
else:
    print(f"WARNING: Pretrained model not found at {pretrained_model_path}")
    print("Please run train_all2all.py first to train the model!")
    exit(1)