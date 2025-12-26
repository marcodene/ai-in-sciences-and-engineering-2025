from new_task import PoissonPINN, DataDrivenSolver, generate_coefficients
import torch 
import os

def get_random_directions(model, norm='filter', ignore='biasbn'):
    directions = []

    for _ in range(2):
        direction = []

        # Generate random direction for each parameter
        for param in model.parameters():
            d = torch.randn_like(param)
            direction.append(d)

        # Ingore bias and batch norm
        for d, param in zip(direction, model.parameters()):
                if param.dim() <= 1:
                    d.fill_(0)

        # Filter normalization
        for d, param in zip(direction, model.parameters()):
            if torch.all(d == 0):
                continue

            if param.dim() == 2:
                for i in range(param.shape[0]):
                    neuron_norm = torch.norm(param[i]) + 1e-10
                    d_neuron_norm = torch.norm(d[i])
                    d[i] = d[i] * (neuron_norm / d_neuron_norm)

            directions.append(direction)
    
    return directions[0], directions[1]

def compute_loss_on_grid(solver, base_weights, delta, eta, alpha_range, beta_range, n_points=41):
    alphas = np.linspace(alpha_range[0], alpha_range[1], n_points)
    betas = np.linspace(beta_range[0], beta_range[1], n_points)
    losses = np.zeros((len(betas), len(alphas)))

    solver.model.eval()

    total_points = len(alphas) * len(betas)
    pbar = tqdm(total=total_points, desc="Computing loss landscape")
    
    for i, beta in enumerate(betas):
        for j, alpha in enumerate(alphas):
            set_weights(solver.model, base_weights, (delta, eta), alpha, beta)

            with torch.set_grad_enabled(True):
                loss_dict = solver.compute_loss()
                loss_value = loss_dict['total'].item()

                if not np.isfinite(loss_value):
                    loss_value = np.nan

                losses[i, j] = loss_value

            solver.model.zero_grad()
            pbar.update(1)
    pbar.close()
            


def main():
    K=1
    N=64
    a_ij = generate_coefficients(K, seed=0)
    pretrained_model_path = "checkpoints/PINN_K1.pt"


    model_pretrained = PoissonPINN(
                N=N, K=K, a_ij=a_ij,
                n_collocation=10000,
                hidden_dim=256, n_layers=6,
                lambda_bc=10.0
            )

    if os.path.exists(pretrained_model_path):
        model_pretrained.model.load_state_dict(torch.load(pretrained_model_path))
        print(f"Loaded pretrained model from {pretrained_model_path}")
    else:
        print(f"WARNING: Pretrained model not found at {pretrained_model_path}")
        print("Please run train_all2all.py first to train the model!")
        exit(1)

    theta_star = model_pretrained.model.state_dict()
    print(f"\nExtracted theta_star with {len(theta_star)} parameter tensors:")
    for name, param in theta_star.items():
        print(f"  {name}: {param.shape}")

    get_random_directions(model_pretrained.model)


if __name__ == '__main__':
    main()