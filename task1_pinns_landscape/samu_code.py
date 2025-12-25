import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

class Pinns:
    def __init__(self, n_int_, n_sb_,n_data, K):
        self.n_int = n_int_
        self.n_sb = n_sb_
        self.K = K 


        # Extrema of the solution domain (t,x) in [0,0.1]x[-1,1]
        self.domain_extrema = torch.tensor([[0, 1],  # X dimension
                                            [0, 1]])  # Y dimension

        # Number of space dimensions
        self.space_dimensions = 2

        # Parameter to balance role of data and PDE
        self.lambda_u = 10

        self.r = 0.5
        torch.manual_seed(0)
        self.a = torch.randn(self.K, self.K, requires_grad=False)

        # F Dense NN to approximate the solution of the underlying heat equation
        self.approximate_solution = NeuralNet(input_dimension=self.domain_extrema.shape[0], output_dimension=1,
                                              n_hidden_layers=4,
                                              neurons=100,
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=42)

        # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])


        self.n_data = n_data
        x_data, u_data = self.add_data_points(self.n_data)

        self.u_mean = u_data.mean()
        self.u_std  = u_data.std().clamp_min(1e-8)

        u_data_n = (u_data - self.u_mean) / self.u_std

        self.training_set_data = DataLoader(
            torch.utils.data.TensorDataset(x_data, u_data_n),
            batch_size=self.n_data,
            shuffle=False
)
        self.text_x = self.soboleng.draw(8000)  # prova 5k o 20k, non 100k

        with torch.no_grad():
            self.exact_output = self.exact_sol(self.text_x).reshape(-1)



    # Exact solution for the heat equation ut = u_xx with the IC above
    def source_term(self, xy):
        x = xy[:, 0:1]
        y = xy[:, 1:2]

        K = self.K
        r = self.r

        a = self.a.to(device=xy.device, dtype=xy.dtype)  # <-- important

        i = torch.arange(1, K+1, device=xy.device, dtype=xy.dtype).view(K, 1)
        j = torch.arange(1, K+1, device=xy.device, dtype=xy.dtype).view(1, K)

        S = i**2 + j**2
        Wf = S**r

        Bx = torch.sin(torch.pi * x * i.T)   # (N,K)
        By = torch.sin(torch.pi * y * j)     # (N,K)

        M = a * Wf
        f = ((Bx @ M) * By).sum(dim=1, keepdim=True)

        return (torch.pi / (K**2)) * f
    def exact_sol(self, xy):
        x = xy[:, 0:1]
        y = xy[:, 1:2]

        K = self.K
        r = self.r

        a = self.a.to(device=xy.device, dtype=xy.dtype)  # <-- important

        i = torch.arange(1, K+1, device=xy.device, dtype=xy.dtype).view(K, 1)
        j = torch.arange(1, K+1, device=xy.device, dtype=xy.dtype).view(1, K)

        S = i**2 + j**2
        Wf = S**(r-1)

        Bx = torch.sin(torch.pi * x * i.T)   # (N,K)
        By = torch.sin(torch.pi * y * j)     # (N,K)

        M = a * Wf
        f = ((Bx @ M) * By).sum(dim=1, keepdim=True)

        return (1 / (torch.pi*(K**2))) * f



    # Function returning the input-output tensor required to assemble the training set S_sb corresponding to the spatial boundary

    def add_data_points(self, n_data):
        x_data = self.soboleng.draw(n_data)
        with torch.no_grad():
            u_data = self.exact_sol(x_data)
        return x_data, u_data




    def compute_loss_data(self, x_data, u_data, verbose=True):
        u_pred = self.approximate_solution(x_data)

        loss = torch.log10(torch.mean((u_pred - u_data)**2)) 

        if verbose:
            print("Data loss:", loss.item())
        with torch.no_grad():
            pred_n = self.approximate_solution(self.text_x).reshape(-1)
            pred   = pred_n * self.u_std + self.u_mean

        err = (torch.mean((pred - self.exact_output)**2) / torch.mean(self.exact_output**2)).sqrt() * 100

        print("L2 Relative Error Norm:", err.item(), "%")

        return loss
    
    def fit_dd(self, num_epochs, optimizer, verbose=True):
        history = list()

        # Loop over epochs
        for epoch in range(num_epochs):
            if verbose: print("################################ ", epoch, " ################################")

            for (x_data, u_data) in self.training_set_data:

                def closure():
                    optimizer.zero_grad()
                    loss = self.compute_loss_data(x_data, u_data, verbose=verbose)
                    loss.backward()

                    history.append(loss.item())
                    return loss

                optimizer.step(closure=closure)

        print('Final Loss: ', history[-1])

        return history
    

    ################################################################################################
    def plotting(self):
        #self.approximate_solution.eval()

        inputs = self.soboleng.draw(80000)  # prova 5k o 20k, non 100k

        with torch.no_grad():
            output = self.approximate_solution(inputs).reshape(-1)
            exact_output = self.exact_sol(inputs).reshape(-1)
        output = output * self.u_std + self.u_mean

        fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
        im1 = axs[0].scatter(inputs[:, 1], inputs[:, 0], c=exact_output, s=2, cmap="jet")
        plt.colorbar(im1, ax=axs[0])
        axs[0].set_xlabel("x"); axs[0].set_ylabel("y"); axs[0].grid(True, which="both", ls=":")

        im2 = axs[1].scatter(inputs[:, 1], inputs[:, 0], c=output, s=2, cmap="jet")
        plt.colorbar(im2, ax=axs[1])
        axs[1].set_xlabel("x"); axs[1].set_ylabel("y"); axs[1].grid(True, which="both", ls=":")

        axs[0].set_title("Exact Solution")
        axs[1].set_title("Approximate Solution")
        plt.show()

        err = (torch.mean((output - exact_output) ** 2) / torch.mean(exact_output ** 2)).sqrt() * 100
        print("L2 Relative Error Norm:", err.item(), "%")




pinn_dd = Pinns(0, 0 ,8000, 16)


optimizer_LBFGS_dd = optim.LBFGS(pinn_dd.approximate_solution.parameters(),
                              lr=float(0.5),
                              max_iter=50000,
                              max_eval=50000,
                              history_size=150,
                              line_search_fn="strong_wolfe",
                              tolerance_change=1.0 * np.finfo(float).eps)

hist = pinn_dd.fit_dd(num_epochs=1,
                optimizer=optimizer_LBFGS_dd,
                verbose=True)


plt.figure(dpi=150)
plt.grid(True, which="both", ls=":")
plt.plot(np.arange(1, len(hist) + 1), hist, label="Train Loss")
plt.xscale("log")
plt.legend()

pinn_dd.plotting()