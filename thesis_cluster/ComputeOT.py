import torch
from torch import nn
from tqdm import tqdm
import numpy as np

# Data sampler

def sample_data_gen(dataset, batch_size):
    while True:
        indices = np.random.choice(len(dataset), batch_size)
        yield dataset[indices]

# Class for computing Optimal Transport

class ComputeOT:
    def __init__(self, input_dim, f_model, g_model, lr_f, lr_g):
        self.input_dim = input_dim
        self.f_model = f_model
        self.g_model = g_model

        # optimizers
        self.optimizer_f = torch.optim.Adam(self.f_model.parameters(), lr=lr_f)
        self.optimizer_g = torch.optim.Adam(self.g_model.parameters(), lr=lr_g)

        # diagnostics !delete later
        self.losses_f = []
        self.W2s = []
        self.reg_term_list_g = []

    def learn(self, x_train, y_train, lambda_reg, batch_size, iters, inner_loop_iters):
        """
        Function for learning the optimal transport
            x_train: torch.tensor
            y_train: torch.tensor
            lambda_reg: float
            batch_size: int
            iters: int
            inner_loop_iters: int
        """
        # batches generators
        data_gen_x = sample_data_gen(x_train, batch_size)
        data_gen_y = sample_data_gen(y_train, batch_size)

        loop_outer = tqdm(range(iters))

        for _ in loop_outer:
            losses_g = [] #! delete

            for _ in range(inner_loop_iters):
                y_batch = next(data_gen_y).clone().detach().requires_grad_(True)
                # Compute the gradient of the potential
                gy = self.g_model(y_batch)
                grad_gy, = torch.autograd.grad(gy, y_batch, grad_outputs=torch.ones_like(gy), create_graph=True)

                reg_term = self.g_model.positive_constraint_loss()
                self.reg_term_list_g.append(reg_term.item()) # ! delete later

                loss_g = torch.mean(self.f_model(grad_gy) - torch.diag(y_batch @ grad_gy.T)) + lambda_reg*reg_term
                losses_g.append(loss_g.item())

                # Backward pass
                self.optimizer_g.zero_grad()
                loss_g.backward()
                self.optimizer_g.step()

                # scheduler_g.step()

                # Enforce g convex
                self.g_model.enforce_positive_weights() # oss.: learning is a lot more stable with this enforced!

                # loop_inner.set_postfix(loss=loss_g.item())

            # plt.plot(losses_g)
            # plt.show()

            x_batch = next(data_gen_x)
            y_batch = next(data_gen_y).clone().detach().requires_grad_(True)

            # torch.autograd.set_detect_anomaly(True)
            gy_for_f = self.g_model(y_batch)
            grad_gy_for_f, = torch.autograd.grad(gy_for_f, y_batch, grad_outputs=torch.ones_like(gy), retain_graph=True)

            loss_f = torch.mean(self.f_model(x_batch) - self.f_model(grad_gy_for_f))
            self.losses_f.append(loss_f.item())

            # Backward pass
            self.optimizer_f.zero_grad()
            loss_f.backward()
            self.optimizer_f.step()

            # scheduler_f.step()

            # impose f convex: project f weights
            self.f_model.enforce_positive_weights()

            self.W2s.append(W2_squared(x_batch, y_batch, self.f_model, self.g_model))

            # print loss
            loop_outer.set_postfix(loss=loss_f.item(), W2_train=self.W2s[-1])

def W2_squared(X, Y, f, g):
    Y = Y.clone().detach().requires_grad_(True)
    gy = g(Y)
    grad_gy, = torch.autograd.grad(gy, Y, grad_outputs=torch.ones_like(gy))
    output = torch.mean(f(grad_gy) - f(X)  - torch.diag(Y @ grad_gy.T) + torch.sum(0.5*X**2 + 0.5*Y**2, dim=1, keepdim=True)).item()
    return output
