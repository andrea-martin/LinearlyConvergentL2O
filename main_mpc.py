import os
import torch
import torch.nn as nn   
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Set device and seeds
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1234)
np.random.seed(1234)

class LinearDynamicalSystem:
    def __init__(self, A, B, x0=None, device='cpu'):
        self.A = A.to(device)
        self.B = B.to(device)
        self.n = self.A.shape[0]
        self.m = self.B.shape[1]
        self.x = x0.to(device) if x0 is not None else torch.zeros((self.n,), device=device)
        self.device = device

    def step(self, u):
        u = u.to(self.device)
        self.x = self.A @ self.x + self.B @ u
        return self.x

    def reset(self, x0=None):
        self.x = x0.to(self.device) if x0 is not None else torch.zeros((self.n, 1), device=self.device)

    def get_matrices(self):
        return self.A, self.B
    
class FiniteHorizonOCP:
    def __init__(self, sys, Qt, Qf, Rt, T):
        self.sys = sys 
        self.Qt = Qt
        self.Qf = Qf
        self.Rt = Rt
        self.T = T
        self.device = self.sys.device

        self.F, self.G = self.compute_F_G()
        self.Q, self.R = self.build_block_diagonal()

        self.H = 2 * (self.G.mT @ self.Q @ self.G + self.R)
        _, S, _ = torch.linalg.svd(self.H)
        self.lambda_max_H = S.max().item()  # Maximum eigenvalue of H
        self.lambda_min_H = S.min().item()  # Minimum eigenvalue of H

    def compute_F_G(self):
        n = self.sys.n
        m = self.sys.m
        T = self.T

        F = torch.zeros(((T + 1) * n, n), device=self.device)
        F[0:n, :] = torch.eye(n, device=self.device)
        for t in range(1, T + 1):
            F[t*n : (t+1)*n, :] = torch.linalg.matrix_power(self.sys.A, t)

        G = torch.zeros(((T + 1) * n, T * m), device=self.device)
        for row in range(1, T + 1): 
            for col in range(row):
                A_power = torch.linalg.matrix_power(self.sys.A, row - col - 1)
                G_block = A_power @ self.sys.B
                G[row*n:(row+1)*n, col*m:(col+1)*m] = G_block

        return F, G
    
    def block_diag_torch(self, *matrices):
        """Constructs a block diagonal matrix from a list of 2D torch tensors."""
        shapes = [m.shape for m in matrices]
        total_rows = sum(s[0] for s in shapes)
        total_cols = sum(s[1] for s in shapes)
        
        result = torch.zeros((total_rows, total_cols), dtype=matrices[0].dtype, device=matrices[0].device)
        
        current_row, current_col = 0, 0
        for m in matrices:
            rows, cols = m.shape
            result[current_row:current_row+rows, current_col:current_col+cols] = m
            current_row += rows
            current_col += cols
            
        return result

    def build_block_diagonal(self):
        Q_blocks = [self.Qt] * self.T + [self.Qf]
        Q = self.block_diag_torch(*Q_blocks)

        R_blocks = [self.Rt] * self.T
        R = self.block_diag_torch(*R_blocks)

        return Q, R

    def cost(self, U):
        x = self.F @ self.sys.x + self.G @ U
        return x.mT @ self.Q @ x + U.mT @ self.R @ U
    
class ModelPredictiveControlDataset(Dataset):
    """
    Generates quadratic optimization problems of the form:
        minimize  0.5 u^T Q u + c(x0)^T u + q
        subject to  A u <= b(x0)
    """
    def __init__(self, ocp, num_samples):
        self.ocp = ocp
        self.num_samples = num_samples
        self.device = ocp.device     

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x0 = 1 * (torch.rand(self.ocp.sys.n, 1, device=self.device) - 0.5) #+ torch.ones(self.ocp.sys.n, 1, device=self.device)

        Q = self.ocp.H
        c = (2 * x0.mT @ self.ocp.F.mT @ self.ocp.Q @ self.ocp.G).mT
        q = x0.mT @ self.ocp.F.mT @ self.ocp.Q @ self.ocp.F @ x0

        A = torch.vstack([
            torch.eye(self.ocp.T * self.ocp.sys.m, self.ocp.T * self.ocp.sys.m, device=self.device),
            -torch.eye(self.ocp.T * self.ocp.sys.m, self.ocp.T * self.ocp.sys.m, device=self.device)
        ])
        b = torch.vstack([
            0.75 * torch.ones(self.ocp.T * self.ocp.sys.m, 1, device=self.device),
            0.75 * torch.ones(self.ocp.T * self.ocp.sys.m, 1, device=self.device)
        ])

        return Q, c, q, A, b

class TwoLayerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device=torch.device('cpu')):
        super().__init__()

        self.rnn1 = nn.LSTMCell(input_size, hidden_size, device=device)
        self.rnn2 = nn.LSTMCell(hidden_size, hidden_size, device=device)
        self.output_layer = nn.Linear(hidden_size, output_size, device=device)  

        self.h1_0 = nn.Parameter((torch.randn(1, hidden_size, device=device) * 0.01)) 
        self.c1_0 = nn.Parameter((torch.randn(1, hidden_size, device=device) * 0.01))
        self.h2_0 = nn.Parameter((torch.randn(1, hidden_size, device=device) * 0.01))
        self.c2_0 = nn.Parameter((torch.randn(1, hidden_size, device=device) * 0.01))

        self.h1, self.c1 = None, None
        self.h2, self.c2 = None, None

    def forward(self, u, h1, c1, h2, c2):
        
        (h1_, c1_) = self.rnn1(u, (h1, c1))
        (h2_, c2_) = self.rnn2(h1_, (h2, c2))

        return self.output_layer(h2_), (h1_, c1_), (h2_, c2_)
    
class TwoLayerLSTMOptimizer(nn.Module):
    def __init__(self, d, m, gamma, device=torch.device('cpu')):
        super().__init__()
        self.lstm = TwoLayerLSTM(input_size=2*d+1, hidden_size=5, output_size=d, device=device)
        self.m = m
        self.poly_coefficients = nn.Parameter(torch.full((m+1,), -2.0, device=device)) # Polynomial coefficients for the evolution before softplus transformation
        self.gamma = gamma

    def forward(self, bx, bl, bg, t):
        batch_size = bx.shape[0]
        bv = []
        for b in range(batch_size):
            # Scaling the feature vector: current iterate, loss, and gradient
            x = bx[b] / torch.norm(bx[b], p=float('inf')) if torch.norm(bx[b], p=float('inf')) > 0 else bx[b]
            l = bl[b] / torch.norm(bl[b], p=float('inf')) if torch.norm(bl[b], p=float('inf')) > 0 else bl[b]
            g = bg[b] / torch.norm(bg[b], p=float('inf')) if torch.norm(bg[b], p=float('inf')) > 0 else bg[b]

            u = torch.vstack([x, l, g]).T
            if t == 0:
                v, (self.h1, self.c1), (self.h2, self.c2) = self.lstm.forward(u, self.lstm.h1_0, self.lstm.c1_0, self.lstm.h2_0, self.lstm.c2_0)
            else:
                v, (self.h1, self.c1), (self.h2, self.c2) = self.lstm.forward(u, self.h1, self.c1, self.h2, self.c2)
            # Scaling the evolution v
            v = torch.tanh(v).squeeze(0) * torch.norm(bx[b], p=float('inf')) if torch.norm(bx[b], p=float('inf')) > 0 else torch.tanh(v).squeeze(0)
            bv.append(v)
        
        bv = torch.stack(bv, dim=0)
        # Compute the polynomial value
        poly_value = torch.dot(F.softplus(self.poly_coefficients), torch.tensor([t ** j for j in range(self.m + 1)], device=bx.device, dtype=bx.dtype))
        bv = poly_value * bv * (self.gamma ** t)

        return bv
    
def gradient_descent(U0, g, l, iterations, step_size):
    U = [U0]
    J = [l(U0)]
    for _ in range(iterations):
        U.append(U[-1] - step_size * g(U[-1]))
        J.append(l(U[-1]))
    return torch.cat(U, dim=1), torch.cat(J, dim=1)

def nesterov_accelerated_gradient_descent(U0, g, l, iterations, step_size, mu):
    U = [U0]
    J = [l(U0)]
    U_prev = U0.clone()
    for _ in range(iterations):
        y = U[-1] + mu * (U[-1] - U_prev)
        U_prev = U[-1].clone()
        U.append(y - step_size * g(y))
        J.append(l(U[-1]))
    return torch.cat(U, dim=1), torch.cat(J, dim=1)

def projected_gradient_descent(U0, g, l, project, iterations, step_size):
    U = [project(U0)]
    J = [l(project(U0))]
    for _ in range(iterations):
        U.append(project(U[-1] - step_size * g(U[-1])))
        J.append(l(U[-1]))
    return torch.cat(U, dim=1), torch.cat(J, dim=1)

def agmon_projection(A, b, x0, max_iters, lambda_=1.0): # Iteratively enforce Ax <= b
    m, n = A.shape
    row_norms_sq = (A.pow(2)).sum(dim=1)    # shape (m,)
    x = x0
    # violations = []

    for _ in range(max_iters):
        # violation vector d_i = a_i^T x - b_i
        d = A.matmul(x) - b                  # (m,)
        max_v, idx = d.max(dim=0)            # both are 0-dim tensors

        if max_v <= 0:
            break
        # violations.append(max_v)
        # projection step
        x = x - lambda_ * (max_v / row_norms_sq[idx]) * A[idx]

    # stack into one tensor
    # violations = torch.stack(violations)     # (max_iters,)
    return x

def meta_training_mpc(model, dataloader, meta_optimizer, initial_guess, step_size, max_iterations, epochs, device):

    dataloader_iter = iter(dataloader)
    model.train()

    for epoch in range(epochs):
        meta_optimizer.zero_grad()
        meta_loss = torch.zeros(1, device=device)

        Ub = initial_guess.clone()
        try:
            Qb, cb, qb, Ab, bb = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            Qb, cb, qb, Ab, bb = next(dataloader_iter)

        gb = lambda U: Qb @ U + cb
        lb = lambda U: 0.5 * U.mT @ Qb @ U + cb.mT @ U + qb
        project = lambda U: torch.clamp(U, min=-0.75, max=0.75)

        for i in range(max_iterations):
            vb = model(Ub, lb(Ub), gb(Ub), i).unsqueeze(-1)

            Ub = project(Ub - step_size * gb(Ub))

            for b in range(vb.shape[0]):
                v_agmon = agmon_projection(Ab[b], (bb[b] - Ab[b] @ Ub[b]).squeeze(-1), vb[b].squeeze(-1), max_iters=100, lambda_=1.0)
                vb[b] = v_agmon.unsqueeze(-1)

            Ub = Ub + vb

            meta_loss = meta_loss + lb(Ub).sum()

        print(f"Epoch {epoch+1} / {epochs} - average meta loss: {(meta_loss / max_iterations).item():.2f}")

        meta_loss.backward()
        meta_optimizer.step()

def l2o_descent(U0, model, g, l, project, A, b, iterations, step_size):
    U = [project(U0)]
    J = [l(project(U0))]
    for i in range(iterations):
        # Compute the update using the learned optimizer
        v = model(U[-1], J[-1], g(U[-1]), i)

        U_base = project(U[-1] - step_size * g(U[-1]))

        for _ in range(v.shape[0]):
            v_agmon = agmon_projection(A[_], (b[_] - A[_] @ U_base[_]).squeeze(-1), v[_].squeeze(-1), max_iters=100, lambda_=1.0)
            v[_] = v_agmon#.unsqueeze(-1)

        U.append(U_base + v.unsqueeze(-1))
        J.append(l(U[-1]))
    return torch.cat(U, dim=1), torch.cat(J, dim=1)

def main():

    # A = torch.rand(1000, 200) -0.5
    # b = torch.rand(1000, 1) -0.5
    # x = torch.rand(200,) -0.5
    # agmon_relaxation_torch(A, b, x, max_iters=1000)

    # torch.manual_seed(0)
    # m, n = 1000, 200
    # # create a random feasible system A x <= b
    # A = torch.randn(m, n, requires_grad=True)
    # x_true = torch.randn(n)
    # slack = torch.rand(m) * 0.5 + 0.1
    # b = A.matmul(x_true) + slack
    # # initial guess
    # x = torch.zeros(n, requires_grad=True)
    # x_final = agmon_projection(A, b, x, max_iters=1000)

    # # use the final violation as a toy loss and backprop
    # loss = violations[-1]
    # loss.backward()

    # print(f"Initial violation: {violations[0].item():.3e}")
    # print(f"Final violation:   {violations[-1].item():.3e}")
    # print(f"∥∂loss/∂x0∥ = {x.grad.norm().item():.3e}")
    # print(f"∥∂loss/∂A∥  = {A.grad.norm().item():.3e}")

    # # plot convergence
    # plt.semilogy(torch.arange(1, violations.size(0)+1).numpy(),
    #              violations.detach().cpu().numpy(),
    #              marker='o', linestyle='-')
    # plt.xlabel("Iteration")
    # plt.ylabel("Max Violation")
    # plt.title("Agmon Relaxation Convergence (PyTorch)")
    # plt.grid(True, which='both', ls='--', lw=0.5)
    # plt.tight_layout()
    # plt.show()



    A = torch.tensor([[1.0, 1.0], [0.0, 1.0]], device=device)
    B = torch.tensor([[0.0], [1.0]], device=device)
    sys = LinearDynamicalSystem(A, B, x0=None, device=device)

    Qt = torch.eye(2, 2, device=device)
    Qf = torch.eye(2, 2, device=device)
    Rt = torch.tensor([[1.0]], device=device)
    T = 10  # Planning horizon for the finite horizon optimal control problem
    ocp = FiniteHorizonOCP(sys, Qt, Qf, Rt, T)

    kappa = ocp.lambda_max_H / ocp.lambda_min_H
    print(f"\nCondition number of the finite horizon planning problem: {kappa:.2f}")

    # Tuning parameters for the optimization algorithms according to Proposition 1 in "Analysis and design of optimization algorithms via integral quadratic constraints" by Lessard et al.
    eta_gd = 1 / ocp.lambda_max_H # Popular tuning for GD 
    eta_pgd = 1 / ocp.lambda_max_H
    eta_nag, mu_nag = 1 / ocp.lambda_max_H, (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1) # Popular tuning for NAG
    # eta_gd = 2 / (ocp.lambda_max_H + ocp.lambda_min_H)  # Optimal tuning for GD
    # eta_nag, mu_nag = 4 / (3 * ocp.lambda_max_H + ocp.lambda_min_H), (np.sqrt(3 * kappa + 1) - 2) / (np.sqrt(3 * kappa + 1) + 2)  # Optimal tuning for NAG
    
    max_iterations = 100

    training_samples, training_batch_size = 2048, 64
    training_dataset = ModelPredictiveControlDataset(ocp, num_samples=training_samples)
    training_dataloader = DataLoader(training_dataset, batch_size=training_batch_size, shuffle=True)

    epochs = 100  # Number of epochs for meta-training
    U0b = torch.zeros((training_batch_size, sys.m * T, 1), device=device)



    # flag = False
    # for Qb, cb, qb, Ab, bb in training_dataloader:

    #     gb = lambda U: Qb @ U + cb
    #     lb = lambda U: 0.5 * U.mT @ Qb @ U + cb.mT @ U + qb
        
    #     # print(f"\nOptimal value of unconstrained optimization problem: {lb(-torch.linalg.inv(Qb) @ cb).item():.2f}")

    #     Ub_gd, Jb_gd = gradient_descent(U0b.clone(), gb, lb, iterations=max_iterations, step_size=eta_gd)
        
    #     if not flag:
    #         J_gd = Jb_gd
    #         flag = True
    #     else:
    #         J_gd = torch.vstack([J_gd, Jb_gd])




    learned_update = TwoLayerLSTMOptimizer(d=sys.m * T, m=0, gamma=0.95, device=device)
    print(f"Total parameters in LearnedUpdate: {sum(p.numel() for p in learned_update.parameters() if p.requires_grad)}")
    meta_optimizer = torch.optim.Adam(learned_update.parameters(), lr=1e-2)
    meta_training_mpc(model=learned_update, dataloader=training_dataloader, meta_optimizer=meta_optimizer, initial_guess=U0b, step_size=eta_pgd, max_iterations=max_iterations, epochs=epochs, device=device)
    learned_update.eval()

    test_samples, test_batch_size = 256, 16
    test_dataset = ModelPredictiveControlDataset(ocp, num_samples=test_samples)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

    U0b = torch.zeros((test_batch_size, sys.m * T, 1), device=device)

    flag = False
    for Qb, cb, qb, Ab, bb in test_dataloader:

        gb = lambda U: Qb @ U + cb
        lb = lambda U: 0.5 * U.mT @ Qb @ U + cb.mT @ U + qb
        project = lambda U: torch.clamp(U, min=-0.75, max=0.75) # Project onto the box constraints

        # print(f"\nOptimal value of unconstrained optimization problem: {lb(-torch.linalg.inv(Qb) @ cb).item():.2f}")

        Ub_gd, Jb_gd = gradient_descent(U0b.clone(), gb, lb, iterations=max_iterations, step_size=eta_gd)
        Ub_nag, Jb_nag = nesterov_accelerated_gradient_descent(U0b.clone(), gb, lb, iterations=max_iterations, step_size=eta_nag, mu=mu_nag)
        Ub_pgd, Jb_pgd = projected_gradient_descent(U0b.clone(), gb, lb, project, iterations=max_iterations, step_size=eta_pgd)

        Ub_l2o, Jb_l2o = l2o_descent(U0b.clone(), learned_update, gb, lb, project, Ab, bb, iterations=max_iterations, step_size=eta_pgd)

        if not flag:
            J_gd, J_pgd, J_nag, J_l2o = Jb_gd, Jb_pgd, Jb_nag, Jb_l2o
            flag = True
        else:
            J_gd = torch.vstack([J_gd, Jb_gd])
            J_pgd = torch.vstack([J_pgd, Jb_pgd])
            J_nag = torch.vstack([J_nag, Jb_nag])
            J_l2o = torch.vstack([J_l2o, Jb_l2o])

    J_gd_mean = torch.mean(J_gd, dim=0).squeeze().cpu()
    J_gd_std = torch.std(J_gd, dim=0).squeeze().cpu()

    J_pgd_mean = torch.mean(J_pgd, dim=0).squeeze().cpu()
    J_pgd_std = torch.std(J_pgd, dim=0).squeeze().cpu()

    J_nag_mean = torch.mean(J_nag, dim=0).squeeze().cpu()
    J_nag_std = torch.std(J_nag, dim=0).squeeze().cpu()

    J_l2o_mean = torch.mean(J_l2o, dim=0).squeeze().cpu()
    J_l2o_std = torch.std(J_l2o, dim=0).squeeze().cpu()

    plt.figure()
    plt.yscale('log')

    plt.plot(range(max_iterations+1), J_gd_mean, label="Gradient Descent")
    plt.fill_between(range(max_iterations+1),
                    (J_gd_mean - J_gd_std).numpy(),
                    (J_gd_mean + J_gd_std).numpy(),
                    alpha=0.3)
    plt.plot(range(max_iterations+1), J_pgd_mean, label="Projected Gradient Descent")
    plt.fill_between(range(max_iterations+1),
                    (J_pgd_mean - J_pgd_std).numpy(),
                    (J_pgd_mean + J_pgd_std).numpy(),
                    alpha=0.3)
    plt.plot(range(max_iterations+1), J_nag_mean, label="Nesterov Accelerated Gradient Descent")
    plt.fill_between(range(max_iterations+1),
                    (J_nag_mean - J_nag_std).numpy(),
                    (J_nag_mean + J_nag_std).numpy(),
                    alpha=0.3)
    plt.plot(range(max_iterations+1), J_l2o_mean.detach(), label="Learned Optimizer")
    plt.fill_between(range(max_iterations+1),
                    (J_l2o_mean - J_l2o_std).detach().numpy(),
                    (J_l2o_mean + J_l2o_std).detach().numpy(),
                    alpha=0.3)
    
    # plt.plot(range(max_iterations), J_gd_mean[1:], label="Gradient Descent")
    # plt.fill_between(range(max_iterations),
    #                 (J_gd_mean[1:] - J_gd_std[1:]).numpy(),
    #                 (J_gd_mean[1:] + J_gd_std[1:]).numpy(),
    #                 alpha=0.3)
    # plt.plot(range(max_iterations), J_pgd_mean[1:], label="Projected Gradient Descent")
    # plt.fill_between(range(max_iterations),
    #                 (J_pgd_mean[1:] - J_pgd_std[1:]).numpy(),
    #                 (J_pgd_mean[1:] + J_pgd_std[1:]).numpy(),
    #                 alpha=0.3)
    # plt.plot(range(max_iterations), J_nag_mean[1:], label="Nesterov Accelerated Gradient Descent")
    # plt.fill_between(range(max_iterations),
    #                 (J_nag_mean[1:] - J_nag_std[1:]).numpy(),
    #                 (J_nag_mean[1:] + J_nag_std[1:]).numpy(),
    #                 alpha=0.3)
    # plt.plot(range(max_iterations), J_l2o_mean[1:].detach(), label="Learned Optimizer")
    # plt.fill_between(range(max_iterations),
    #                 (J_l2o_mean[1:] - J_l2o_std[1:]).detach().numpy(),
    #                 (J_l2o_mean[1:] + J_l2o_std[1:]).detach().numpy(),
    #                 alpha=0.3)

    plt.legend()
    plt.grid(True)

    # plt.figure()
    # plt.plot(range(T), Ub_gd[-1][0], label="Gradient Descent")
    # plt.plot(range(T), Ub_pgd[-1][0], label="Projected Gradient Descent")
    # plt.plot(range(T), Ub_nag[-1][0], label="Nesterov Accelerated Gradient Descent")
    # plt.legend()
    # plt.grid(True)
    # plt.title("Control Input Trajectories")
    plt.show()

    print(f"That's all folks!")
    # m = A.shape[0] # Number of rows in matrix A (dimension of output vector b)
    # d = A.shape[1] # Number of columns in matrix A (dimension of parameter vector x)

    # x0 = 1e-4 * torch.rand(d, 1, device=device) # Initial parameter vector   

    # training_samples = 1024 # Number of linear regression tasks to sample for training
    # T = 50000  # Number of iterations for the inner loop 
    # epochs = 50 # Number of epochs for meta-training  

    # # Instantiate the training dataset and dataloader
    # training_dataset = LinearRegressionDataset(m=m, d=d, num_samples=training_samples, device=device)
    # training_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True)
    
    # load = True
    # if load:
    #     # Load the pre-trained learned optimizer parameters
    #     directory_path = './trained_models/'
    #     file_path = os.path.join(directory_path, 'bcsstk02_nag_50epochs_10000T200_rho95_02randn.pt')
    #     if os.path.exists(file_path):
    #         print(f"Loading pre-trained learned optimizer from {file_path}")
    #         learned_update = LearnedUpdate(d, q=0, rho=0.95, hidden_sizes=[256, 256, 256], architecture='lstm').to(device)
    #         learned_update.load_state_dict(torch.load(file_path))
    #         print("Pre-trained learned optimizer loaded successfully.")
    #     else:
    #         print(f"Pre-trained model not found at {file_path}.")
    # else:
    #     # Initialize the LearnedUpdate module with a fixed rho (e.g., 0.99)
    #     learned_update = LearnedUpdate(d, q=0, rho=0.95, hidden_sizes=[256, 256, 256], architecture='lstm').to(device)
    #     print(f"Total parameters in LearnedUpdate: {sum(p.numel() for p in learned_update.parameters() if p.requires_grad)}")

    #     # Meta optimizer (updates both the MLP and the alpha parameters)
    #     meta_optimizer = torch.optim.Adam(learned_update.parameters(), lr=1e-3)

    #     learned_update = meta_training_nag(learned_update, A, x0, training_dataloader, meta_optimizer, T, device, epochs=epochs)
    #     # Save the trained learned optimizer parameters
    #     directory_path = './trained_models/'
    #     os.makedirs(directory_path, exist_ok=True)
    #     file_path = os.path.join(directory_path, 'bcsstk02_nag_50epochs_10000T200_rho95_02randn.pt')
    #     torch.save(learned_update.state_dict(), file_path)
    #     print(f"Trained learned optimizer saved to {file_path}")

    # test_samples = 128 # Number of linear regression tasks to sample for testing
    # # Instantiate the test dataset and dataloader
    # test_dataset = LinearRegressionDataset(m=m, d=d, num_samples=test_samples, device=device)
    # test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    # # Evaluate the learned optimizer
    # evaluate_nag(learned_update, A, x0, test_dataloader, T, device)

if __name__ == "__main__":
    main()