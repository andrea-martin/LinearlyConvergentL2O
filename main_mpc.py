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
        self.x = x0.to(device) if x0 is not None else torch.zeros((self.n, 1), device=device)
        self.device = device

    def step(self, u, noisy=False):
        u = u.to(self.device)
        self.x = self.A @ self.x + self.B @ u
        if noisy: 
            self.x = self.x + torch.randn((self.n, 1), device=self.device) * 0.01  # Add small noise
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
    
    def generate_qp(self, x0):
        Q = self.H
        c = (2 * x0.mT @ self.F.mT @ self.Q @ self.G).mT
        q = x0.mT @ self.F.mT @ self.Q @ self.F @ x0

        A = torch.vstack([
            torch.eye(self.T * self.sys.m, self.T * self.sys.m, device=self.device),
            -torch.eye(self.T * self.sys.m, self.T * self.sys.m, device=self.device)
        ])
        b = torch.vstack([
            0.25 * torch.ones(self.T * self.sys.m, 1, device=self.device),
            0.25 * torch.ones(self.T * self.sys.m, 1, device=self.device)
        ])

        return Q, c, q, A, b
    
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
        # x0 = 2 * (torch.rand(self.ocp.sys.n, 1, device=self.device) - 0.5) #+ torch.ones(self.ocp.sys.n, 1, device=self.device)
        # Define standard deviations for each component (adjust the values as needed)
        stds = torch.tensor([0.5, 0.5], device=self.device).unsqueeze(1)
        x0 = torch.randn((self.ocp.sys.n, 1), device=self.device) * stds
        # multipliers = torch.tensor([5, 0.5, 0.5, 0.5, 0.5], device=self.device).unsqueeze(1)
        # x0 = torch.tensor([[5.0], [0.0], [0.0], [0.0], [0.0]], device=self.device) + (torch.rand((5, 1), device=self.device) - 0.5) * 0 #multipliers
        return self.ocp.generate_qp(x0)
    
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


        # print("I did something")

    # stack into one tensor
    # violations = torch.stack(violations)     # (max_iters,)
    return x

def meta_training_mpc(model, dataloader, meta_optimizer, initial_guess, step_size, max_iterations, epochs, device):

    dataloader_iter = iter(dataloader)
    model.train()

    # Save the trained learned optimizer parameters
    directory_path = './trained_mpc_solvers/'
    os.makedirs(directory_path, exist_ok=True)

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
        project = lambda U: torch.clamp(U, min=-0.25, max=0.25)

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

        if (epoch+1) % 5 == 0:
            file_path = os.path.join(directory_path, f'learned_optimizer_aircraft_e{epoch+1}.pt')
            torch.save(model.state_dict(), file_path)
            print(f"Trained learned optimizer saved to {file_path}")

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

def run_mpc_loop_gd(ocp, eta_gd, max_iterations, control_horizon):
    X = [ocp.sys.x]
    for t in range(control_horizon):
        # print(f"Current state at time {t}: {ocp.sys.x.squeeze().cpu().numpy()}")
        Q, c, q, A, b = ocp.generate_qp(ocp.sys.x)
        U = torch.zeros((1, ocp.sys.m * ocp.T, 1), device=device)

        g = lambda U: Q @ U + c
        l = lambda U: 0.5 * U.mT @ Q @ U + c.mT @ U + q
        
        U, J = gradient_descent(U, g, l, iterations=max_iterations, step_size=eta_gd)
        U = U[:, -ocp.sys.m * ocp.T:-ocp.sys.m * (ocp.T - 1), :].squeeze(0)
        # print(f"Optimal control input at time {t}: {U.squeeze().cpu().numpy()}\n")
        ocp.sys.step(U)

        X.append(ocp.sys.x)

    return torch.cat(X, dim=1)

def run_mpc_loop_pgd(ocp, eta_pgd, max_iterations, control_horizon):
    X = [ocp.sys.x]
    U = []
    for t in range(control_horizon):
        # print(f"Current state at time {t}: {ocp.sys.x.squeeze().cpu().numpy()}")
        Q, c, q, A, b = ocp.generate_qp(ocp.sys.x)
        u = torch.zeros((1, ocp.sys.m * ocp.T, 1), device=device)
        g = lambda U: Q @ U + c
        l = lambda U: 0.5 * U.mT @ Q @ U + c.mT @ U + q
        project = lambda U: torch.clamp(U, min=-0.25, max=0.25)

        u, J = projected_gradient_descent(u, g, l, project, iterations=max_iterations, step_size=eta_pgd)
        u = u[:, -ocp.sys.m * ocp.T:-ocp.sys.m * (ocp.T - 1), :].squeeze(0)
        # print(f"Optimal control input at time {t}: {U.squeeze().cpu().numpy()}\n")
        ocp.sys.step(u, noisy=False)

        X.append(ocp.sys.x)
        U.append(u)

    return torch.cat(X, dim=1), torch.cat(U, dim=1)

def run_mpc_loop_l2o(ocp, learned_update, eta_pgd, max_iterations, control_horizon):
    X = [ocp.sys.x]
    U = []
    for t in range(control_horizon):
        # print(f"Current state at time {t}: {ocp.sys.x.squeeze().cpu().numpy()}")
        Q, c, q, A, b = ocp.generate_qp(ocp.sys.x)
        Q, c, q, A, b = Q.unsqueeze(0), c.unsqueeze(0), q.unsqueeze(0), A.unsqueeze(0), b.unsqueeze(0)  # Add batch dimension for compatibility
        u = torch.zeros((1, ocp.sys.m * ocp.T, 1), device=device)
        g = lambda U: Q @ U + c
        l = lambda U: 0.5 * U.mT @ Q @ U + c.mT @ U + q
        project = lambda U: torch.clamp(U, min=-0.25, max=0.25)

        u, J = l2o_descent(u, learned_update, g, l, project, A, b, iterations=max_iterations, step_size=eta_pgd)
        if (u > 0.25).any() or (u < -0.25).any():
            print("Violations!")
        u = u[:, -ocp.sys.m * ocp.T:-ocp.sys.m * (ocp.T - 1), :].squeeze(0)
        # print(f"Optimal control input at time {t}: {U.squeeze().cpu().numpy()}\n")
        ocp.sys.step(u, noisy=False)

        X.append(ocp.sys.x)
        U.append(u)

    return torch.cat(X, dim=1), torch.cat(U, dim=1)

def main():
    # Initialize the linear dynamical system
    A = torch.tensor([[1.0, 1.0], [0.0, 1.0]], device=device)
    B = torch.tensor([[0.0], [1.0]], device=device)
    
    # x0 = 2 * (torch.rand(A.shape[0], 1, device=device) - 0.5)
    stds = torch.tensor([0.5, 0.5], device=device).unsqueeze(1)
    x0 = torch.randn((A.shape[0], 1), device=device) * stds
    sys = LinearDynamicalSystem(A, B, x0=x0, device=device)

    Qt = torch.eye(2, 2, device=device)
    Qf = torch.eye(2, 2, device=device)
    Rt = torch.tensor([[1.0]], device=device)
    T = 20  # Planning horizon for the finite horizon optimal control problem



    # Ac = torch.tensor([[0.99, 0.01, 0.18, -0.09,   0],
    #                 [   0, 0.94,    0,  0.29,   0],
    #                 [   0, 0.14, 0.81,  -0.9,   0],
    #                 [   0, -0.2,    0,  0.95,   0],
    #                 [   0, 0.09,    0,     0, 0.9]], device=device)
    # Bc = torch.tensor([[ 0.01, -0.02],
    #                 [-0.14,     0],
    #                 [ 0.05,  -0.2],
    #                 [ 0.02,     0],
    #                 [-0.01, 0]], device=device)
    # Ts = 0.2 # Sampling time
    # A = torch.eye(5, device=device) + Ac * Ts
    # B = Bc * Ts
    # x0 = torch.tensor([[10], [0], [0], [0], [0]], device=device)  # Initial state
    # sys = LinearDynamicalSystem(A, B, x0=x0, device=device)

    # Qt = 10 * torch.eye(5, 5, device=device)
    # Qf = 10 * torch.eye(5, 5, device=device)
    # Rt = torch.tensor([[3, 0], [0, 2]], device=device)
    # T = 20  # Planning horizon for the finite horizon optimal control problem

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
    perform_meta_training = False

    if perform_meta_training:

        training_samples, training_batch_size = 4096, 128
        training_dataset = ModelPredictiveControlDataset(ocp, num_samples=training_samples)
        training_dataloader = DataLoader(training_dataset, batch_size=training_batch_size, shuffle=True)

        epochs = 100  # Number of epochs for meta-training
        U0b = torch.zeros((training_batch_size, sys.m * T, 1), device=device)

        learned_update = TwoLayerLSTMOptimizer(d=sys.m * T, m=0, gamma=0.95, device=device)
        print(f"Total parameters in LearnedUpdate: {sum(p.numel() for p in learned_update.parameters() if p.requires_grad)}")
        meta_optimizer = torch.optim.Adam(learned_update.parameters(), lr=.5e-3)
        meta_training_mpc(model=learned_update, dataloader=training_dataloader, meta_optimizer=meta_optimizer, initial_guess=U0b, step_size=eta_pgd, max_iterations=max_iterations, epochs=epochs, device=device)
    else:
        print("Skipping meta-training. Using pre-trained learned optimizer.")
        # Load the pre-trained learned optimizer parameters
        directory_path = './trained_mpc_solvers/'
        file_path = os.path.join(directory_path, 'learned_optimizer_x0g05_T20_upm025_e65.pt')
        if os.path.exists(file_path):
            print(f"Loading pre-trained learned optimizer from {file_path}")
            learned_update = TwoLayerLSTMOptimizer(d=sys.m * T, m=0, gamma=0.95, device=device)
            learned_update.load_state_dict(torch.load(file_path))
            print("Pre-trained learned optimizer loaded successfully.")
        else:
            print(f"Pre-trained model not found at {file_path}.")

    learned_update.eval()


    data_file = 'mpc_cost_data.npz'
    
    if False:#not os.path.exists(data_file):
        print("Starting MPC loop with different solvers...")
        with torch.no_grad():
            control_horizon = 30  # Control horizon for the MPC loop
            ocp_test = FiniteHorizonOCP(sys, Qt, Qf, Rt, control_horizon)
            n_iter = 100
            cost_pgd_a = torch.empty(n_iter, device=device)
            cost_pgd_b = torch.empty(n_iter, device=device)
            cost_pgd_c = torch.empty(n_iter, device=device)
            cost_pgd_0 = torch.empty(n_iter, device=device)
            cost_pgd_1 = torch.empty(n_iter, device=device)
            cost_pgd_2 = torch.empty(n_iter, device=device)
            cost_pgd_3 = torch.empty(n_iter, device=device)
            cost_pgd_4 = torch.empty(n_iter, device=device)
            cost_pgd_5 = torch.empty(n_iter, device=device)
            cost_pgd_6 = torch.empty(n_iter, device=device)
            cost_pgd_7 = torch.empty(n_iter, device=device)

            cost_l2o_a = torch.empty(n_iter, device=device)
            cost_l2o_b = torch.empty(n_iter, device=device)
            cost_l2o_c = torch.empty(n_iter, device=device)
            cost_l2o_0 = torch.empty(n_iter, device=device) 
            cost_l2o_1 = torch.empty(n_iter, device=device)
            cost_l2o_2 = torch.empty(n_iter, device=device)
            cost_l2o_3 = torch.empty(n_iter, device=device)
            cost_l2o_4 = torch.empty(n_iter, device=device)
            cost_l2o_5 = torch.empty(n_iter, device=device)
            cost_l2o_6 = torch.empty(n_iter, device=device)
            cost_l2o_7 = torch.empty(n_iter, device=device)
        
            for i in range(n_iter):
                # x0 = 2 * (torch.rand(A.shape[0], 1, device=device) - 0.5)
                x0 = torch.randn((A.shape[0], 1), device=device) * stds

                ocp.sys.reset(x0)  # Reset the system state before running the MPC loop

                # X_gd_1 = run_mpc_loop_gd(ocp, eta_gd, max_iterations=3, control_horizon=20)
                # ocp.sys.reset(x0)  # Reset the system state for the next run

                # X_gd_2 = run_mpc_loop_gd(ocp, eta_gd, max_iterations=30, control_horizon=20)
                # ocp.sys.reset(x0)  # Reset the system state for the next run

                # X_gd_3 = run_mpc_loop_gd(ocp, eta_gd, max_iterations=300, control_horizon=20)
                # ocp.sys.reset(x0)  # Reset the system state for the next run

                X_pgd_a, U_pgd_a = run_mpc_loop_pgd(ocp, eta_pgd, max_iterations=0, control_horizon=control_horizon)
                ocp.sys.reset(x0)  # Reset the system state for the next run

                X_pgd_b, U_pgd_b = run_mpc_loop_pgd(ocp, eta_pgd, max_iterations=1, control_horizon=control_horizon)
                ocp.sys.reset(x0)  # Reset the system state for the next run

                X_pgd_c, U_pgd_c = run_mpc_loop_pgd(ocp, eta_pgd, max_iterations=2, control_horizon=control_horizon)
                ocp.sys.reset(x0)  # Reset the system state for the next run

                X_pgd_0, U_pgd_0 = run_mpc_loop_pgd(ocp, eta_pgd, max_iterations=3, control_horizon=control_horizon)
                ocp.sys.reset(x0)  # Reset the system state for the next run

                X_pgd_1, U_pgd_1 = run_mpc_loop_pgd(ocp, eta_pgd, max_iterations=5, control_horizon=control_horizon)
                ocp.sys.reset(x0)  # Reset the system state for the next run

                X_pgd_2, U_pgd_2 = run_mpc_loop_pgd(ocp, eta_pgd, max_iterations=10, control_horizon=control_horizon)
                ocp.sys.reset(x0)  # Reset the system state for the next run

                X_pgd_3, U_pgd_3 = run_mpc_loop_pgd(ocp, eta_pgd, max_iterations=20, control_horizon=control_horizon)
                ocp.sys.reset(x0)  # Reset the system state for the next run

                X_pgd_4, U_pgd_4 = run_mpc_loop_pgd(ocp, eta_pgd, max_iterations=30, control_horizon=control_horizon)
                ocp.sys.reset(x0)  # Reset the system state for the next run

                X_pgd_5, U_pgd_5 = run_mpc_loop_pgd(ocp, eta_pgd, max_iterations=50, control_horizon=control_horizon)
                ocp.sys.reset(x0)  # Reset the system state for the next run

                X_pgd_6, U_pgd_6 = run_mpc_loop_pgd(ocp, eta_pgd, max_iterations=75, control_horizon=control_horizon)
                ocp.sys.reset(x0)  # Reset the system state for the next run

                X_pgd_7, U_pgd_7 = run_mpc_loop_pgd(ocp, eta_pgd, max_iterations=100, control_horizon=control_horizon)
                ocp.sys.reset(x0)  # Reset the system state for the next run


                X_l2o_a, U_l2o_a = run_mpc_loop_l2o(ocp, learned_update, eta_pgd, max_iterations=0, control_horizon=control_horizon)
                ocp.sys.reset(x0)  # Reset the system state for the next run

                X_l2o_b, U_l2o_b = run_mpc_loop_l2o(ocp, learned_update, eta_pgd, max_iterations=1, control_horizon=control_horizon)
                ocp.sys.reset(x0)  # Reset the system state for the next run

                X_l2o_c, U_l2o_c = run_mpc_loop_l2o(ocp, learned_update, eta_pgd, max_iterations=2, control_horizon=control_horizon)
                ocp.sys.reset(x0)  # Reset the system state for the next run

                X_l2o_0, U_l2o_0 = run_mpc_loop_l2o(ocp, learned_update, eta_pgd, max_iterations=3, control_horizon=control_horizon)
                ocp.sys.reset(x0)  # Reset the system state for the next run

                X_l2o_1, U_l2o_1 = run_mpc_loop_l2o(ocp, learned_update, eta_pgd, max_iterations=5, control_horizon=control_horizon)
                ocp.sys.reset(x0)  # Reset the system state for the next run

                X_l2o_2, U_l2o_2 = run_mpc_loop_l2o(ocp, learned_update, eta_pgd, max_iterations=10, control_horizon=control_horizon)
                ocp.sys.reset(x0)  # Reset the system state for the next run

                X_l2o_3, U_l2o_3 = run_mpc_loop_l2o(ocp, learned_update, eta_pgd, max_iterations=20, control_horizon=control_horizon)
                ocp.sys.reset(x0)  # Reset the system state for the next run

                X_l2o_4, U_l2o_4 = run_mpc_loop_l2o(ocp, learned_update, eta_pgd, max_iterations=30, control_horizon=control_horizon)
                ocp.sys.reset(x0)  # Reset the system state for the next run

                X_l2o_5, U_l2o_5 = run_mpc_loop_l2o(ocp, learned_update, eta_pgd, max_iterations=50, control_horizon=control_horizon)
                ocp.sys.reset(x0)  # Reset the system state for the next run

                X_l2o_6, U_l2o_6 = run_mpc_loop_l2o(ocp, learned_update, eta_pgd, max_iterations=75, control_horizon=control_horizon)
                ocp.sys.reset(x0)  # Reset the system state for the next run

                X_l2o_7, U_l2o_7 = run_mpc_loop_l2o(ocp, learned_update, eta_pgd, max_iterations=100, control_horizon=control_horizon)
                ocp.sys.reset(x0)  # Reset the system state for the next run

                cost_pgd_a[i] = ocp_test.cost(U_pgd_a.T).item()
                cost_pgd_b[i] = ocp_test.cost(U_pgd_b.T).item()
                cost_pgd_c[i] = ocp_test.cost(U_pgd_c.T).item()
                cost_pgd_0[i] = ocp_test.cost(U_pgd_0.T).item()
                cost_pgd_1[i] = ocp_test.cost(U_pgd_1.T).item()
                cost_pgd_2[i] = ocp_test.cost(U_pgd_2.T).item()
                cost_pgd_3[i] = ocp_test.cost(U_pgd_3.T).item()
                cost_pgd_4[i] = ocp_test.cost(U_pgd_4.T).item()
                cost_pgd_5[i] = ocp_test.cost(U_pgd_5.T).item()
                cost_pgd_6[i] = ocp_test.cost(U_pgd_6.T).item()
                cost_pgd_7[i] = ocp_test.cost(U_pgd_7.T).item()

                cost_l2o_a[i] = ocp_test.cost(U_l2o_a.T).item()
                cost_l2o_b[i] = ocp_test.cost(U_l2o_b.T).item()
                cost_l2o_c[i] = ocp_test.cost(U_l2o_c.T).item()
                cost_l2o_0[i] = ocp_test.cost(U_l2o_0.T).item()
                cost_l2o_1[i] = ocp_test.cost(U_l2o_1.T).item()
                cost_l2o_2[i] = ocp_test.cost(U_l2o_2.T).item()
                cost_l2o_3[i] = ocp_test.cost(U_l2o_3.T).item()
                cost_l2o_4[i] = ocp_test.cost(U_l2o_4.T).item()
                cost_l2o_5[i] = ocp_test.cost(U_l2o_5.T).item()
                cost_l2o_6[i] = ocp_test.cost(U_l2o_6.T).item()
                cost_l2o_7[i] = ocp_test.cost(U_l2o_7.T).item()

        # print("Mean cost PGD (3 iters): {:.2f} | Variance of cost PGD (3 iters): {:.2f}".format(cost_pgd_0.mean().item(), cost_pgd_0.var().item()))
        # print("Mean cost L2O (3 iters): {:.2f} | Variance of cost L2O (3 iters): {:.2f}".format(cost_l2o_0.mean().item(), cost_l2o_0.var().item()))

        # print("Mean cost PGD (5 iters): {:.2f} | Variance of cost PGD (5 iters): {:.2f}".format(cost_pgd_1.mean().item(), cost_pgd_1.var().item()))
        # print("Mean cost L2O (5 iters): {:.2f} | Variance of cost L2O (5 iters): {:.2f}".format(cost_l2o_1.mean().item(), cost_l2o_1.var().item()))

        # print("Mean cost PGD (10 iters): {:.2f} | Variance of cost PGD (10 iters): {:.2f}".format(cost_pgd_2.mean().item(), cost_pgd_2.var().item()))
        # print("Mean cost L2O (10 iters): {:.2f} | Variance of cost L2O (10 iters): {:.2f}".format(cost_l2o_2.mean().item(), cost_l2o_2.var().item()))
        
        # print("Mean cost PGD (20 iters): {:.2f} | Variance of cost PGD (20 iters): {:.2f}".format(cost_pgd_3.mean().item(), cost_pgd_3.var().item()))
        # print("Mean cost L2O (20 iters): {:.2f} | Variance of cost L2O (20 iters): {:.2f}".format(cost_l2o_3.mean().item(), cost_l2o_3.var().item()))
        
        # print("Mean cost PGD (30 iters): {:.2f} | Variance of cost PGD (30 iters): {:.2f}".format(cost_pgd_4.mean().item(), cost_pgd_4.var().item()))
        # print("Mean cost L2O (30 iters): {:.2f} | Variance of cost L2O (30 iters): {:.2f}".format(cost_l2o_4.mean().item(), cost_l2o_4.var().item()))

        # print("Mean cost PGD (50 iters): {:.2f} | Variance of cost PGD (50 iters): {:.2f}".format(cost_pgd_5.mean().item(), cost_pgd_5.var().item()))   
        # print("Mean cost L2O (50 iters): {:.2f} | Variance of cost L2O (50 iters): {:.2f}".format(cost_l2o_5.mean().item(), cost_l2o_5.var().item()))

        # print("Mean cost PGD (70 iters): {:.2f} | Variance of cost PGD (70 iters): {:.2f}".format(cost_pgd_6.mean().item(), cost_pgd_6.var().item()))   
        # print("Mean cost L2O (70 iters): {:.2f} | Variance of cost L2O (70 iters): {:.2f}".format(cost_l2o_6.mean().item(), cost_l2o_6.var().item()))

        # print("Mean cost PGD (100 iters): {:.2f} | Variance of cost PGD (100 iters): {:.2f}".format(cost_pgd_7.mean().item(), cost_pgd_7.var().item()))   
        # print("Mean cost L2O (100 iters): {:.2f} | Variance of cost L2O (100 iters): {:.2f}".format(cost_l2o_7.mean().item(), cost_l2o_7.var().item()))

        # Save the data to a .npz file if it doesn't exist
        np.savez(data_file,
                cost_pgd_a=cost_pgd_a.cpu().numpy(),
                cost_l2o_a=cost_l2o_a.cpu().numpy(),
                cost_pgd_b=cost_pgd_b.cpu().numpy(),
                cost_l2o_b=cost_l2o_b.cpu().numpy(),
                cost_pgd_c=cost_pgd_c.cpu().numpy(),
                cost_l2o_c=cost_l2o_c.cpu().numpy(),
                cost_pgd_0=cost_pgd_0.cpu().numpy(),
                cost_l2o_0=cost_l2o_0.cpu().numpy(),
                cost_pgd_1=cost_pgd_1.cpu().numpy(),
                cost_l2o_1=cost_l2o_1.cpu().numpy(),
                cost_pgd_2=cost_pgd_2.cpu().numpy(),
                cost_l2o_2=cost_l2o_2.cpu().numpy(),
                cost_pgd_3=cost_pgd_3.cpu().numpy(),
                cost_l2o_3=cost_l2o_3.cpu().numpy(),
                cost_pgd_4=cost_pgd_4.cpu().numpy(),
                cost_l2o_4=cost_l2o_4.cpu().numpy(),
                cost_pgd_5=cost_pgd_5.cpu().numpy(),
                cost_l2o_5=cost_l2o_5.cpu().numpy(),
                cost_pgd_6=cost_pgd_6.cpu().numpy(),
                cost_l2o_6=cost_l2o_6.cpu().numpy(),
                cost_pgd_7=cost_pgd_7.cpu().numpy(),
                cost_l2o_7=cost_l2o_7.cpu().numpy())
        print(f"Data saved to {data_file}")


    # Visualize the costs using a box plot
    import matplotlib.pyplot as plt

    import matplotlib.patches as mpatches

    # data = [
    #     cost_pgd_1.cpu().numpy(),
    #     cost_l2o_1.cpu().numpy(),
    #     cost_pgd_2.cpu().numpy(),
    #     cost_l2o_2.cpu().numpy(),
    #     cost_pgd_3.cpu().numpy(),
    #     cost_l2o_3.cpu().numpy(),
    # ]

    # Load the data from the .npz file
    data = np.load(data_file)
    cost_pgd_a = data['cost_pgd_a']
    cost_l2o_a = data['cost_l2o_a']
    cost_pgd_b = data['cost_pgd_b']
    cost_l2o_b = data['cost_l2o_b']
    cost_pgd_c = data['cost_pgd_c']
    cost_l2o_c = data['cost_l2o_c']
    cost_pgd_0 = data['cost_pgd_0']
    cost_l2o_0 = data['cost_l2o_0']
    cost_pgd_1 = data['cost_pgd_1']
    cost_l2o_1 = data['cost_l2o_1']
    cost_pgd_2 = data['cost_pgd_2']
    cost_l2o_2 = data['cost_l2o_2']
    cost_pgd_3 = data['cost_pgd_3']
    cost_l2o_3 = data['cost_l2o_3']
    cost_pgd_4 = data['cost_pgd_4']
    cost_l2o_4 = data['cost_l2o_4']
    cost_pgd_5 = data['cost_pgd_5']
    cost_l2o_5 = data['cost_l2o_5']
    cost_pgd_6 = data['cost_pgd_6']
    cost_l2o_6 = data['cost_l2o_6']
    cost_pgd_7 = data['cost_pgd_7']
    cost_l2o_7 = data['cost_l2o_7']
    print(f"Data loaded from {data_file}")

    # Define positions for pairs:
    # First three pairs: a, b, c; then pairs for 0, 5, 10, 20, 30, 50, and 100 iterations.
    positions = [0, 0.4, 1, 1.4, 2, 2.4, 3, 3.4, 4, 4.4, 5, 5.4, 6, 6.4, 7, 7.4, 8, 8.4, 9, 9.4, 10, 10.4]

    # Arrange data in the order:
    # [PGD (a), L2O (a), PGD (b), L2O (b), PGD (c), L2O (c),
    #  PGD (0 iters), L2O (0 iters), PGD (5 iters), L2O (5 iters), PGD (10 iters), L2O (10 iters),
    #  PGD (20 iters), L2O (20 iters), PGD (30 iters), L2O (30 iters), PGD (50 iters), L2O (50 iters),
    #  PGD (100 iters), L2O (100 iters)]
    data = [cost_pgd_a, cost_l2o_a, cost_pgd_b, cost_l2o_b, cost_pgd_c, cost_l2o_c,
            cost_pgd_0, cost_l2o_0, cost_pgd_1, cost_l2o_1, cost_pgd_2, cost_l2o_2,
            cost_pgd_3, cost_l2o_3, cost_pgd_4, cost_l2o_4, cost_pgd_5, cost_l2o_5,
            cost_pgd_6, cost_l2o_6, cost_pgd_7, cost_l2o_7]

    # # First box plot with custom positions and colors, removing outlier markers
    # plt.figure(figsize=(8, 6))
    # bp = plt.boxplot(data, positions=positions, widths=0.3, patch_artist=True, showfliers=False)

    # # Set colors: even indexed boxes for PGD (red), odd indexed for L2O (green)
    # for i, box in enumerate(bp['boxes']):
    #     box.set_facecolor("red" if i % 2 == 0 else "green")

    # y_max = plt.ylim()[1]
    # for pos, d in zip(positions, data):
    #     mean_val = np.mean(d)
    #     std_val = np.std(d)
    #     plt.text(pos, y_max * 0.95, f'{mean_val:.2f}\n±{std_val:.2f}', ha='center', va='top', fontsize=10, color="blue")

    # # Center x-axis ticks at positions corresponding to iteration counts
    # plt.xticks([1.2, 2.2, 3.2, 4.2, 5.2], ['5', '10', '20', '30', '50'])
    # plt.ylabel("Cost")
    # plt.yscale('log')  # Log scale for better visibility of differences
    # plt.title("Cost Comparison Box Plot")

    # # Add legend for colors
    # red_patch = mpatches.Patch(color='red', label='PGD')
    # green_patch = mpatches.Patch(color='green', label='L2O')
    # plt.legend(handles=[red_patch, green_patch])
    # plt.tight_layout()

    # New plot: average cost vs. iterations (0, 5, 10, 20, 30, 50, 100) with mean and mean+std (dotted) lines
    plt.figure(figsize=(8, 4))
    # x_vals = np.array([0, 1, 2, 3, 5, 10, 20, 30, 50, 75, 100])
    x_vals = np.array([1, 5, 10, 20, 30, 50, 75, 100])
    
    # PGD: compute mean and std for each iteration count, including 0 iterations
    pgd_means = np.array([
        # np.mean(cost_pgd_a),
        np.mean(cost_pgd_b),
        # np.mean(cost_pgd_c),
        # np.mean(cost_pgd_0),
        np.mean(cost_pgd_1),
        np.mean(cost_pgd_2),
        np.mean(cost_pgd_3),
        np.mean(cost_pgd_4),
        np.mean(cost_pgd_5),
        np.mean(cost_pgd_6),
        np.mean(cost_pgd_7)
    ])
    pgd_stds = np.array([
        # np.std(cost_pgd_a),
        np.std(cost_pgd_b),
        # np.std(cost_pgd_c),
        # np.std(cost_pgd_0),
        np.std(cost_pgd_1),
        np.std(cost_pgd_2),
        np.std(cost_pgd_3),
        np.std(cost_pgd_4),
        np.std(cost_pgd_5),
        np.std(cost_pgd_6),
        np.std(cost_pgd_7)
    ])
    
    # L2O: compute mean and std for each iteration count, including 0 iterations
    l2o_means = np.array([
        # np.mean(cost_l2o_a),
        np.mean(cost_l2o_b),
        # np.mean(cost_l2o_c),
        # np.mean(cost_l2o_0),
        np.mean(cost_l2o_1),
        np.mean(cost_l2o_2),
        np.mean(cost_l2o_3),
        np.mean(cost_l2o_4),
        np.mean(cost_l2o_5),
        np.mean(cost_l2o_6),
        np.mean(cost_l2o_7)
    ])
    l2o_stds = np.array([
        # np.std(cost_l2o_a),
        np.std(cost_l2o_b),
        # np.std(cost_l2o_c),
        # np.std(cost_l2o_0),
        np.std(cost_l2o_1),
        np.std(cost_l2o_2),
        np.std(cost_l2o_3),
        np.std(cost_l2o_4),
        np.std(cost_l2o_5),
        np.std(cost_l2o_6),
        np.std(cost_l2o_7)
    ])
    
    # Plot the mean cost
    plt.plot(x_vals, pgd_means, 'o-', color='red', label='Projected gradient descent')
    plt.plot(x_vals, l2o_means, 'o-', color='green', label='Linearly convergent L2O')
    
    # Compute lower and upper quantiles (e.g., 10% and 90% quantiles)

    upper_quantile = 0.8
    pgd_lower = np.array([
        np.quantile(cost_pgd_b, 0.1),
        np.quantile(cost_pgd_1, 0.1),
        np.quantile(cost_pgd_2, 0.1),
        np.quantile(cost_pgd_3, 0.1),
        np.quantile(cost_pgd_4, 0.1),
        np.quantile(cost_pgd_5, 0.1),
        np.quantile(cost_pgd_6, 0.1),
        np.quantile(cost_pgd_7, 0.1)
    ])
    pgd_upper = np.array([
        np.quantile(cost_pgd_b, upper_quantile),
        np.quantile(cost_pgd_1, upper_quantile),
        np.quantile(cost_pgd_2, upper_quantile),
        np.quantile(cost_pgd_3, upper_quantile),
        np.quantile(cost_pgd_4, upper_quantile),
        np.quantile(cost_pgd_5, upper_quantile),
        np.quantile(cost_pgd_6, upper_quantile),
        np.quantile(cost_pgd_7, upper_quantile)
    ])

    l2o_lower = np.array([
        np.quantile(cost_l2o_b, 0.1),
        np.quantile(cost_l2o_1, 0.1),
        np.quantile(cost_l2o_2, 0.1),
        np.quantile(cost_l2o_3, 0.1),
        np.quantile(cost_l2o_4, 0.1),
        np.quantile(cost_l2o_5, 0.1),
        np.quantile(cost_l2o_6, 0.1),
        np.quantile(cost_l2o_7, 0.1)
    ])
    l2o_upper = np.array([
        np.quantile(cost_l2o_b, upper_quantile),
        np.quantile(cost_l2o_1, upper_quantile),
        np.quantile(cost_l2o_2, upper_quantile),
        np.quantile(cost_l2o_3, upper_quantile),
        np.quantile(cost_l2o_4, upper_quantile),
        np.quantile(cost_l2o_5, upper_quantile),
        np.quantile(cost_l2o_6, upper_quantile),
        np.quantile(cost_l2o_7, upper_quantile)
    ])

    # Plot filled areas for the quantile ranges
    plt.fill_between(x_vals, pgd_lower, pgd_upper, color='red', alpha=0.1)
    plt.fill_between(x_vals, l2o_lower, l2o_upper, color='green', alpha=0.1)

    
    plt.xlabel("Number of optimization steps")
    plt.ylabel("Closed-loop control cost")
    # plt.title("Average Cost vs. Iterations")
    plt.yscale('log')  # Log scale for better visibility of differences
    plt.ylim(bottom=1e1)
    plt.xlim(left=1, right=100)  # Adjust x-axis limits to fit the data range
    # Bring all line markers to the foreground so their full shape is visible ("in primo piano")
    for line in plt.gca().get_lines():
        line.set_clip_on(False)
    # plt.xticks(x_vals, ['1', '5', '10', '20', '30', '50', '75', '100'])
    plt.legend()
    plt.tight_layout()
    plt.minorticks_on()
    plt.grid(True, which='major', linestyle='-', linewidth=0.5, color='gray')
    plt.grid(True, which='minor', linestyle=':', linewidth=0.25, color='gray')
    # plt.grid(which='major', color='gray', linestyle='-', linewidth=0.75)
    # plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)
    plt.savefig('my_plot.png', bbox_inches='tight')
    
    plt.show()

    # Second box plot using categorical labels
    # plt.figure(figsize=(8, 6))
    # labels = ['PGD', 'L2O', 'PGD', 'L2O', 'PGD', 'L2O']
    # plt.boxplot(data, labels=labels, patch_artist=True)
    # plt.title("Cost Comparison Box Plot")
    # plt.ylabel("Cost")
    # plt.yscale('log')  # Log scale for better visibility of differences
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.show()




            # print(ocp_test.cost(U_pgd_1.T).item(), ocp_test.cost(U_pgd_2.T).item(), ocp_test.cost(U_pgd_3.T).item())
            # print(ocp_test.cost(U_l2o_1.T).item(), ocp_test.cost(U_l2o_2.T).item(), ocp_test.cost(U_l2o_3.T).item())


        # plt.figure()
        # # # plt.plot(range(21), X_gd_1[0,:].cpu().numpy())
        # # # plt.plot(range(21), X_gd_2[0,:].cpu().numpy())
        # # # plt.plot(range(21), X_gd_3[0,:].cpu().numpy())

        # # # plt.plot(range(21), X_gd_1[1,:].cpu().numpy())
        # # # plt.plot(range(21), X_gd_2[1,:].cpu().numpy())
        # # # plt.plot(range(21), X_gd_3[1,:].cpu().numpy())

        # # plt.plot(range(21), X_pgd_1[0,:].cpu().numpy())
        # plt.plot(range(21), X_pgd_2[0,:].cpu().numpy())
        # plt.plot(range(21), X_pgd_3[0,:].cpu().numpy())

        # # # plt.plot(range(21), X_pgd_1[1,:].cpu().numpy())
        
        # # plt.plot(range(21), X_l2o_1[0,:].cpu().numpy())
        # plt.plot(range(21), X_l2o_2[0,:].cpu().numpy())
        # plt.plot(range(21), X_l2o_3[0,:].cpu().numpy())

        # plt.legend(['x1 (PGD 30 iters)', 'x1 (PGD 300 iters)', 'x1 (L2O 30 iters)', 'x1 (L2O 300 iters)'])

        # plt.figure()
        # plt.plot(range(21), X_pgd_2[1,:].cpu().numpy())
        # plt.plot(range(21), X_pgd_3[1,:].cpu().numpy())
        
        # # plt.plot(range(21), X_l2o_1[1,:].cpu().numpy())
        # plt.plot(range(21), X_l2o_2[1,:].cpu().numpy())
        # plt.plot(range(21), X_l2o_3[1,:].cpu().numpy())

        # plt.legend(['x2 (PGD 30 iters)', 'x2 (PGD 300 iters)', 'x2 (L2O 30 iters)', 'x2 (L2O 300 iters)'])
        # plt.show()

        # plt.legend(['x1 (GD 3 iters)', 'x1 (GD 30 iters)', 'x1 (GD 300 iters)', 'x2 (GD 3 iters)', 'x2 (GD 30 iters)', 'x2 (GD 300 iters)', 'x1 (PGD 3 iters)', 'x1 (PGD 30 iters)', 'x1 (PGD 300 iters)', 'x2 (PGD 3 iters)', 'x2 (PGD 30 iters)', 'x2 (PGD 300 iters)'])
        








    

    



    ocp = FiniteHorizonOCP(sys, Qt, Qf, Rt, T)
    # Test the learned optimizer on a batch of problems
    test_samples, test_batch_size = 256, 16
    test_dataset = ModelPredictiveControlDataset(ocp, num_samples=test_samples)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

    U0b = torch.zeros((test_batch_size, sys.m * T, 1), device=device)

    flag = False
    for Qb, cb, qb, Ab, bb in test_dataloader:

        gb = lambda U: Qb @ U + cb
        lb = lambda U: 0.5 * U.mT @ Qb @ U + cb.mT @ U + qb
        project = lambda U: torch.clamp(U, min=-.25, max=.25) # Project onto the box constraints

        # print(f"\nOptimal value of unconstrained optimization problem: {lb(-torch.linalg.inv(Qb) @ cb).item():.2f}")

        # Ub_gd, Jb_gd = gradient_descent(U0b.clone(), gb, lb, iterations=max_iterations, step_size=eta_gd)
        # Ub_nag, Jb_nag = nesterov_accelerated_gradient_descent(U0b.clone(), gb, lb, iterations=max_iterations, step_size=eta_nag, mu=mu_nag)
        Ub_pgd, Jb_pgd = projected_gradient_descent(U0b.clone(), gb, lb, project, iterations=max_iterations, step_size=eta_pgd)

        Ub_l2o, Jb_l2o = l2o_descent(U0b.clone(), learned_update, gb, lb, project, Ab, bb, iterations=max_iterations, step_size=eta_pgd)

        if not flag:
            # J_gd = Jb_gd
            J_pgd = Jb_pgd
            # J_nag = Jb_nag
            J_l2o = Jb_l2o
            flag = True
        else:
            # J_gd = torch.vstack([J_gd, Jb_gd])
            J_pgd = torch.vstack([J_pgd, Jb_pgd])
            # J_nag = torch.vstack([J_nag, Jb_nag])
            J_l2o = torch.vstack([J_l2o, Jb_l2o])

    # J_gd_mean = torch.mean(J_gd, dim=0).squeeze().cpu()
    # J_gd_std = torch.std(J_gd, dim=0).squeeze().cpu()
    
    upper_quantile = 0.85

    J_pgd_mean = torch.mean(J_pgd, dim=0).squeeze().cpu()
    J_pgd_std = torch.std(J_pgd, dim=0).squeeze().cpu()
    J_pgd_quantile_low = torch.quantile(J_pgd, 0.1, dim=0).detach().squeeze().cpu()
    J_pgd_quantile_high = torch.quantile(J_pgd, upper_quantile, dim=0).detach().squeeze().cpu()

    # J_nag_mean = torch.mean(J_nag, dim=0).squeeze().cpu()
    # J_nag_std = torch.std(J_nag, dim=0).squeeze().cpu()

    J_l2o_mean = torch.mean(J_l2o, dim=0).squeeze().cpu()
    J_l2o_std = torch.std(J_l2o, dim=0).squeeze().cpu()
    J_l2o_quantile_low = torch.quantile(J_l2o, 0.1, dim=0).detach().squeeze().cpu()
    J_l2o_quantile_high = torch.quantile(J_l2o, upper_quantile, dim=0).detach().squeeze().cpu()

    plt.figure(figsize=(8, 4))
    plt.yscale('log')

    # plt.plot(range(max_iterations+1), J_gd_mean, label="Gradient Descent")
    # plt.fill_between(range(max_iterations+1),
    #                 (J_gd_mean - J_gd_std).numpy(),
    #                 (J_gd_mean + J_gd_std).numpy(),
    #                 alpha=0.3)
    
    
    plt.plot(range(1, max_iterations+1), J_pgd_mean[1:], color='red', label="Projected Gradient Descent")
    # plt.fill_between(range(max_iterations),
    #                 (J_pgd_mean[1:] - J_pgd_std[1:]).numpy(),
    #                 (J_pgd_mean[1:] + J_pgd_std[1:]).numpy(),
    #                 alpha=0.3)
    plt.plot(range(1, max_iterations+1), J_l2o_mean[1:].detach(), color='green', label="Linearly convergent L2O")
    # plt.fill_between(range(max_iterations),
    #                 (J_l2o_mean[1:] - J_l2o_std[1:]).detach().numpy(),
    #                 (J_l2o_mean[1:] + J_l2o_std[1:]).detach().numpy(),
    #                 alpha=0.3)

    # Plot filled areas for the quantile ranges
    plt.fill_between(range(1, max_iterations+1), J_pgd_quantile_low[1:], J_pgd_quantile_high[1:], color='red', alpha=0.1)
    plt.fill_between(range(1, max_iterations+1), J_l2o_quantile_low[1:], J_l2o_quantile_high[1:], color='green', alpha=0.1)

    # plt.plot(range(max_iterations+1), J_nag_mean, label="Nesterov Accelerated Gradient Descent")
    # plt.fill_between(range(max_iterations+1),
    #                 (J_nag_mean - J_nag_std).numpy(),
    #                 (J_nag_mean + J_nag_std).numpy(),
    #                 alpha=0.3)
    
    plt.xlabel("Number of optimization steps")
    plt.ylabel("Value of the quadratic objective function")
    # plt.title("Average Cost vs. Iterations")
    plt.yscale('log')  # Log scale for better visibility of differences
    plt.ylim(bottom=1e1)
    plt.xlim(left=1, right=100)  # Adjust x-axis limits to fit the data range
    plt.yticks([10], [r'$10^1$'])
    # Bring all line markers to the foreground so their full shape is visible ("in primo piano")
    for line in plt.gca().get_lines():
        line.set_clip_on(False)
    # plt.xticks(x_vals, ['1', '5', '10', '20', '30', '50', '75', '100'])
    plt.legend()
    plt.tight_layout()
    plt.minorticks_on()
    plt.grid(True, which='major', linestyle='-', linewidth=0.5, color='gray')
    plt.grid(True, which='minor', linestyle=':', linewidth=0.25, color='gray')
    # plt.grid(which='major', color='gray', linestyle='-', linewidth=0.75)
    # plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)
    # plt.savefig('my_plot.png', bbox_inches='tight')

    # plt.plot(range(max_iterations+1), J_pgd_mean, label="Projected Gradient Descent")
    # plt.fill_between(range(max_iterations+1),
    #                 (J_pgd_mean - J_pgd_std).numpy(),
    #                 (J_pgd_mean + J_pgd_std).numpy(),
    #                 alpha=0.3)
    # plt.plot(range(max_iterations+1), J_l2o_mean.detach(), label="Learned Optimizer")
    # plt.fill_between(range(max_iterations+1),
    #                 (J_l2o_mean - J_l2o_std).detach().numpy(),
    #                 (J_l2o_mean + J_l2o_std).detach().numpy(),
    #                 alpha=0.3)

    plt.legend()
    plt.grid(True)

    plt.savefig('my_plot.png', bbox_inches='tight')
    # plt.figure()
    # plt.plot(range(T), Ub_gd[-1][0], label="Gradient Descent")
    # plt.plot(range(T), Ub_pgd[-1][0], label="Projected Gradient Descent")
    # plt.plot(range(T), Ub_nag[-1][0], label="Nesterov Accelerated Gradient Descent")
    # plt.legend()
    # plt.grid(True)
    # plt.title("Control Input Trajectories")
    plt.show()


    print(f"That's all folks!")
 
if __name__ == "__main__":
    main()