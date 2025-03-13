import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def evaluate(model, A, m, d, device):
    
    
    # d = int(model.get_dimension()) # Number of columns in matrix A (dimension of parameter vector x)
    
    T_eval = 50  # Number of evaluation iterations

    eta = 0.001

    # min_dim = min(m, d)    
    # # Sample a new quadratic task
    # eigenvalues_ATA = torch.randint(1, 1001, (m - 2,), device=device, dtype=torch.float)
    # eigenvalues_ATA = torch.cat([eigenvalues_ATA, torch.tensor([1.0, 1000.0], device=device)])
    # # eigenvalues_ATA = eigenvalues_ATA[torch.randperm(len(eigenvalues_ATA))]
    
    A = torch.ones(m, d, device=device)
    #A[:min_dim, :min_dim] = torch.diag(torch.sqrt(eigenvalues_ATA[:min_dim]))

    # Generate b vector randomly
    b = torch.rand(m, device=device)

    # Use the same initial point for both optimizers
    x0 = torch.zeros(d, 1, device=device)

    # --------- Learned Optimizer Evaluation ---------
    model.eval()  # set to evaluation mode
    x_learned = x0.clone()
    loss_history_learned = []
    # --- Add this before the learned optimizer evaluation loop ---
    v_history = []

    for t in range(T_eval):
        Ax = A@x_learned
        loss = torch.sum((Ax - b.unsqueeze(1)) ** 2)
        loss_history_learned.append(loss.item())

        grad = 2 * A.T @ (Ax - b.unsqueeze(1))  # shape: (d, 1)
        #grad = grad.T  # reshape to (1, d)
        loss_val = loss.view(1, 1)

        with torch.no_grad():
            v_t = model(x_learned.T, loss_val, grad.T, t)
            v_history.append(v_t.norm().item())

        x_learned = x_learned - eta * grad + v_t.T

    # --------- Nesterov Accelerated Gradient (NAG) Evaluation ---------
    # We'll implement a basic version of NAG.
    lr_NAG = eta  # learning rate for Nesterov
    beta = 0   # momentum parameter
    x_nag = x0.clone()
    x_prev = x_nag.clone()
    loss_history_nag = []

    for t in range(T_eval):
        if t == 0:
            y = x_nag
        else:
            y = x_nag + beta * (x_nag - x_prev)

        Ax_y = A@y
        loss_y = torch.sum((Ax_y - b.unsqueeze(1)) ** 2)
        loss_history_nag.append(loss_y.item())

        grad_y = 2 * torch.matmul(A.T, (Ax_y - b.unsqueeze(1)))  # shape: (d, 1)
        # grad_y = grad_y.T  # shape: (1, d)

        x_new = y - lr_NAG * grad_y
        x_prev = x_nag.clone()
        x_nag = x_new


    # Plot the loss curves
    plt.figure(figsize=(6, 4))
    plt.plot(range(T_eval), loss_history_learned, label="Learned Optimizer", marker='o')
    plt.plot(range(T_eval), loss_history_nag, label="Nesterov", marker='s')
    plt.xlabel("Iteration")
    plt.ylabel("Quadratic Loss")
    plt.title("Optimization Comparison on a Random Quadratic Task")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- After the evaluation loops, add this code to plot the v_t values ---
    plt.figure(figsize=(6, 4))
    plt.plot(range(T_eval), v_history, label="||v_t||", marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("||v_t||")
    plt.title("Norm of Learned Update v_t over Iterations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
