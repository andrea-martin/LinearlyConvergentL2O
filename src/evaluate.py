import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def evaluate(model, A, dataloader, T, device):
    
    m = A.shape[0] # Number of rows in matrix A (dimension of output vector b)
    d = A.shape[1] # Number of columns in matrix A (dimension of parameter vector x)
    U, S, V = torch.linalg.svd(A)
    lambda_max = S.max().item() ** 2
    eta = 1/(2*lambda_max)/2
    
    model.eval() # set to evaluation mode

    loss_total = torch.zeros(T, device=device)

    # v_history = []
    
    for b_batch in dataloader:
        batch_size = b_batch.shape[0]

        # Initialize parameter vector x (shape: (d, 1))
        x_learned_batch = torch.zeros(batch_size, d, 1, device=device)

        for t in range(T):
            residual_batch = A@x_learned_batch - b_batch.unsqueeze(-1) # A@x is (batch_size, m) and b is (batch_size, m)
            residual_squared_batch = residual_batch ** 2
            loss_batch = residual_squared_batch.sum(dim=1).unsqueeze(-1)
            grad_batch = 2 * A.T @ residual_batch
            
            with torch.no_grad():
                v_t_batch = model(x_learned_batch, loss_batch, grad_batch, t)
                # v_history.append(v_t_batch.norm().item())

            x_learned_batch = x_learned_batch - eta * grad_batch + v_t_batch.unsqueeze(-1)

            loss_total[t] = loss_total[t] + loss_batch.sum()


    loss_gd_total = torch.zeros(T, device=device)
    for b_batch in dataloader:
            batch_size = b_batch.shape[0]

            # Initialize parameter vector x (shape: (d, 1))
            x_gd_batch = torch.zeros(batch_size, d, 1, device=device)

            for t in range(T):
                residual_batch = A@x_gd_batch - b_batch.unsqueeze(-1) # A@x is (batch_size, m) and b is (batch_size, m)
                residual_squared_batch = residual_batch ** 2
                loss_batch = residual_squared_batch.sum(dim=1).unsqueeze(-1)
                grad_batch = 2 * A.T @ residual_batch


                x_gd_batch = x_gd_batch - eta * grad_batch

                loss_gd_total[t] = loss_gd_total[t] + loss_batch.sum()

        # # --------- Nesterov Accelerated Gradient (NAG) Evaluation ---------
        # # We'll implement a basic version of NAG.
        # lr_NAG = eta  # learning rate for Nesterov
        # beta = 0   # momentum parameter
        # x_nag = x0.clone()
        # x_prev = x_nag.clone()
        # loss_history_nag = []

        # for t in range(T):
        #     if t == 0:
        #         y = x_nag
        #     else:
        #         y = x_nag + beta * (x_nag - x_prev)

        #     Ax_y = A@y
        #     loss_y = torch.sum((Ax_y - b.unsqueeze(1)) ** 2)
        #     loss_history_nag.append(loss_y.item())

        #     grad_y = 2 * torch.matmul(A.T, (Ax_y - b.unsqueeze(1)))  # shape: (d, 1)
        #     # grad_y = grad_y.T  # shape: (1, d)

        #     x_new = y - lr_NAG * grad_y
        #     x_prev = x_nag.clone()
        #     x_nag = x_new


    loss_total = loss_total.cpu().tolist()
    loss_gd_total = loss_gd_total.cpu().tolist()
    # Plot the loss curves
    plt.figure(figsize=(6, 4))
    plt.plot(range(T), loss_total, label="Learned Optimizer", marker='o')
    plt.plot(range(T), loss_gd_total, label="Gradient descent", marker='s')
    plt.xlabel("Iteration")
    plt.ylabel("Quadratic Loss")
    plt.title("Optimization Comparison on a Random Quadratic Task")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # # --- After the evaluation loops, add this code to plot the v_t values ---
    # plt.figure(figsize=(6, 4))
    # plt.plot(range(T), v_history, label="||v_t||", marker='o')
    # plt.xlabel("Iteration")
    # plt.ylabel("||v_t||")
    # plt.title("Norm of Learned Update v_t over Iterations")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
