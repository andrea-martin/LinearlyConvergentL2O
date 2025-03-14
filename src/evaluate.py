import torch
import numpy as np
import matplotlib.pyplot as plt

def evaluate(model, A, dataloader, T, device):
    
    d = A.shape[1] # Number of columns in matrix A (dimension of parameter vector x)
    _, S, _ = torch.linalg.svd(A)
    lambda_max = S.max().item() ** 2
    lambda_min = S.min().item() ** 2
    beta = 2*lambda_max
    alpha = 2*lambda_min
    kappa = beta/alpha # condition number

    eta = 1/beta / 2 # learning rate for learned optimizer
    mu = 0.7#(np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)

    model.eval() # set to evaluation mode

    # Track losses for each sample at each timestep
    loss_per_sample = []  
    loss_gd_per_sample = []
    loss_nag_per_sample = []

    # v_history = []
    
    for b_batch in dataloader:
        batch_size = b_batch.shape[0]

        # # --------- Learned Optimizer Evaluation ---------
        x_learned_batch = torch.zeros(batch_size, d, 1, device=device)
        # Track losses for this batch
        batch_loss_history = torch.zeros((T, batch_size), device=device)

        for t in range(T):
            residual_batch = A@x_learned_batch - b_batch.unsqueeze(-1) # A@x is (batch_size, m) and b is (batch_size, m)
            residual_squared_batch = residual_batch ** 2
            loss_batch = residual_squared_batch.sum(dim=1).unsqueeze(-1)
            grad_batch = 2 * A.T @ residual_batch
            
            with torch.no_grad():
                v_t_batch = model(x_learned_batch, loss_batch, grad_batch, t)
                # v_history.append(v_t_batch.norm().item())

            x_learned_batch = x_learned_batch - eta * grad_batch + v_t_batch.unsqueeze(-1)
            # Save loss per sample at this timestep
            batch_loss_history[t] = loss_batch.squeeze()  # Removes extra dimensions
        
        # Store the batch's loss history for analysis later
        loss_per_sample.append(batch_loss_history.cpu())






        # # --------- Gradient Descent (GD) Evaluation ---------
        x_gd_batch = torch.zeros(batch_size, d, 1, device=device)
        # Track losses for this batch
        batch_loss_gd_history = torch.zeros((T, batch_size), device=device)

        for t in range(T):
            residual_batch = A@x_gd_batch - b_batch.unsqueeze(-1) # A@x is (batch_size, m) and b is (batch_size, m)
            residual_squared_batch = residual_batch ** 2
            loss_batch = residual_squared_batch.sum(dim=1).unsqueeze(-1)
            grad_batch = 2 * A.T @ residual_batch

            x_gd_batch = x_gd_batch - eta * grad_batch
            # Save loss per sample at this timestep
            batch_loss_gd_history[t] = loss_batch.squeeze()  # Removes extra dimensions

        # Store the batch's loss history for analysis later
        loss_gd_per_sample.append(batch_loss_gd_history.cpu())






        # # --------- Nesterov Accelerated Gradient (NAG) Evaluation ---------
        x_prev_nag_batch = torch.zeros(batch_size, d, 1, device=device)
        x_nag_batch = x_prev_nag_batch.clone()
        y_nag_batch = x_nag_batch.clone()
        # Track losses for this batch
        batch_loss_nag_history = torch.zeros((T, batch_size), device=device)

        for t in range(T):
            
            
            # Maybe evaluate the loss on y_nag_batch instead of x_nag_batch?  Still not make a difference....
            residual_batch = A@x_nag_batch - b_batch.unsqueeze(-1) # A@x is (batch_size, m) and b is (batch_size, m)
            residual_squared_batch = residual_batch ** 2
            
            loss_batch = residual_squared_batch.sum(dim=1).unsqueeze(-1)
            
            grad_y_nag_batch = 2 * A.T @ (A@y_nag_batch - b_batch.unsqueeze(-1))

            x_next_nag_batch = y_nag_batch - eta * grad_y_nag_batch
            # Update the auxiliary point y
            y_nag_batch = (1+mu) * x_next_nag_batch - mu * x_prev_nag_batch           
            
            # Shift variables for the next iteration
            x_prev_nag_batch = x_nag_batch.clone()
            x_nag_batch = x_next_nag_batch.clone()

            # Save loss per sample at this timestep
            batch_loss_nag_history[t] = loss_batch.squeeze()  # Removes extra dimensions

        # Store the batch's loss history for analysis later
        loss_nag_per_sample.append(batch_loss_nag_history.cpu())


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

    # Combine all batches into one tensor (T, total_samples)
    loss_per_sample = torch.cat(loss_per_sample, dim=1)
    # Compute mean and std at each time step
    mean_loss = loss_per_sample.mean(dim=1)
    std_loss = loss_per_sample.std(dim=1)

    # Combine all batches into one tensor (T, total_samples)
    loss_gd_per_sample = torch.cat(loss_gd_per_sample, dim=1)
    # Compute mean and std at each time step
    mean_loss_gd = loss_gd_per_sample.mean(dim=1)
    std_loss_gd = loss_gd_per_sample.std(dim=1)

    # Combine all batches into one tensor (T, total_samples)
    loss_nag_per_sample = torch.cat(loss_nag_per_sample, dim=1)
    # Compute mean and std at each time step
    mean_loss_nag = loss_nag_per_sample.mean(dim=1)
    std_loss_nag = loss_nag_per_sample.std(dim=1)

    time_steps = range(T)
    # Plot the loss curves
    plt.figure(figsize=(6, 4))
    # plt.plot(range(T), loss_total, label="Learned Optimizer", marker='o')
    # plt.plot(range(T), loss_gd_total, label="Gradient descent", marker='s')
    plt.plot(time_steps, mean_loss.cpu().tolist(), label='Mean Loss', color='blue')
    plt.fill_between(time_steps, (mean_loss - std_loss).cpu().tolist(), (mean_loss + std_loss).cpu().tolist(), color='blue', alpha=0.3, label='±1 Std Dev')
    
    plt.plot(time_steps, mean_loss_gd.cpu().tolist(), label='Mean Loss', color='red')
    plt.fill_between(time_steps, (mean_loss_gd - std_loss_gd).cpu().tolist(), (mean_loss_gd + std_loss_gd).cpu().tolist(), color='red', alpha=0.3, label='±1 Std Dev')

    plt.plot(time_steps, mean_loss_nag.cpu().tolist(), label='Mean Loss', color='green')
    plt.fill_between(time_steps, (mean_loss_nag - std_loss_nag).cpu().tolist(), (mean_loss_nag + std_loss_nag).cpu().tolist(), color='green', alpha=0.3, label='±1 Std Dev')

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
