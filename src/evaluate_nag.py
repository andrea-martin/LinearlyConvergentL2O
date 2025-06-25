import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def evaluate_nag(model, A, x0, dataloader, T, device):
    
    d = A.shape[1] # Number of columns in matrix A (dimension of parameter vector x)
    _, S, _ = torch.linalg.svd(A)
    sigma_max_A = S.max().item() # Maximum singular value of A = maximum eigenvalue of A since A > 0
    sigma_min_A = S.min().item() # Minimum singular value of A = minimum eigenvalue of A since A > 0
    lambda_max_ATA = sigma_max_A**2
    lambda_min_ATA = sigma_min_A**2
    beta = 2*lambda_max_ATA
    alpha = 2*lambda_min_ATA
    kappa = beta/alpha # condition number

    eta = 2/(beta + alpha) # learning rate for learned optimizer

    eta_nag = 4 / (3*beta + alpha) # learning rate for Nesterov
    mu = (np.sqrt(3*kappa + 1) - 2) / (np.sqrt(3*kappa + 1) + 2)
    # mu = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)

    model.eval() # set to evaluation mode

    # Track losses for each sample at each timestep
    loss_per_sample = []  
    loss_gd_per_sample = []
    loss_nag_per_sample = []
    loss_hb_per_sample = []

    v = []
    
    for b_batch in dataloader:
        batch_size = b_batch.shape[0]

    #     # # --------- Learned Optimizer Evaluation ---------
        x_learned_batch = x0.unsqueeze(0).repeat(batch_size, 1, 1)  #torch.zeros(batch_size, d, 1, device=device)
        x_learned_batch_prev = x_learned_batch.clone()
        # Track losses for this batch
        batch_loss_history = torch.zeros((T, batch_size), device=device)

        for t in range(T):
            residual_batch = A@x_learned_batch - b_batch.unsqueeze(-1) # A@x is (batch_size, m) and b is (batch_size, m)
            residual_squared_batch = residual_batch ** 2
            loss_batch = residual_squared_batch.sum(dim=1).unsqueeze(-1)
            
            y_learned_batch = x_learned_batch + mu * (x_learned_batch - x_learned_batch_prev)
            grad_y_batch = 2 * A.T @ (A@y_learned_batch - b_batch.unsqueeze(-1))

            if t < 10000: # All the times, other time we use 200
                with torch.no_grad():
                    v_t_batch = model(x_learned_batch, loss_batch, grad_y_batch, t)
                    v.append(v_t_batch.norm().item())
                x_learned_batch_new = y_learned_batch - eta_nag * grad_y_batch + v_t_batch.unsqueeze(-1)
            else:
                x_learned_batch_new = y_learned_batch - eta_nag * grad_y_batch 
                
            x_learned_batch_prev = x_learned_batch.clone()
            x_learned_batch = x_learned_batch_new

            # if t == 2000:
            #     print(f"Iterazione 2000\n")
            #     print(f"x_learned_batch: {x_learned_batch.view(-1).T}\n, x_learned_batch_prev: {x_learned_batch_prev.view(-1).T}")
            # if t == 6000:
            #     print(f"Iterazione 6000\n")
            #     print(f"x_learned_batch: {x_learned_batch.view(-1).T}\n, x_learned_batch_prev: {x_learned_batch_prev.view(-1).T}")

            # Save loss per sample at this timestep
            batch_loss_history[t] = loss_batch.squeeze()  # Removes extra dimensions
        
        # Store the batch's loss history for analysis later
        loss_per_sample.append(batch_loss_history.cpu())






        # # --------- Gradient Descent (GD) Evaluation ---------
        x_gd_batch = x0.unsqueeze(0).repeat(batch_size, 1, 1) #torch.zeros(batch_size, d, 1, device=device)
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

        # # # --------- Polyak Heavy-Ball (HB) Evaluation ---------
        # x_prev_hb_batch = x0.unsqueeze(0).repeat(batch_size, 1, 1) #torch.zeros(batch_size, d, 1, device=device)
        # x_hb_batch = x_prev_hb_batch.clone()
        # # Track losses for this batch
        # batch_loss_hb_history = torch.zeros((T, batch_size), device=device)

        # for t in range(T):
            
        #     # Evaluate the loss on x_hb_batch to match the gradient computation
        #     residual_batch = A@x_hb_batch - b_batch.unsqueeze(-1)
        #     residual_squared_batch = residual_batch ** 2
            
        #     loss_batch = residual_squared_batch.sum(dim=1).unsqueeze(-1)
            
        #     grad_x_hb_batch = 2 * A.T @ (A@x_hb_batch - b_batch.unsqueeze(-1))

        #     x_next_hb_batch = x_hb_batch - eta * grad_x_hb_batch + mu * (x_hb_batch - x_prev_hb_batch)       
            
        #     # Shift variables for the next iteration
        #     x_prev_hb_batch = x_hb_batch.clone()
        #     x_hb_batch = x_next_hb_batch.clone()

        #     # Save loss per sample at this timestep
        #     batch_loss_hb_history[t] = loss_batch.squeeze()  # Removes extra dimensions

        # # Store the batch's loss history for analysis later
        # loss_hb_per_sample.append(batch_loss_hb_history.cpu())


        # # --------- Nesterov Accelerated Gradient (NAG) Evaluation ---------
        x_prev_nag_batch = x0.unsqueeze(0).repeat(batch_size, 1, 1) #torch.zeros(batch_size, d, 1, device=device)
        x_nag_batch = x_prev_nag_batch.clone()
        # Track losses for this batch
        batch_loss_nag_history = torch.zeros((T, batch_size), device=device)

        for t in range(T):

            
            # Evaluate the loss on y_nag_batch to match the gradient computation
            residual_batch = A@x_nag_batch - b_batch.unsqueeze(-1) # A@y_nag_batch is (batch_size, m) and b is (batch_size, m)
            residual_squared_batch = residual_batch ** 2
            
            loss_batch = residual_squared_batch.sum(dim=1).unsqueeze(-1)
            
            y_nag_batch = x_nag_batch + mu * (x_nag_batch - x_prev_nag_batch)   

            grad_y_nag_batch = 2 * A.T @ (A@y_nag_batch - b_batch.unsqueeze(-1))

            x_next_nag_batch = y_nag_batch - eta_nag * grad_y_nag_batch
            
            # Shift variables for the next iteration
            x_prev_nag_batch = x_nag_batch.clone()
            x_nag_batch = x_next_nag_batch.clone()

            # Save loss per sample at this timestep
            batch_loss_nag_history[t] = loss_batch.squeeze()  # Removes extra dimensions

        # Store the batch's loss history for analysis later
        loss_nag_per_sample.append(batch_loss_nag_history.cpu())

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

    print(f"Average meta loss L2O: {(mean_loss.sum() / T).item():.2f}")
    print(f"Average meta loss GD: {(mean_loss_gd.sum() / T).item():.2f}")
    print(f"Average meta loss NAG: {(mean_loss_nag.sum() / T).item():.2f}")

    # # Combine all batches into one tensor (T, total_samples)
    # loss_hb_per_sample = torch.cat(loss_hb_per_sample, dim=1)
    # # Compute mean and std at each time step
    # mean_loss_hb = loss_hb_per_sample.mean(dim=1)
    # std_loss_hb = loss_hb_per_sample.std(dim=1)

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # Prepare time steps
    time_steps = np.arange(T)
    main_range = slice(0, 10000)
    inset_range = slice(10000, T)

    # Create figure and main axes
    fig, ax = plt.subplots(figsize=(6, 4))

    # --- Main plot ---
    ax.plot(time_steps[main_range], mean_loss[main_range].cpu(), label='L2O', color='blue')
    ax.fill_between(time_steps[main_range], 
                    (mean_loss - std_loss)[main_range].cpu(), 
                    (mean_loss + std_loss)[main_range].cpu(), 
                    color='blue', alpha=0.3)

    ax.plot(time_steps[main_range], mean_loss_gd[main_range].cpu(), label='GD', color='red')
    ax.fill_between(time_steps[main_range], 
                    (mean_loss_gd - std_loss_gd)[main_range].cpu(), 
                    (mean_loss_gd + std_loss_gd)[main_range].cpu(), 
                    color='red', alpha=0.3)

    ax.plot(time_steps[main_range], mean_loss_nag[main_range].cpu(), label='NAG', color='green')
    ax.fill_between(time_steps[main_range], 
                    (mean_loss_nag - std_loss_nag)[main_range].cpu(), 
                    (mean_loss_nag + std_loss_nag)[main_range].cpu(), 
                    color='green', alpha=0.3)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Quadratic Loss")
    # ax.set_title("Optimization Comparison (Initial Phase)")
    ax.legend()
    ax.grid(True)

    # --- Inset plot (10k to 50k, log-log) ---
    # axins = inset_axes(ax, width="35%", height="35%", loc='upper right')

    # axins.plot(time_steps[inset_range], mean_loss[inset_range].cpu(), color='blue')
    # axins.fill_between(time_steps[inset_range], 
    #                 (mean_loss - std_loss)[inset_range].cpu(), 
    #                 (mean_loss + std_loss)[inset_range].cpu(), 
    #                 color='blue', alpha=0.3)

    # axins.plot(time_steps[inset_range], mean_loss_gd[inset_range].cpu(), color='red')
    # axins.fill_between(time_steps[inset_range], 
    #                 (mean_loss_gd - std_loss_gd)[inset_range].cpu(), 
    #                 (mean_loss_gd + std_loss_gd)[inset_range].cpu(), 
    #                 color='red', alpha=0.3)
    # axins.plot(time_steps[inset_range], mean_loss_nag[inset_range].cpu(), color='green')
    # axins.fill_between(time_steps[inset_range], 
    #                 (mean_loss_nag - std_loss_nag)[inset_range].cpu(), 
    #                 (mean_loss_nag + std_loss_nag)[inset_range].cpu(), 
    #                 color='green', alpha=0.3)

    # axins.set_xscale("log")
    # axins.set_yscale("log")
    # axins.set_xticks([10000, 30000, 50000])
    # axins.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # axins.tick_params(axis='both', which='both', labelsize=6)
    # axins.set_title("Asymptotic (log-log)", fontsize=9)

    # plt.legend()
    # plt.tight_layout()
    plt.show()

    # time_steps = range(T)
    # # Plot the loss curves
    # plt.figure(figsize=(6, 4))
    # plt.plot(time_steps, mean_loss.cpu().tolist(), label='L2O', color='blue')
    # plt.fill_between(time_steps, (mean_loss - std_loss).cpu().tolist(), (mean_loss + std_loss).cpu().tolist(), color='blue', alpha=0.3)
    
    # plt.plot(time_steps, mean_loss_gd.cpu().tolist(), label='GD', color='red')
    # plt.fill_between(time_steps, (mean_loss_gd - std_loss_gd).cpu().tolist(), (mean_loss_gd + std_loss_gd).cpu().tolist(), color='red', alpha=0.3)

    # plt.plot(time_steps, mean_loss_nag.cpu().tolist(), label='NAG', color='green')
    # plt.fill_between(time_steps, (mean_loss_nag - std_loss_nag).cpu().tolist(), (mean_loss_nag + std_loss_nag).cpu().tolist(), color='green', alpha=0.3)

    # # plt.plot(time_steps, mean_loss_hb.cpu().tolist(), label='HB', color='yellow')
    # # plt.fill_between(time_steps, (mean_loss_hb - std_loss_hb).cpu().tolist(), (mean_loss_hb + std_loss_hb).cpu().tolist(), color='yellow', alpha=0.3)

    # # plt.yscale('log')
    # plt.xlabel("Iteration")
    # plt.ylabel("Quadratic Loss")
    # plt.title("Optimization Comparison on a Random Quadratic Task")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # # # --- After the evaluation loops, add this code to plot the v_t values ---
    # v_array = np.array(v)
    # # Reshape v_array assuming there are multiple batches: each batch contributes T iterations.
    # # This computes the average norm per iteration across batches.
    # v_avg = v_array.reshape(-1, T).mean(axis=0)
    
    # plt.figure(figsize=(6, 4))
    # plt.plot(range(T), v_avg, label="||v_t||", marker='o')
    # plt.xlabel("Iteration")
    # plt.ylabel("||v_t||")
    # plt.title("Norm of Learned Update v_t over Iterations")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()



