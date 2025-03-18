import torch

def meta_training(model, A, x0, dataloader, meta_optimizer, T, device, epochs):

    m = A.shape[0] # Number of rows in matrix A (dimension of output vector b)
    d = A.shape[1] # Number of columns in matrix A (dimension of parameter vector x)
    _, S, _ = torch.linalg.svd(A)
    sigma_max_A = S.max().item() # Maximum singular value of A = maximum eigenvalue of A since A > 0
    sigma_min_A = S.min().item() # Minimum singular value of A = minimum eigenvalue of A since A > 0
    lambda_max_ATA = sigma_max_A**2
    lambda_min_ATA = sigma_min_A**2
    beta = 2*lambda_max_ATA
    alpha = 2*lambda_min_ATA
    
    #eta = 1/(2*lambda_max)/2
    eta = 2/(beta + alpha)

    dataloader_iter = iter(dataloader)
    b_batch = next(dataloader_iter)
    batch_size = b_batch.shape[0]

    model.train() # set to training mode

    for epoch in range(epochs):
        
        meta_optimizer.zero_grad()
        meta_loss = torch.zeros(1, device=device)
        
        # Initialize parameter vector x (shape: (d, 1))
        x_batch = x0.unsqueeze(0).repeat(batch_size, 1, 1) #torch.zeros(batch_size, d, 1, device=device)
        
        for t in range(T):
            # Gradient computation
            try:
                b_batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                b_batch = next(dataloader_iter)


            # Compute f(x) = ||Ax - b||²
            residual_batch = A@x_batch - b_batch.unsqueeze(-1) # A@x is (batch_size, m) and b is (batch_size, m)
            residual_squared_batch = residual_batch ** 2
            loss_batch = residual_squared_batch.sum(dim=1).unsqueeze(-1)
            
            # Compute gradient: ∇f(x) = 2 * Aᵀ * (Ax - b)
            grad_batch = 2 * A.T @ residual_batch  # shape: (batch_size, d, 1)
            
            # Prepare inputs for the learned update:
            # loss_val = loss_batch.view(1, 1)  # shape: (1, 1)
            v_t_batch = model(x_batch, 0*loss_batch, 0*grad_batch, t)

            # Update parameter vector: standard gradient descent step plus the learned update v_t
            x_batch = x_batch - eta * grad_batch + v_t_batch.unsqueeze(-1)

            # Accumulate the weighted loss (meta-loss) over the inner trajectory
            meta_loss = meta_loss +  loss_batch.sum()

        print_every = 1
        if epoch == 0 or (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch+1} - average meta loss: {(meta_loss / (T*batch_size)).item():.2f}")
        
        meta_loss.backward()
        meta_optimizer.step()

    return model