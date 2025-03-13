import torch

def meta_training(model, A, dataloader, meta_optimizer, T, device, epochs):

    m = A.shape[0] # Number of rows in matrix A (dimension of output vector b)
    d = A.shape[1] # Number of columns in matrix A (dimension of parameter vector x)
    U, S, V = torch.svd(A)
    lambda_max = S.max().item() ** 2
    eta = 1/(2*lambda_max)/2
    
    dataloader_iter = iter(dataloader)
    b_batch = next(dataloader_iter)
    batch_size = b_batch.shape[0]

    model.train() # set to training mode

    print("starting training....")
    for epoch in range(epochs):
        
        meta_optimizer.zero_grad()
        meta_loss = torch.zeros(1, device=device)
        
        # Initialize parameter vector x (shape: (d, 1))
        x_batch = torch.zeros(batch_size, d, 1, device=device)
        
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
            v_t_batch = model(x_batch, loss_batch, grad_batch, t)

            # Update parameter vector: standard gradient descent step plus the learned update v_t
            x_batch = x_batch - eta * grad_batch + v_t_batch.unsqueeze(-1)

            # Accumulate the weighted loss (meta-loss) over the inner trajectory
            meta_loss = meta_loss +  loss_batch.sum()

        meta_loss = meta_loss / batch_size
        print(f"Epoch {epoch+1} - Average Meta Loss: {meta_loss.item()}")
        
        meta_loss.backward()
        meta_optimizer.step()

    return model