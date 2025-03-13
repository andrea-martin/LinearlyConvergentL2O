import torch

def meta_training(model, A, dataloader, meta_optimizer, T, device, epochs):

    d = int(model.get_dimension()) # Number of columns in matrix A (dimension of parameter vector x)

    # # gamma = 5e-2    # Discount factor for weighting the meta-loss

    # # eigenvalues_ATA = torch.randint(1, 1001, (m - 2,), device=device, dtype=torch.float)
    # # eigenvalues_ATA = torch.cat([eigenvalues_ATA, torch.tensor([1.0, 1000.0], device=device)])
    # # Sigma = torch.zeros(m, d, device=device)
    # # min_dim = min(m, d)
    # # Sigma[:min_dim, :min_dim] = torch.diag(torch.sqrt(eigenvalues_ATA[:min_dim]))

    # # beta = 2 * max(eigenvalues_ATA)
    # # eta = 1/beta/5

    # iterations = 30
    # T = 50          # Number of inner optimization steps (per task)

    model.train() # set to training mode

    print("starting training....")
    for epoch in range(epochs):
        
        meta_optimizer.zero_grad()
        meta_loss = torch.zeros(1, device=device)
        
        # Initialize parameter vector x (shape: (d, 1))
        x = torch.ones(d, 1, device=device)
        
        dataloader_iter = iter(dataloader)

        for t in range(T):

            # Gradient computation

            try:
                b = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                b = next(dataloader_iter)


            # Compute f(x) = ||Ax - b||²
            residual = A@x.squeeze(-1) - b # A@x is (batch_size, m) and b is (batch_size, m)
            loss = torch.sum(residual ** 2)
            loss = loss.view(1, 1)  # scalar loss, convert to shape (1, 1)
            # Compute gradient: ∇f(x) = 2 * Aᵀ * (Ax - b)
            grad = 2 * A.transpose(1, 2) @ residual.unsqueeze(-1)  # shape: (batch_size, d, 1)
            grad = grad.mean(dim=0)  # average gradient over the batch, shape: (d, 1)
            
            # Prepare inputs for the learned update:
            loss_val = loss.view(1, 1)  # shape: (1, 1)
            v_t = model(x.T, loss_val, grad.T, t)

            # Update parameter vector: standard gradient descent step plus the learned update v_t
            x = x - 0.0001 * grad + v_t.T

            # Accumulate the weighted loss (meta-loss) over the inner trajectory
            meta_loss = meta_loss + .95 ** (A.shape[0] - 1 - t) * loss

        meta_loss = meta_loss / len(dataloader) 
        print(f"Epoch {epoch+1} - Average Meta Loss: {meta_loss.item()}")
        
        meta_loss.backward()
        meta_optimizer.step()

    return model


    #     for i in range(iterations):
    #         # Sample a new quadratic task: f(x) = ||Ax - b||²
    #         # Generate random orthogonal matrices U and V
    #         #U, _ = torch.qr(torch.randn(m, m, device=device))
    #         #V, _ = torch.qr(torch.randn(d, d, device=device))
    #         # Construct A using U, Sigma, and V^T
    #         #A = U @ Sigma @ V.T
    #         #A = Sigma 

    #         # Create a diagonal matrix with the desired singular values
    #         eigenvalues_ATA = torch.randint(1, 1001, (m - 2,), device=device, dtype=torch.float)
    #         eigenvalues_ATA = torch.cat([eigenvalues_ATA, torch.tensor([1.0, 1000.0], device=device)])
    #         A = torch.zeros(m, d, device=device)
    #         A[:min_dim, :min_dim] = torch.diag(torch.sqrt(eigenvalues_ATA[:min_dim]))

    #         b = torch.ones(m, 1, device=device)
    #         # Initialize parameter vector x (shape: (1, d))
    #         x = torch.zeros(1, d, device=device)

    #         mloss = torch.zeros(1, device=device)

    #         for t in range(T):
    #             # Compute f(x) = ||Ax - b||²
    #             Ax = torch.matmul(A, x.T)                  # shape: (m, 1)
    #             loss = torch.sum((Ax - b) ** 2) # scalar loss

    #             # Compute gradient: ∇f(x) = 2 Aᵀ (Ax - b)
    #             grad = 2 * torch.matmul(A.T, (Ax - b))  # shape: (d, 1)
    #             grad = grad.T  # reshape to (1, d)

    #             # Prepare inputs for the learned update:
    #             loss_val = loss.view(1, 1)  # shape: (1, 1)
    #             v_t = model(x, loss_val, grad, t)

    #             # Update parameter vector: standard gradient descent step plus the learned update v_t
    #             x = x - eta * grad + v_t

    #             # Accumulate the weighted loss (meta-loss) over the inner trajectory
    #             mloss = mloss + .95 ** (T - 1 - t) * loss

    #         meta_loss = meta_loss + mloss

    #     meta_loss = meta_loss / iterations
    #     print(f"Epoch {epoch+1} - Average Meta Loss: {meta_loss.item()}")

    #     meta_loss.backward()
    #     meta_optimizer.step()

    # return model