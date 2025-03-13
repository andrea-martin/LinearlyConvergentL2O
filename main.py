import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import ssgetpy
from scipy.io import mmread
import tarfile
import glob

from src.models import SimpleMLP
from src.models import LearnedUpdate
from src.meta_training import meta_training 
from src.evaluate import evaluate

from utils.datasets import LinearRegressionDataset

# Set device and seeds
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1234)
np.random.seed(1234)

def load_sparse_matrix_from_ssget(matrix_name):
    """
    Downloads, extracts, and loads a matrix from the SuiteSparse Matrix Collection
    given a matrix name.

    Args:
        matrix_name (str): The name of the matrix to download (e.g., "bcsstk13").

    Returns:
        scipy.sparse.csr_matrix: The loaded sparse matrix.
    """
    # Step 1: Fetch and download the matrix
    matrix_data = ssgetpy.search(name=matrix_name)[0].download()

    # Step 2: Extract the tar.gz file
    extraction_dir = os.path.dirname(matrix_data[0])
    with tarfile.open(matrix_data[0], "r:gz") as tar:
        tar.extractall(path=extraction_dir)

    # Step 3: Locate the .mtx file inside the extracted folder
    mtx_files = glob.glob(f"{extraction_dir}/**/*.mtx", recursive=True)

    if not mtx_files:
        raise FileNotFoundError(f"No .mtx file found after extraction for {matrix_name}!")

    # Step 4: Load the matrix from the .mtx file
    matrix = mmread(mtx_files[0])

    # Step 5: Print matrix shape and return the matrix
    print(f"✅ Loaded matrix '{matrix_name}' with shape: {matrix.shape}")
    return matrix

def main():
    # -------------------------------------------------------------------------------
    # The task is to minimize f(x) = ||Ax - b||², with gradient ∇f(x) = 2 Aᵀ (Ax - b)
    # -------------------------------------------------------------------------------
    A = load_sparse_matrix_from_ssget("bcsstk13")
    A = torch.tensor(A.toarray(), dtype=torch.float32, device=device)
    m = A.shape[0] # Number of rows in matrix A (dimension of output vector b)
    d = A.shape[1] # Number of columns in matrix A (dimension of parameter vector x)

    training_samples = 64 # Number of linear regression tasks to sample for training

    # Instantiate the training dataset and dataloader
    training_dataset = LinearRegressionDataset(m=m, d=d, num_samples=training_samples, device=device)
    training_dataloader = DataLoader(training_dataset, batch_size=2, shuffle=True)
    
    # Initialize the LearnedUpdate module with a fixed rho (e.g., 0.99)
    learned_update = LearnedUpdate(d, q=0, rho=0.99, hidden_size1=5, hidden_size2=5).to(device)
    print(f"Total parameters in LearnedUpdate: {sum(p.numel() for p in learned_update.parameters() if p.requires_grad)}")

    # Meta optimizer (updates both the MLP and the alpha parameters)
    meta_optimizer = torch.optim.Adam(learned_update.parameters(), lr=1e-3)

    learned_update = meta_training(learned_update, A, training_dataloader, meta_optimizer, m, device, epochs=3)

    # Save the trained learned optimizer parameters
    directory_path = './trained_models/'
    os.makedirs(directory_path, exist_ok=True)
    file_path = os.path.join(directory_path, 'learnt_optimizer_quadratic_updated.pt')
    torch.save(learned_update.state_dict(), file_path)
    print(f"Trained learned optimizer saved to {file_path}")

    # test_samples = 128  # Number of linear regression tasks to sample for testing
    # # Instantiate the test dataset and dataloader
    # test_dataset = LinearRegressionDataset(m=m, d=d, num_samples=test_samples, device=device)
    # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Evaluate the learned optimizer
    evaluate(learned_update, A, m, d, device)

if __name__ == "__main__":
    main()