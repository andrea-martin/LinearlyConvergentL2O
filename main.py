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

import seaborn as sns

from src.models import LearnedUpdate
from src.meta_training import meta_training 
from src.meta_training_nag import meta_training_nag
from src.evaluate import evaluate
from src.evaluate_nag import evaluate_nag

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

    # Step 3: Locate .mtx files inside the extracted folder
    mtx_files = glob.glob(f"{extraction_dir}/**/*.mtx", recursive=True)

    if not mtx_files:
        raise FileNotFoundError(f"No .mtx file found after extraction for '{matrix_name}'!")

    # Step 4: Handle multiple .mtx files intelligently
    if len(mtx_files) == 1:
        mtx_file_path = mtx_files[0]
    else:
        # Prefer files matching the matrix name exactly or common patterns
        preferred_files = [f for f in mtx_files if matrix_name in os.path.basename(f).lower()]
        
        if len(preferred_files) == 1:
            mtx_file_path = preferred_files[0]
        else:
            # If still ambiguous, ask the user which one to use
            print("⚠️ Multiple .mtx files found. Please select one:")
            for idx, file in enumerate(mtx_files):
                print(f"  [{idx+1}] {file}")
            choice = int(input("Enter the number of the file you want to load: ")) - 1
            mtx_file_path = mtx_files[choice]

    # Step 5: Load the selected matrix file
    matrix = mmread(mtx_file_path)

    # Step 6: Print matrix shape and return the matrix
    print(f"✅ Loaded matrix '{matrix_name}' from '{os.path.basename(mtx_file_path)}' with shape: {matrix.shape}")
    return matrix

def main():
    # -------------------------------------------------------------------------------
    # The task is to minimize f(x) = ||Ax - b||², with gradient ∇f(x) = 2 Aᵀ (Ax - b)
    # -----------------------------------------------------------------------------
    # A = load_sparse_matrix_from_ssget("685_bus")
    A = load_sparse_matrix_from_ssget("bcsstk02")
    # A = load_sparse_matrix_from_ssget("bcsstk09")
    # A = load_sparse_matrix_from_ssget("1138_bus")
    # A = load_sparse_matrix_from_ssget("msc01440")
    # A = load_sparse_matrix_from_ssget("Trefethen_20b")
    # A = load_sparse_matrix_from_ssget("Journals")
    A = torch.tensor(A.toarray(), dtype=torch.float32, device=device)

    sns.heatmap(A.numpy(), annot=True, cmap='coolwarm')  # annot=True adds numbers


    m = A.shape[0] # Number of rows in matrix A (dimension of output vector b)
    d = A.shape[1] # Number of columns in matrix A (dimension of parameter vector x)

    x0 = 1e-4 * torch.rand(d, 1, device=device) # Initial parameter vector   

    training_samples = 1024 # Number of linear regression tasks to sample for training
    T = 50000  # Number of iterations for the inner loop 
    epochs = 50 # Number of epochs for meta-training  

    # Instantiate the training dataset and dataloader
    training_dataset = LinearRegressionDataset(m=m, d=d, num_samples=training_samples, device=device)
    training_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True)
    
    load = True
    if load:
        # Load the pre-trained learned optimizer parameters
        directory_path = './trained_models/'
        file_path = os.path.join(directory_path, 'bcsstk02_nag_50epochs_10000T200_rho95_02randn.pt')
        if os.path.exists(file_path):
            print(f"Loading pre-trained learned optimizer from {file_path}")
            learned_update = LearnedUpdate(d, q=0, rho=0.95, hidden_sizes=[256, 256, 256], architecture='lstm').to(device)
            learned_update.load_state_dict(torch.load(file_path))
            print("Pre-trained learned optimizer loaded successfully.")
        else:
            print(f"Pre-trained model not found at {file_path}.")
    else:
        # Initialize the LearnedUpdate module with a fixed rho (e.g., 0.99)
        learned_update = LearnedUpdate(d, q=0, rho=0.95, hidden_sizes=[256, 256, 256], architecture='lstm').to(device)
        print(f"Total parameters in LearnedUpdate: {sum(p.numel() for p in learned_update.parameters() if p.requires_grad)}")

        # Meta optimizer (updates both the MLP and the alpha parameters)
        meta_optimizer = torch.optim.Adam(learned_update.parameters(), lr=1e-3)

        learned_update = meta_training_nag(learned_update, A, x0, training_dataloader, meta_optimizer, T, device, epochs=epochs)
        # Save the trained learned optimizer parameters
        directory_path = './trained_models/'
        os.makedirs(directory_path, exist_ok=True)
        file_path = os.path.join(directory_path, 'bcsstk02_nag_50epochs_10000T200_rho95_02randn.pt')
        torch.save(learned_update.state_dict(), file_path)
        print(f"Trained learned optimizer saved to {file_path}")

    test_samples = 128 # Number of linear regression tasks to sample for testing
    # Instantiate the test dataset and dataloader
    test_dataset = LinearRegressionDataset(m=m, d=d, num_samples=test_samples, device=device)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    # Evaluate the learned optimizer
    evaluate_nag(learned_update, A, x0, test_dataloader, T, device)

if __name__ == "__main__":
    main()