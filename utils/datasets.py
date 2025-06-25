import torch
from torch.utils.data import Dataset

class LinearRegressionDataset(Dataset):
    def __init__(self, m, d, num_samples, device):
        """
        Generates (A, b) pairs on the fly.
        
        Args:
            m (int): Number of rows in A
            d (int): Number of columns in A
            num_samples (int): Number of (A, b) pairs to generate
            device (torch.device): Device (CPU/GPU)
        """
        self.m = m
        self.d = d
        self.num_samples = num_samples
        self.device = device

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        
        # Add a random perturbation to the A matrix
        # binary_mask = (A != 0).int()

        # Generate b vector randomly
        b = 0.5 * torch.ones(self.m, device=self.device) + 0.2 * torch.randn(self.m, device=self.device)

        return b