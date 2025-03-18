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
        # Generate A matrix with custom eigenvalues
        # min_dim = min(self.m, self.d)        

        # eigenvalues_ATA = torch.randint(1, 1001, (self.m - 2,), device=self.device, dtype=torch.float)
        # eigenvalues_ATA = torch.cat([eigenvalues_ATA, torch.tensor([1.0, 1000.0], device=self.device)])
        # eigenvalues_ATA = eigenvalues_ATA[torch.randperm(len(eigenvalues_ATA))]
        
        # A = torch.zeros(self.m, self.d, device=self.device)
        # A[:min_dim, :min_dim] = torch.diag(torch.sqrt(eigenvalues_ATA[:min_dim]))

        # Generate b vector randomly
        b = 0.5 * torch.ones(self.m, device=self.device) + 0.25 * torch.rand(self.m, device=self.device)

        return b