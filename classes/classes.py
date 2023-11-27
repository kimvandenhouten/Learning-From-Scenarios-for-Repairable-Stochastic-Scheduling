import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, y_true):
        self.y_true = y_true

    def __len__(self):
        return len(self.y_true)

    def __getitem__(self, idx):
        sample = {'y_true': self.y_true[idx],
                  'idx': idx}
        return sample


class Model(nn.Module):
    def __init__(self, dim):
        super(Model, self).__init__()
        # Define two learnable parameters: ln_theta_mu and ln_theta_sigma
        self.ln_theta_mu = nn.Parameter(torch.zeros(dim))  # Initial value can be set as needed
        self.ln_theta_sigma = nn.Parameter(torch.zeros(dim))  # Initial value can be set as needed

    def forward(self, mu_bar, sigma_bar):
        # Calculate theta_mu and theta_sigma using the learned parameters
        theta_mu = torch.exp(self.ln_theta_mu)
        theta_sigma = torch.exp(self.ln_theta_sigma)

        # Calculate mu and sigma using theta_mu and theta_sigma
        mu = theta_mu * mu_bar
        sigma = theta_sigma * sigma_bar
        return mu, sigma


class LinearModel(nn.Module):
    def __init__(self, dim):
        super(LinearModel, self).__init__()
        # Define two learnable parameters: ln_theta_mu and ln_theta_sigma
        self.ln_theta_mu = nn.Parameter(torch.zeros(dim))  # Initial value can be set as needed

    def forward(self, mu_bar):
        # Calculate theta_mu and theta_sigma using the learned parameters
        theta_mu = torch.exp(self.ln_theta_mu)

        # Calculate mu and sigma using theta_mu and theta_sigma
        mu = theta_mu * mu_bar
        return mu