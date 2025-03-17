"""
All loss functions here
"""
import torch


def simple_MSE_loss(pred, label):
    """
    Simple MSE loss
    """
    loss = torch.nn.MSELoss()
    return loss(pred, label)


def sparsity_loss(activations):
    return torch.norm(activations, 1)
