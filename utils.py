import torch
import os
import matplotlib.pyplot as plt
import numpy as np
# Function to compute Pearson correlation
@torch.no_grad()
def pearson_correlation(x, y):
    x = x.flatten()
    y = y.flatten()

    # Validate inputs
    if x.shape != y.shape:
        raise ValueError("Inputs must have the same shape")
    if torch.any(torch.isnan(x)) or torch.any(torch.isnan(y)):
        raise ValueError("Inputs contain NaN values")
    if torch.any(torch.isinf(x)) or torch.any(torch.isinf(y)):
        raise ValueError("Inputs contain infinite values")

    x_centered = x - x.mean()
    y_centered = y - y.mean()
    cov = torch.dot(x_centered, y_centered) / (x.shape[0] - 1)
    std_x = x_centered.norm() / ((x.shape[0] - 1) ** 0.5)
    std_y = y_centered.norm() / ((y.shape[0] - 1) ** 0.5)

    # Check for zero variance or near-zero denominator
    if std_x < 1e-10 or std_y < 1e-10:
        return float('nan')  # Undefined correlation due to zero variance

    corr = cov / ((std_x * std_y) + 1e-3)

    if torch.isinf(corr):
        return torch.tensor(-1.0, dtype=x.dtype)
    return corr

def plot_bchw_tensor(tensor, title="Images", save_path=None):
    """
    Plot the first two images from a BCHW tensor in a row.
    
    Args:
        tensor (torch.Tensor): Tensor of shape (B, C, H, W)
        title (str): Title for the plot
        save_path (str, optional): Path to save the plot
    """
    if tensor.shape[0] < 2:
        raise ValueError("Batch size must be at least 2 to plot two images.")
    
    # Convert tensor to numpy and transpose from (B, C, H, W) to (B, H, W, C)
    images = tensor.detach().cpu().float().numpy()
    images = np.transpose(images, (0, 2, 3, 1))  # Shape: (B, H, W, C)
    
    # Normalize to [0, 1] if necessary (assuming pixel values are not already normalized)
    if images.max() > 1.0 or images.min() < 0.0:
        images = (images - images.min()) / (images.max() - images.min())
    
    # Create a figure with two subplots in a row
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    # Plot the first two images
    for i in range(2):
        axes[i].imshow(images[i])
        axes[i].set_title(f"Image {i+1}")
        axes[i].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()