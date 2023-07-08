# heatmap visualization for the attention features of the ResNet18_CBAM model
# Coded by Tekin O.

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# Set random seed for reproducibility
torch.manual_seed(0)

# Define the ResNet18_CBAM model for feature extraction
embedding_model = ResNet18_CBAM()

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained ResNet18_CBAM model
checkpoint = torch.load("resnet18_cbam_model.pth")
embedding_model.load_state_dict(checkpoint['model_state_dict'])
embedding_model.to(device)
embedding_model.eval()

# Define the hook to extract attention maps
attention_maps = []

def hook_fn(module, input, output):
    attention_maps.append(output[0].squeeze().cpu().numpy())

embedding_model.cbam.channel_attention.register_forward_hook(hook_fn)
embedding_model.cbam.spatial_attention.register_forward_hook(hook_fn)

# Extract feature maps and attention maps from the dataset using ResNet18_CBAM
feature_maps = []
labels = []

with torch.no_grad():
    for images, target in test_loader:
        images = images.to(device)
        target = target.to(device)

        # Extract feature map from ResNet18_CBAM
        feature_map = embedding_model(images)
        feature_map = feature_map.squeeze().cpu().numpy()

        feature_maps.append(feature_map)
        labels.append(target.item())

# Convert attention maps to numpy array
attention_maps = np.array(attention_maps)

# Plot the attention heatmaps
fig, axs = plt.subplots(nrows=len(feature_maps), ncols=3, figsize=(8, 8))
fig.suptitle("Attention Heatmaps")

for i, (feature_map, attention_map, label) in enumerate(zip(feature_maps, attention_maps, labels)):
    axs[i, 0].imshow(np.transpose(feature_map, (1, 2, 0)))
    axs[i, 0].axis('off')
    axs[i, 0].set_title("Feature Map")

    axs[i, 1].imshow(attention_map[0], cmap='hot')
    axs[i, 1].axis('off')
    axs[i, 1].set_title("Channel Attention")

    axs[i, 2].imshow(attention_map[1], cmap='hot')
    axs[i, 2].axis('off')
    axs[i, 2].set_title("Spatial Attention")

    if i == len(feature_maps) - 1:
        # Add colorbar for the spatial attention heatmap
        cax = axs[i, 2].imshow(attention_map[1], cmap='hot')
        fig.colorbar(cax, ax=axs[i, 2])

plt.tight_layout()
plt.show()
