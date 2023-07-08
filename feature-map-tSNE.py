import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


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

# Extract features from the dataset using ResNet18_CBAM
features = []
labels = []

with torch.no_grad():
    for images, target in test_loader:
        images = images.to(device)
        target = target.to(device)

        # Extract feature map from ResNet18_CBAM
        feature_map = embedding_model(images)
        feature_map = feature_map.squeeze().cpu().numpy()

        features.append(feature_map)
        labels.append(target.item())

features = torch.tensor(features)
labels = torch.tensor(labels)

# Reshape feature maps
features = features.view(features.size(0), -1)

# Perform t-SNE visualization
tsne = TSNE(n_components=2, random_state=0)
features_tsne = tsne.fit_transform(features)

# Plot the t-SNE visualization
plt.figure(figsize=(10, 8))
scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, cmap='tab10')
plt.legend(handles=scatter.legend_elements()[0], labels=test_dataset.classes)
plt.title("t-SNE Visualization of ResNet18_CBAM Feature Maps")
plt.colorbar(scatter)
plt.show()
