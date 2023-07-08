# First-Order Model Agnostic Meta-Learning algorithm using ResNet18 and CBAM attention module.
# Coded by Tekin O.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import ResNet_CBAM, ResNet18

# Define the ResNet18_CBAM model for feature extraction
embedding_model = ResNet18_CBAM()
#embedding_model = ResNet18()

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "CPU")

# Set the path to the Plant Village dataset
data_path = "/images"
torch.manual_seed(0)

# Define the transform for image preprocessing
transform = transforms.Compose([
    transforms.Resize((126, 126)),
    transforms.ToTensor()
])

# Load the Plant Village dataset for meta-learning
train_dataset = ImageFolder(root=data_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Define the cross-entropy loss and Adam optimizer for feature extraction
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(embedding_model.parameters(), lr=0.001, betas=(0.9, 0.999))

# Train the feature extractor
num_epochs = 100
lr_decay_epochs = [20, 40, 60, 80]

embedding_model.to(device)
embedding_model.train()

for epoch in range(num_epochs):
    if epoch in lr_decay_epochs:
        optimizer.param_groups[0]["lr"] *= 0.1

    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = embedding_model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        if (i + 1) % 1000 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Iteration [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")


# Meta-learning
num_tasks = 50000
num_iterations = 5000
inner_lr = 0.01
outer_lr = 0.01
inner_lr_decay_iter = 1000
inner_steps = 10
eval_tasks = 600

# Define the unseen classes for evaluation
unseen_classes = range(10, 20)

# Create a new dataset for meta-learning
meta_train_dataset = ImageFolder(root=data_path, transform=transform)

# Define the inner loop optimizer
inner_optimizer = optim.SGD(embedding_model.parameters(), lr=inner_lr)

embedding_model.to(device)
embedding_model.train()

for iteration in range(num_iterations):
    meta_gradients = []
    
    for task in range(num_tasks):
        # Sample a task from the meta-training dataset
        task_data = meta_train_dataset[task]
        task_inputs, task_labels = task_data[0].unsqueeze(0).to(device), task_data[1].unsqueeze(0).to(device)

        # Clone the initial parameters of the embedding model
        embedding_model_copy = embedding_model.copy()

        # Perform inner loop optimization
        for inner_step in range(inner_steps):
            inner_optimizer.zero_grad()
            task_outputs = embedding_model_copy(task_inputs)
            task_loss = criterion(task_outputs, task_labels)
            task_loss.backward()
            inner_optimizer.step()

        # Compute gradients of the embedding model parameters
        task_gradients = torch.autograd.grad(task_loss, embedding_model_copy.parameters())

        meta_gradients.append(task_gradients)

    # Update the embedding model parameters using meta-gradients
    for param, meta_gradient in zip(embedding_model.parameters(), meta_gradients):
        meta_gradient_avg = torch.stack(meta_gradient).mean(dim=0)
        param.data -= outer_lr * meta_gradient_avg

    if (iteration + 1) % inner_lr_decay_iter == 0:
        inner_lr *= 0.1
        inner_optimizer = optim.SGD(embedding_model.parameters(), lr=inner_lr)

    # Evaluate the performance on unseen classes
    if (iteration + 1) % 100 == 0:
        embedding_model.eval()
        # Perform evaluation on a subset of unseen classes
        unseen_class_subset = torch.randperm(len(unseen_classes))[:eval_tasks]
        correct = 0
        total = 0

        for task in unseen_class_subset:
            task_data = meta_train_dataset[task]
            task_inputs, task_labels = task_data[0].unsqueeze(0).to(device), task_data[1].unsqueeze(0).to(device)
            task_outputs = embedding_model(task_inputs)
            _, predicted = torch.max(task_outputs.data, 1)
            total += task_labels.size(0)
            correct += (predicted == task_labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Iteration [{iteration+1}/{num_iterations}], Unseen Class Accuracy: {accuracy:.2f}%")

    embedding_model.train()
