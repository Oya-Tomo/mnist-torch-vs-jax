import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


def get_datasets(batch_size):
    """Load MNIST train and test datasets."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # automatically normalizes to [0, 1]
            transforms.Lambda(
                lambda x: x.view(28, 28, 1)
            ),  # reshape to match JAX format
        ]
    )

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    return train_loader, test_loader


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 5 * 5, 256)  # 64 channels * 5x5 after pooling
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # Input shape: (batch_size, 28, 28, 1) -> need to permute to (batch_size, 1, 28, 28)
        x = x.permute(0, 3, 1, 2)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape((x.size(0), -1))  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_step(model, optimizer, criterion, batch):
    """Train for a single step."""
    images, labels = batch

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    return loss.item()


def compute_metrics(model, data_loader, criterion, device):
    """Compute loss and accuracy on a dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total

    model.train()
    return avg_loss, accuracy

    # Move to CPU for visualization


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.01
    momentum = 0.9

    # Load datasets
    train_loader, test_loader = get_datasets(batch_size)

    # Initialize model, optimizer, and loss function
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Print model summary
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    # Training loop
    metrics_history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    start_time = time.perf_counter_ns()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Train step
            loss = train_step(model, optimizer, criterion, (images, labels))
            train_loss += loss

            # Compute training accuracy
            with torch.no_grad():
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

        # Compute epoch metrics
        train_loss_avg = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total

        # Compute test metrics
        test_loss_avg, test_accuracy = compute_metrics(
            model, test_loader, criterion, device
        )

        # Record metrics
        metrics_history["train_loss"].append(train_loss_avg)
        metrics_history["train_accuracy"].append(train_accuracy)
        metrics_history["test_loss"].append(test_loss_avg)
        metrics_history["test_accuracy"].append(test_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(
            f"Train - Loss: {train_loss_avg:.4f}, Accuracy: {train_accuracy*100:.2f}%"
        )
        print(f"Test  - Loss: {test_loss_avg:.4f}, Accuracy: {test_accuracy*100:.2f}%")
        print("-" * 50)

    print(f"Training completed in {time.perf_counter_ns() - start_time} ns")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.set_title("Loss")
    ax1.plot(metrics_history["train_loss"], label="train_loss")
    ax1.plot(metrics_history["test_loss"], label="test_loss")
    ax1.legend()
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    ax2.set_title("Accuracy")
    ax2.plot(metrics_history["train_accuracy"], label="train_accuracy")
    ax2.plot(metrics_history["test_accuracy"], label="test_accuracy")
    ax2.legend()
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")

    plt.tight_layout()
    plt.show()

    # Visualize predictions
    model.eval()
    test_batch = next(iter(test_loader))
    images, labels = test_batch
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

    # Move to CPU for visualization
    images = images.cpu()
    predictions = predictions.cpu()

    fig, axs = plt.subplots(5, 5, figsize=(12, 12))
    for i, ax in enumerate(axs.flatten()):
        if i < len(images):
            # Convert back to original format for visualization
            img = images[i].squeeze().numpy()
            ax.imshow(img, cmap="gray")
            ax.set_title(f"Pred: {predictions[i].item()}")
            ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
