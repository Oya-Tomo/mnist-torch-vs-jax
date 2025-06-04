import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataset import get_datasets, get_batches_torch

from flax import nnx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 5 * 5, 256)  # 64 channels * 5x5 after pooling
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape((x.size(0), -1))  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


cnn = CNN().to(device)
print(cnn)

learning_rate = 1e-3
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)


metrics = nnx.MultiMetric(
    accuracy=nnx.metrics.Accuracy(),
    loss=nnx.metrics.Average("loss"),
)


def loss_fn(model: CNN, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = model(images)
    loss = F.cross_entropy(logits, labels).mean()
    return loss, logits


def train_step(
    model: CNN,
    optimizer: torch.optim.Optimizer,
    metrics: nnx.MultiMetric,
    images: torch.Tensor,
    labels: torch.Tensor,
):
    optimizer.zero_grad()
    loss, logits = loss_fn(model, images, labels)
    loss.backward()
    optimizer.step()
    metrics.update(
        loss=loss.item(),
        logits=logits.cpu().detach().numpy(),
        labels=labels.cpu().detach().numpy(),
    )


def eval_step(
    model: CNN,
    metrics: nnx.MultiMetric,
    images: torch.Tensor,
    labels: torch.Tensor,
):
    with torch.no_grad():
        loss, logits = loss_fn(model, images, labels)
        metrics.update(
            loss=loss.item(),
            logits=logits.cpu().numpy(),
            labels=labels.cpu().numpy(),
        )


metrics_history = {
    "train_loss": [],
    "train_accuracy": [],
    "test_loss": [],
    "test_accuracy": [],
    "train_step_time": [],
    "eval_step_time": [],
}

epochs = 10
batch_size = 32

train_images, train_labels, test_images, test_labels = get_datasets(channel_last=False)
train_images = torch.tensor(train_images, dtype=torch.float32).to(device)
train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)
test_images = torch.tensor(test_images, dtype=torch.float32).to(device)
test_labels = torch.tensor(test_labels, dtype=torch.long).to(device)

start_time = time.perf_counter_ns()
print("Training started...")

for epoch in range(epochs):
    for images, labels in get_batches_torch(
        train_images,
        train_labels,
        batch_size,
    ):
        t = time.perf_counter_ns()
        train_step(cnn, optimizer, metrics, images, labels)
        metrics_history["train_step_time"].append(time.perf_counter_ns() - t)

    for metric, value in metrics.compute().items():
        metrics_history[f"train_{metric}"].append(value)
    metrics.reset()

    for images, labels in get_batches_torch(
        test_images,
        test_labels,
        batch_size,
    ):
        t = time.perf_counter_ns()
        eval_step(cnn, metrics, images, labels)
        metrics_history["eval_step_time"].append(time.perf_counter_ns() - t)

    for metric, value in metrics.compute().items():
        metrics_history[f"test_{metric}"].append(value)
    metrics.reset()

    print(
        f"Epoch {epoch + 1}/{epochs} - "
        f"Train Loss: {metrics_history['train_loss'][-1]:.4f}, "
        f"Train Accuracy: {metrics_history['train_accuracy'][-1]:.4f}, "
        f"Test Loss: {metrics_history['test_loss'][-1]:.4f}, "
        f"Test Accuracy: {metrics_history['test_accuracy'][-1]:.4f}"
    )

end_time = time.perf_counter_ns()
print(f"Training completed in {(end_time - start_time) / 1e9:.10f} seconds.")


with open("result/jax_benchmark.json", "w") as f:
    json.dump(metrics_history, f, indent=4)
