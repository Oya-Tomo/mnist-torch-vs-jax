import jax
import numpy as np
import torch
from torchvision import transforms
import torchvision
from typing import Iterator


def get_datasets(channel_last=False):
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_images = []
    train_labels = []
    for img, label in train_dataset:
        train_images.append(img.numpy())
        train_labels.append(label)

    test_images = []
    test_labels = []
    for img, label in test_dataset:
        test_images.append(img.numpy())
        test_labels.append(label)

    train_images = np.array(train_images)  # Shape: (60000, 1, 28, 28)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)  # Shape: (10000, 1, 28, 28)
    test_labels = np.array(test_labels)

    if channel_last:
        train_images = train_images.transpose(0, 3, 2, 1)
        test_images = test_images.transpose(0, 3, 2, 1)

    return train_images, train_labels, test_images, test_labels


def get_batches_torch(
    images, labels, batch_size=32
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    ridx = torch.randperm(len(images))
    for i in range(len(images) // batch_size):
        idx = ridx[i * batch_size : (i + 1) * batch_size]
        yield images[idx], labels[idx]


def get_batches_jax(
    images, labels, batch_size=32, key=None
) -> Iterator[tuple[jax.Array, jax.Array]]:
    ridx = jax.random.permutation(key, len(images))
    for i in range(len(images) // batch_size):
        idx = ridx[i * batch_size : (i + 1) * batch_size]
        yield images[idx], labels[idx]


if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = get_datasets()
    print("Train images shape:", train_images.shape)
    print("Train labels shape:", train_labels.shape)
    print("Test images shape:", test_images.shape)
    print("Test labels shape:", test_labels.shape)
