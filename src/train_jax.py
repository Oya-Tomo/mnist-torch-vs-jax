import time

import jax
import jax.numpy as jnp  # JAX NumPy
from flax import nnx
from flax.training import train_state  # Useful dataclass to keep train state
from flax import struct  # Flax dataclasses
import optax
from functools import partial
from clu import metrics

from dataset import get_datasets


class CNN(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(
            in_features=1,
            out_features=32,
            kernel_size=(3, 3),
            padding="VALID",
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            in_features=32,
            out_features=64,
            kernel_size=(3, 3),
            padding="VALID",
            rngs=rngs,
        )
        self.avg_pool = partial(
            nnx.avg_pool,
            window_shape=(2, 2),
            strides=(2, 2),
            padding="VALID",
        )
        self.fc1 = nnx.Linear(
            in_features=64 * 5 * 5,
            out_features=256,
            rngs=rngs,
        )
        self.fc2 = nnx.Linear(
            in_features=256,
            out_features=10,
            rngs=rngs,
        )

    def __call__(self, x):
        x = self.conv1(x)
        x = nnx.relu(x)
        x = self.avg_pool(x)
        x = self.conv2(x)
        x = nnx.relu(x)
        x = self.avg_pool(x)
        x = x.reshape((x.shape[0], -1))
        x = self.fc1(x)
        x = nnx.relu(x)
        x = self.fc2(x)
        return x


cnn = CNN(rngs=nnx.Rngs(0))
print(nnx.tabulate(cnn, jnp.ones([1, 28, 28, 1])))
