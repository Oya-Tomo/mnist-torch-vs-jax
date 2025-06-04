import time

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from dataset import get_batches_jax, get_datasets


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

    @nnx.jit
    def __call__(self, x) -> jax.Array:
        x = self.conv1(x)
        x = nnx.relu(x)
        x = nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = self.conv2(x)
        x = nnx.relu(x)
        x = nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = x.reshape((x.shape[0], -1))
        x = self.fc1(x)
        x = nnx.relu(x)
        x = self.fc2(x)
        return x


cnn = CNN(rngs=nnx.Rngs(0))
print(nnx.tabulate(cnn, jnp.ones([1, 28, 28, 1])))


learning_rate = 1e-3
optimizer = nnx.Optimizer(cnn, optax.adam(learning_rate=learning_rate))
metrics = nnx.MultiMetric(
    accuracy=nnx.metrics.Accuracy(),
    loss=nnx.metrics.Average("loss"),
)


@nnx.jit
def loss_fn(model: nnx.Module, images: jax.Array, labels: jax.Array):
    logits = model(images)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits,
        labels=labels,
    ).mean()
    return loss, logits


@nnx.jit
def train_step(
    model: CNN,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    images: jax.Array,
    labels: jax.Array,
):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, images, labels)
    metrics.update(loss=loss, logits=logits, labels=labels)
    optimizer.update(grads)


@nnx.jit
def eval_step(
    model: CNN,
    metrics: nnx.MultiMetric,
    images: jax.Array,
    labels: jax.Array,
):
    loss, logits = loss_fn(model, images, labels)
    metrics.update(loss=loss, logits=logits, labels=labels)


metrics_history = {
    "train_loss": [],
    "train_accuracy": [],
    "test_loss": [],
    "test_accuracy": [],
}

epochs = 10
batch_size = 32

train_images, train_labels, test_images, test_labels = get_datasets(channel_last=True)
train_images = jnp.array(train_images, dtype=jnp.float32)
train_labels = jnp.array(train_labels, dtype=jnp.int32)
test_images = jnp.array(test_images, dtype=jnp.float32)
test_labels = jnp.array(test_labels, dtype=jnp.int32)

start_time = time.perf_counter_ns()
print("Training started...")

for epoch in range(epochs):
    for images, labels in get_batches_jax(
        train_images,
        train_labels,
        batch_size=batch_size,
        key=jax.random.PRNGKey(epoch),
    ):
        train_step(cnn, optimizer, metrics, images, labels)

    for metric, value in metrics.compute().items():
        metrics_history[f"train_{metric}"].append(value)
    metrics.reset()

    for images, labels in get_batches_jax(
        test_images,
        test_labels,
        batch_size=batch_size,
        key=jax.random.PRNGKey(epoch + 1000),
    ):
        eval_step(cnn, metrics, images, labels)

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
