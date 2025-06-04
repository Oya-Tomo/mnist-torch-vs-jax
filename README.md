# Pytorch vs Jax

## Comparison 1: Weight update speed

The winner of this comparison is Jax.

- Jax compiles functions at the first access. So, 1st step takes very long time.
- Jax is 1.5x faster than pytorch.

### Jax

- 1st step: 2743.198883 ms (2743198883 ns)
- 2nd step: 730.498320 ms (730498320 ns)
- avg from 3rd: 0.812424 ms (812424.6259867719 ns)

### Pytorch (without torch.compile)

- 1st step: 258.808069 ms (258808069 ns)
- 2nd step: 2.161421 ms (2161421 ns)
- avg from 3rd: 1.218887 ms (1218887.3912417325 ns)

## Comparison 2: Self dataloader implementation

- [*] is `torch` or `jax` (in dataset.py)

```python
for images, labels in get_batches_[*](
        train_images, train_labels, batch_size=32, ...
    ):
```

- Shuffled indecies generation of Jax is little slower than pytorch.
- Indexed access of Jax is very slower than pytorch.

### Jax

- for loop duration: 2768.744654 ms
- random permutation time: 0.633029 ms

### Pytorch

- for loop duration: 49.750045 ms
- random permutation time: 0.594493 ms
