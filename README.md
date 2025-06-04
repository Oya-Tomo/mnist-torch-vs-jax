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
