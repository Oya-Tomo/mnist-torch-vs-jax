# Pytorch vs Jax

## Comparison 1

- Jax: mnist example from official flax quickstart.
- Pytorch: translated Jax example with Claude Sonnet 4.

> 💡 Fix some error manualy & changed optimizer from SGD to Adam

### Result

- Jax: 16.942138698 [sec]
- Pytorch: 41.440016595 [sec]

### Discussion

- This result show that Jax mnist example is faster than pytorch mnist.
- But, this example uses Dataloader provided by pytorch. We can improve this point ...
