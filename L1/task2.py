# Task 2: Eager vs graph execution
# TODO: Write a small function f(x) = x^3 + 2x
# TODO: Implement f(x) in ONE of:
# - PyTorch (eager)
# - TensorFlow with @tf.function (graph)
# - JAX with @jit (graph-like)
# TODO: Print the output and note how execution differs

import torch 


def f(x):
    return x**3 + 2*x

x = torch.tensor(3.0)
y = f(x)

print("Input:", x)
print("Output:", y)