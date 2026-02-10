# Task 1: Vector and matrix basics (NumPy)
# TODO: Create two vectors (length 3) and compute:
# - dot product
# - L2 norm
# - cosine similarity
# TODO: Create a 2x3 matrix and multiply it by a length-3 vector

import numpy as np 
import torch
import time

##  ----- Vectors ------
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

dot_product = np.dot(v1, v2)

# L2-norm 
norm_v1 = np.linalg.norm(v1)
norm_v2 = np.linalg.norm(v2)

# Cosine similarity
cosine_similarity = dot_product / (norm_v1 * norm_v2)

print("Dot product:", dot_product)
print("L2 norm v1:", norm_v1)
print("L2 norm v2:", norm_v2)
print("Cosine similarity:", cosine_similarity)

# --- Matris ---
M = np.array([
    [1,2,3],
    [4,5,6]
])

result = M @ v1
print("Matrix-vector product:", result)



def f(x):
    return x**3 + 2*x

print("Vi testar funktionen med 7, svaret är:", f(7))

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(device)

print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version PyTorch was built with:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())

x = torch.tensor(7.0)
y = f(x)
print(y)

# 1. Eager Execution (Standard)
x = torch.randn(10000, device="cuda" if torch.cuda.is_available() else "cpu")
# Warm up
start_time = time.time()
_ = f(x)
eager_time = time.time() - start_time
print(f"Eager execution time: {eager_time} seconds")


# This 'traces' the function and optimizes the kernels
compiled_f = torch.compile(f)
# First call triggers the compilation (slower), subsequent calls are fast
start_time = time.time()
_ = compiled_f(x)
graph_time = time.time() - start_time
print(f"Graph execution time: {graph_time} seconds")