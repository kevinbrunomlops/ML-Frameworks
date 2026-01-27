# Task 1: Vector and matrix basics (NumPy)
# TODO: Create two vectors (length 3) and compute:
# - dot product
# - L2 norm
# - cosine similarity
# TODO: Create a 2x3 matrix and multiply it by a length-3 vector

import numpy as np 

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