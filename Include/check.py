import numpy as np

a = np.load("/Users/venkat/Documents/Sign2Text(RAG)/landmarks/48. Hello/MVI_0029.npy")
b = np.load("/Users/venkat/Documents/Sign2Text(RAG)/preview_demo/48. Hello/MVI_0029.npy")

print(a.shape, b.shape)
print("Same numbers:", np.allclose(a, b))