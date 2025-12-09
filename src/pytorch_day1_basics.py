import torch

# Detect GPU (MPS for Apple Silicon)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

# -----------------------
# Task 1: Create tensors
# -----------------------
A = torch.tensor([1., 2., 3.], device=device)
B = torch.tensor([4., 5., 6.], device=device)
C = torch.tensor([7., 8., 9.], device=device)

print("Tensor A:", A)
print("Tensor B:", B)
print("Tensor C:", C)

# -----------------------
# Task 2: Basic operations
# -----------------------
D = A * B
print("\nTensor D (A * B):", D)

E = C * A
print("Tensor E (C * A):", E)

F = C * C
print("Tensor F (C * C):", F)

# -----------------------
# Task 3: Reshaping
# -----------------------
G = torch.arange(0, 12, device=device)
print("\nTensor G:", G)

print("Reshaped G (3,4):\n", G.reshape(3, 4))
print("Reshaped G (4,3):\n", G.reshape(4, 3))
print("Flattened G:", G.flatten())

# -----------------------
# Optional Task 4: test GPU
# -----------------------
if device == "mps":
    X = torch.ones((2,2), device=device)
    Y = torch.ones((2,2), device=device)
    print("\nGPU Matrix Multiply (MPS):", X @ Y)
else:
    print("\nGPU (MPS) not available.")