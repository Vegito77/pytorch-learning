import torch

print("PyTorch version:", torch.__version__)

# Check MPS availability (Apple Silicon GPU)
mps_available = torch.backends.mps.is_available()
mps_built = torch.backends.mps.is_built()

print("MPS built:", mps_built)
print("MPS available:", mps_available)

if mps_available:
    device = torch.device("mps")
    print("Using device:", device)

    # Perform a simple test operation on MPS
    a = torch.randn(2, 2, device=device)
    b = torch.randn(2, 2, device=device)
    c = a @ b

    print("Matrix multiply successful on device:", c.device)
    print(c)
else:
    print("MPS not available — using CPU instead.")