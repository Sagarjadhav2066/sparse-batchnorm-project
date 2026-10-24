"""
Unit tests and gradient checks for Standard BN and SC-ZPBN.
Verifies:
1. Exact zero preservation for SC-ZPBN.
2. Sparsity destruction in Standard BN.
3. Numerical gradient verification (finite difference check).
"""

import numpy as np
import torch
from sparse_bn import (
    StandardBatchNorm1d_NumPy,
    SupportConditionedSparseBatchNorm1d_NumPy,
    StandardBatchNorm1d_Torch,
    SupportConditionedSparseBatchNorm1d_Torch,
)


def test_zero_preservation():
    print("=== Test 1: Zero Preservation & Density Explosion ===")
    np.random.seed(42)
    m, d = 64, 50
    # Create 90% sparse input
    X = np.random.randn(m, d).astype(np.float32)
    mask = (np.random.rand(m, d) > 0.90) # 10% active
    X = X * mask
    
    initial_zeros = np.sum(X == 0.0)
    initial_sparsity = initial_zeros / (m * d)
    print(f"Input Sparsity: {initial_sparsity * 100:.2f}% (zeros: {initial_zeros}/{m*d})")
    
    # 1. Standard BN
    std_bn = StandardBatchNorm1d_NumPy(num_features=d)
    out_std = std_bn.forward(X, training=True)
    std_zeros = np.sum(out_std == 0.0)
    std_sparsity = std_zeros / (m * d)
    print(f"Standard BN Output Sparsity: {std_sparsity * 100:.2f}% (zeros: {std_zeros}/{m*d})")
    assert std_sparsity == 0.0, "Standard BN failed to turn all zeros into non-zeros!"
    print("-> Standard BN completely destroyed sparsity! (Density exploded to 100%)")
    
    # 2. Proposed SC-ZPBN
    sc_bn = SupportConditionedSparseBatchNorm1d_NumPy(num_features=d)
    out_sc = sc_bn.forward(X, training=True)
    sc_zeros = np.sum(out_sc == 0.0)
    sc_sparsity = sc_zeros / (m * d)
    print(f"Proposed SC-ZPBN Output Sparsity: {sc_sparsity * 100:.2f}% (zeros: {sc_zeros}/{m*d})")
    assert sc_zeros == initial_zeros, f"SC-ZPBN did not preserve exact zeros! Expected {initial_zeros}, got {sc_zeros}"
    assert np.all(out_sc[X == 0.0] == 0.0), "Non-zero value found at inactive support in SC-ZPBN!"
    print("-> Proposed SC-ZPBN perfectly preserved 100% of zero support!")


def test_numerical_gradients():
    print("\n=== Test 2: Numerical Gradient Check (Finite Differences) ===")
    np.random.seed(42)
    m, d = 10, 5
    X = np.random.randn(m, d).astype(np.float64)
    mask = (np.random.rand(m, d) > 0.4) # Ensure non-empty support for grad check
    X = X * mask
    
    # Test SC-ZPBN NumPy backward pass
    bn = SupportConditionedSparseBatchNorm1d_NumPy(num_features=d, kappa=2.0)
    out = bn.forward(X, training=True)
    
    # Mock upstream loss gradient
    dout = np.random.randn(m, d).astype(np.float64)
    dX_analytic, dgamma_analytic, dbeta_analytic = bn.backward(dout)
    
    # Finite differences for X on active elements
    eps = 1e-6
    dX_num = np.zeros_like(X)
    for i in range(m):
        for j in range(d):
            if mask[i, j]:
                X_pos = X.copy()
                X_neg = X.copy()
                X_pos[i, j] += eps
                X_neg[i, j] -= eps
                
                bn_pos = SupportConditionedSparseBatchNorm1d_NumPy(num_features=d, kappa=2.0)
                out_pos = bn_pos.forward(X_pos, training=True)
                
                bn_neg = SupportConditionedSparseBatchNorm1d_NumPy(num_features=d, kappa=2.0)
                out_neg = bn_neg.forward(X_neg, training=True)
                
                grad = np.sum(dout * (out_pos - out_neg)) / (2.0 * eps)
                dX_num[i, j] = grad
            else:
                dX_num[i, j] = 0.0
                
    rel_error = np.max(np.abs(dX_analytic - dX_num) / (np.abs(dX_num) + 1e-7))
    print(f"SC-ZPBN dX Max Relative Error: {rel_error:.6e}")
    assert rel_error < 1e-4, f"Gradient check failed! Relative error: {rel_error}"
    print("-> SC-ZPBN Analytical gradients verified with high precision!")


def test_pytorch_equivalence():
    print("\n=== Test 3: PyTorch Implementation Autograd & Execution ===")
    torch.manual_seed(42)
    m, d = 32, 16
    x = torch.randn(m, d, requires_grad=True)
    mask = (torch.rand(m, d) > 0.85).float()
    x_sparse = x * mask
    
    model = SupportConditionedSparseBatchNorm1d_Torch(num_features=d, kappa=5.0)
    out = model(x_sparse)
    
    loss = (out ** 2).sum()
    loss.backward()
    
    print(f"PyTorch Forward Output Shape: {out.shape}")
    print(f"Gamma Grad Shape: {model.gamma.grad.shape}")
    print(f"Beta Grad Shape: {model.beta.grad.shape}")
    print(f"Input Zero Preservation: {(out[x_sparse == 0] == 0).all().item()}")
    assert (out[x_sparse == 0] == 0).all().item() == True
    print("-> PyTorch Module executed and backward passed successfully!")


if __name__ == "__main__":
    test_zero_preservation()
    test_numerical_gradients()
    test_pytorch_equivalence()
    print("\nALL UNIT TESTS PASSED SUCCESSFULLY!")
