"""
Implementation of Standard Batch Normalization and Support-Conditioned Zero-Preserving
Batch Normalization (SC-ZPBN) from scratch in pure NumPy and PyTorch.

Mathematical formulation:
- Standard BN centers all elements, turning zeros into non-zero offsets (-mu / sigma),
  destroying sparsity and distorting empirical statistics under high sparsity.
- Support-Conditioned Zero-Preserving BN (SC-ZPBN) conditions moments on the active support
  S = {i : x_i != 0}, preserves exact zero support, and applies Empirical Bayes
  shrinkage regularization for low sample counts.
"""

import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# PART 1: PURE NUMPY IMPLEMENTATION FROM SCRATCH (Forward & Exact Backward)
# ============================================================================

class StandardBatchNorm1d_NumPy:
    """
    Standard Batch Normalization (Ioffe & Szegedy, 2015) implemented from scratch in NumPy.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable affine parameters
        self.gamma = np.ones(num_features, dtype=np.float32)
        self.beta = np.zeros(num_features, dtype=np.float32)
        
        # Running statistics for inference
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)
        
        # Cache for backpropagation
        self.cache = None

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        X: shape (m, d) where m is batch size, d is feature dimension
        """
        m, d = X.shape
        if training:
            # Mini-batch mean and variance (pools all elements, including zeros)
            mu = np.mean(X, axis=0) # (d,)
            var = np.var(X, axis=0) # (d,)
            
            # Standardization
            std_inv = 1.0 / np.sqrt(var + self.eps) # (d,)
            X_hat = (X - mu) * std_inv # (m, d)
            
            # Affine transformation
            out = self.gamma * X_hat + self.beta # (m, d)
            
            # Update running statistics via exponential moving average
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            
            self.cache = (X, X_hat, mu, var, std_inv)
        else:
            std_inv = 1.0 / np.sqrt(self.running_var + self.eps)
            X_hat = (X - self.running_mean) * std_inv
            out = self.gamma * X_hat + self.beta
            
        return out

    def backward(self, dout: np.ndarray) -> tuple:
        """
        Analytical backpropagation for Standard Batch Normalization.
        dout: upstream gradient dL/dout of shape (m, d)
        Returns: (dX, dgamma, dbeta)
        """
        X, X_hat, mu, var, std_inv = self.cache
        m, d = dout.shape
        
        dgamma = np.sum(dout * X_hat, axis=0) # (d,)
        dbeta = np.sum(dout, axis=0)         # (d,)
        
        dX_hat = dout * self.gamma # (m, d)
        
        # dvar = sum(dX_hat * (X - mu) * -0.5 * (var + eps)^(-3/2))
        dvar = np.sum(dX_hat * (X - mu), axis=0) * (-0.5) * (std_inv ** 3) # (d,)
        
        # dmu = sum(dX_hat * -std_inv) + dvar * sum(-2 * (X - mu)) / m
        dmu = np.sum(dX_hat * (-std_inv), axis=0) + dvar * np.mean(-2.0 * (X - mu), axis=0) # (d,)
        
        # dX = dX_hat * std_inv + dvar * 2(X - mu)/m + dmu / m
        dX = dX_hat * std_inv + dvar * (2.0 * (X - mu) / m) + dmu / m
        
        return dX, dgamma, dbeta


class SupportConditionedSparseBatchNorm1d_NumPy:
    """
    Support-Conditioned Zero-Preserving Batch Normalization (SC-ZPBN) from scratch in NumPy.
    - Zero preservation: X == 0 remains exactly 0.
    - Support-conditioned moments: mu_S and var_S computed strictly on active entries {i : X_i != 0}.
    - Empirical Bayes shrinkage: regularizes variance toward running variance when support count is small.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, kappa: float = 5.0):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.kappa = kappa # Prior pseudo-count for shrinkage
        
        self.gamma = np.ones(num_features, dtype=np.float32)
        self.beta = np.zeros(num_features, dtype=np.float32)
        
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)
        
        self.cache = None

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        m, d = X.shape
        mask = (X != 0.0) # Active support indicator
        
        if training:
            # Active counts per feature
            n_S = np.sum(mask, axis=0).astype(np.float32) # (d,)
            
            # Avoid division by zero for completely inactive features in mini-batch
            n_S_safe = np.maximum(n_S, 1.0)
            
            # Conditional mean over active support: mu_S = sum_{i in S} X_i / |S|
            sum_X = np.sum(X, axis=0) # zeros contribute 0
            mu_S = sum_X / n_S_safe
            
            # Conditional variance: var_S = sum_{i in S} (X_i - mu_S)^2 / |S|
            # (X - mu_S)^2 only for active entries
            sq_diff = np.where(mask, (X - mu_S) ** 2, 0.0)
            var_S = np.sum(sq_diff, axis=0) / n_S_safe
            
            # Empirical Bayes / James-Stein shrinkage toward running variance
            # lambda_weight = n_S / (n_S + kappa)
            shrinkage_lambda = n_S / (n_S + self.kappa)
            tilde_var = shrinkage_lambda * var_S + (1.0 - shrinkage_lambda) * self.running_var
            
            # Handle degenerate case where n_S == 0
            tilde_var = np.where(n_S > 0, tilde_var, self.running_var)
            mu_S = np.where(n_S > 0, mu_S, self.running_mean)
            
            std_inv = 1.0 / np.sqrt(tilde_var + self.eps)
            
            # Standardize ONLY active elements
            X_hat = np.where(mask, (X - mu_S) * std_inv, 0.0)
            
            # Zero-preserving affine: apply gamma to standardized active entries,
            # and beta conditioned on active support
            out = np.where(mask, self.gamma * X_hat + self.beta, 0.0)
            
            # Update running stats on active support features
            active_features = (n_S > 0)
            self.running_mean[active_features] = (1 - self.momentum) * self.running_mean[active_features] + self.momentum * mu_S[active_features]
            self.running_var[active_features] = (1 - self.momentum) * self.running_var[active_features] + self.momentum * tilde_var[active_features]
            
            self.cache = (X, mask, X_hat, mu_S, tilde_var, std_inv, n_S, shrinkage_lambda)
        else:
            std_inv = 1.0 / np.sqrt(self.running_var + self.eps)
            X_hat = np.where(mask, (X - self.running_mean) * std_inv, 0.0)
            out = np.where(mask, self.gamma * X_hat + self.beta, 0.0)
            
        return out

    def backward(self, dout: np.ndarray) -> tuple:
        """
        Exact analytical backpropagation for SC-ZPBN on sparse support.
        Gradient dX is zero on inactive support, preserving sparsity in the backward pass!
        """
        X, mask, X_hat, mu_S, tilde_var, std_inv, n_S, shrinkage_lambda = self.cache
        m, d = dout.shape
        
        # Filter dout by active support (inactive entries don't contribute)
        dout_active = np.where(mask, dout, 0.0)
        
        dgamma = np.sum(dout_active * X_hat, axis=0)
        dbeta = np.sum(dout_active, axis=0)
        
        dX_hat = dout_active * self.gamma
        
        n_S_safe = np.maximum(n_S, 1.0)
        
        # dtilde_var = sum_{i in S} dX_hat * (X - mu_S) * -0.5 * (tilde_var + eps)^(-3/2)
        dtilde_var = np.sum(dX_hat * np.where(mask, X - mu_S, 0.0), axis=0) * (-0.5) * (std_inv ** 3)
        
        # Chain rule through shrinkage: dvar_S = dtilde_var * shrinkage_lambda
        dvar_S = dtilde_var * shrinkage_lambda
        
        # dmu_S = sum_{i in S} [ dX_hat * (-std_inv) + dvar_S * (-2 * (X - mu_S) / n_S) ]
        dmu_S = np.sum(np.where(mask, dX_hat * (-std_inv), 0.0), axis=0) + \
                dvar_S * np.sum(np.where(mask, -2.0 * (X - mu_S) / n_S_safe, 0.0), axis=0)
        
        # dX = dX_hat * std_inv + dvar_S * (2 * (X - mu_S) / n_S) + dmu_S / n_S (strictly on mask)
        dX_active = dX_hat * std_inv + \
                    dvar_S * (2.0 * (X - mu_S) / n_S_safe) + \
                    (dmu_S / n_S_safe)
        
        dX = np.where(mask, dX_active, 0.0)
        
        return dX, dgamma, dbeta


# ============================================================================
# PART 2: PYTORCH IMPLEMENTATION WITH CUSTOM AUTOGRAD AND MODULES
# ============================================================================

class StandardBatchNorm1d_Torch(nn.Module):
    """
    Standard Batch Normalization module in PyTorch for direct baseline comparison.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


class SupportConditionedSparseBatchNorm1d_Torch(nn.Module):
    """
    Support-Conditioned Zero-Preserving Batch Normalization (SC-ZPBN) in PyTorch.
    Guarantees:
    1. Zero preservation: input zeros remain exact output zeros.
    2. Support conditioning: empirical moments computed only on non-zero activations.
    3. Shrinkage regularized variance: eliminates gradient explosion on rare features.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, kappa: float = 5.0):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.kappa = kappa
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (m, d)
        m, d = x.shape
        mask = (x != 0.0) # (m, d) boolean support mask
        
        if self.training:
            n_S = mask.sum(dim=0).float() # (d,)
            n_S_safe = torch.clamp(n_S, min=1.0)
            
            # Conditional mean over active support
            mu_S = x.sum(dim=0) / n_S_safe # (d,)
            
            # Conditional variance over active support
            diff = torch.where(mask, x - mu_S.unsqueeze(0), torch.zeros_like(x))
            var_S = (diff ** 2).sum(dim=0) / n_S_safe # (d,)
            
            # Empirical Bayes shrinkage
            lam = n_S / (n_S + self.kappa) # (d,)
            tilde_var = lam * var_S + (1.0 - lam) * self.running_var
            
            # Fallback if no active elements in the batch
            tilde_var = torch.where(n_S > 0, tilde_var, self.running_var)
            mu_S = torch.where(n_S > 0, mu_S, self.running_mean)
            
            # Standardize active support
            std_inv = 1.0 / torch.sqrt(tilde_var + self.eps) # (d,)
            x_hat = torch.where(mask, diff * std_inv.unsqueeze(0), torch.zeros_like(x))
            
            # Zero-preserving affine transformation
            out = torch.where(mask, self.gamma.unsqueeze(0) * x_hat + self.beta.unsqueeze(0), torch.zeros_like(x))
            
            # Update running stats
            with torch.no_grad():
                active = (n_S > 0)
                self.running_mean[active] = (1 - self.momentum) * self.running_mean[active] + self.momentum * mu_S[active]
                self.running_var[active] = (1 - self.momentum) * self.running_var[active] + self.momentum * tilde_var[active]
                
            return out
        else:
            # Inference mode
            std_inv = 1.0 / torch.sqrt(self.running_var + self.eps)
            diff = torch.where(mask, x - self.running_mean.unsqueeze(0), torch.zeros_like(x))
            x_hat = torch.where(mask, diff * std_inv.unsqueeze(0), torch.zeros_like(x))
            out = torch.where(mask, self.gamma.unsqueeze(0) * x_hat + self.beta.unsqueeze(0), torch.zeros_like(x))
            return out
