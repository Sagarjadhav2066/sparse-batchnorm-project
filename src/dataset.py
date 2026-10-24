"""
Sparse Dataset Generator and Loader.
Generates realistic high-dimensional sparse classification datasets adhering to
a Two-Part (Hurdle) Zipfian distribution:
1. Feature presence probability p_j follows a power-law / Zipf distribution.
2. Signal magnitude given presence X_{i,j} | (X_{i,j} != 0) follows a log-normal distribution.
3. Multi-class target generated via latent sparse linear weights with noise.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


def generate_synthetic_sparse_dataset(
    n_samples: int = 6000,
    n_features: int = 300,
    n_classes: int = 5,
    base_sparsity: float = 0.95,
    random_state: int = 42
):
    """
    Generates a high-dimensional sparse classification dataset.
    
    Parameters:
    - n_samples: Total number of observations (default: 6,000)
    - n_features: Feature dimensionality (default: 300)
    - n_classes: Number of target classes (default: 5)
    - base_sparsity: Target average sparsity level (default: 0.95, i.e., 95% zeros)
    - random_state: Seed for reproducibility
    """
    rng = np.random.RandomState(random_state)
    
    # 1. Feature presence probability follows a power-law / Zipf distribution
    ranks = np.arange(1, n_features + 1)
    # Power-law probabilities: p_j proportional to 1 / rank^alpha
    alpha = 0.7
    raw_probs = 1.0 / (ranks ** alpha)
    # Scale probabilities so average presence is (1 - base_sparsity)
    target_active_prob = 1.0 - base_sparsity
    probs = raw_probs * (target_active_prob / np.mean(raw_probs))
    probs = np.clip(probs, 0.005, 0.5) # Feature active rate between 0.5% and 50%
    
    # 2. Sample binary support indicators S_{i,j} ~ Bernoulli(p_j)
    S = rng.binomial(1, probs, size=(n_samples, n_features)).astype(np.float32)
    
    # 3. For active features, sample continuous magnitude from LogNormal (mimicking TF-IDF / count frequencies)
    magnitudes = rng.lognormal(mean=0.5, sigma=0.8, size=(n_samples, n_features)).astype(np.float32)
    X = S * magnitudes
    
    # 4. Generate sparse ground-truth latent weights W of shape (n_features, n_classes)
    # Only 20% of features are truly predictive (sparse ground-truth)
    W_mask = rng.binomial(1, 0.20, size=(n_features, n_classes))
    W = rng.randn(n_features, n_classes).astype(np.float32) * W_mask
    
    # 5. Logits = X @ W + noise
    logits = X @ W + rng.randn(n_samples, n_classes).astype(np.float32) * 0.1
    probs_classes = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs_classes /= np.sum(probs_classes, axis=1, keepdims=True)
    y = np.argmax(probs_classes, axis=1).astype(np.int64)
    
    actual_sparsity = np.mean(X == 0.0)
    return X, y, actual_sparsity


def get_dataloaders(
    n_samples: int = 6000,
    n_features: int = 300,
    n_classes: int = 5,
    base_sparsity: float = 0.95,
    batch_size: int = 64,
    random_state: int = 42
):
    X, y, actual_sparsity = generate_synthetic_sparse_dataset(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        base_sparsity=base_sparsity,
        random_state=random_state
    )
    
    # Train (70%), Val (15%), Test (15%) split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )
    
    # Convert to PyTorch Tensors
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, actual_sparsity
