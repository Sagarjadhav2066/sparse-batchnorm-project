"""
Deep Neural Network Architectures for Sparse Data Evaluation.
Supports:
1. No Normalization ('none')
2. Standard Batch Normalization ('standard_bn')
3. Proposed Support-Conditioned Zero-Preserving Batch Normalization ('proposed_sc_zpbn')
"""

import torch
import torch.nn as nn
from sparse_bn import StandardBatchNorm1d_Torch, SupportConditionedSparseBatchNorm1d_Torch


class SparseClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 300,
        hidden_dim: int = 128,
        num_classes: int = 5,
        norm_type: str = 'proposed_sc_zpbn',
        kappa: float = 5.0
    ):
        super().__init__()
        self.norm_type = norm_type
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        
        if norm_type == 'none':
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
        elif norm_type == 'standard_bn':
            self.bn1 = StandardBatchNorm1d_Torch(hidden_dim)
            self.bn2 = StandardBatchNorm1d_Torch(hidden_dim // 2)
        elif norm_type == 'proposed_sc_zpbn':
            self.bn1 = SupportConditionedSparseBatchNorm1d_Torch(hidden_dim, kappa=kappa)
            self.bn2 = SupportConditionedSparseBatchNorm1d_Torch(hidden_dim // 2, kappa=kappa)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer 1
        h1 = self.fc1(x)
        h1_norm = self.bn1(h1)
        a1 = self.relu1(h1_norm)
        
        # Layer 2
        h2 = self.fc2(a1)
        h2_norm = self.bn2(h2)
        a2 = self.relu2(h2_norm)
        
        # Output Logits
        logits = self.fc3(a2)
        return logits

    def get_layer1_sparsity_ratio(self, x: torch.Tensor) -> float:
        """
        Computes the ratio of non-zero entries before and after the first normalization layer.
        """
        with torch.no_grad():
            h1 = self.fc1(x)
            h1_norm = self.bn1(h1)
            zeros_before = (h1 == 0).float().mean().item()
            zeros_after = (h1_norm == 0).float().mean().item()
            return zeros_before, zeros_after
