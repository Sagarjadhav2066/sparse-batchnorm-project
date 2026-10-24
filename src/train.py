"""
Comprehensive Training and Evaluation Pipeline.
Trains Baseline (No BN), Standard BN, and Proposed SC-ZPBN on sparse datasets.
Logs:
1. Training and Validation Loss curves
2. Validation Accuracy and Macro-F1 curves
3. Gradient norm stability and variance
4. Sparsity retention and non-zero fill-in across varying sparsity regimes (70% - 98%)
"""

import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

from dataset import get_dataloaders, generate_synthetic_sparse_dataset
from model import SparseClassifier


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    grad_norms = []
    
    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        
        logits = model(batch_X)
        loss = criterion(logits, batch_y)
        loss.backward()
        
        # Track gradient norms across mini-batches to analyze stability
        total_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_grad_norm += param_norm ** 2
        total_grad_norm = total_grad_norm ** 0.5
        grad_norms.append(total_grad_norm)
        
        optimizer.step()
        total_loss += loss.item() * batch_X.size(0)
        
    avg_loss = total_loss / len(loader.dataset)
    grad_norm_mean = float(np.mean(grad_norms))
    grad_norm_var = float(np.var(grad_norms))
    return avg_loss, grad_norm_mean, grad_norm_var


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        logits = model(batch_X)
        loss = criterion(logits, batch_y)
        
        total_loss += loss.item() * batch_X.size(0)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(batch_y.cpu().numpy())
        
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    return avg_loss, acc, f1


def run_training_experiment(epochs: int = 35, lr: float = 0.01, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    
    print("=" * 70)
    print("EXPERIMENT 1: Training Convergence and Stability (Base Sparsity = 95%)")
    print("=" * 70)
    
    train_loader, val_loader, test_loader, sparsity = get_dataloaders(
        n_samples=6000, n_features=300, n_classes=5, base_sparsity=0.95, batch_size=64, random_state=seed
    )
    print(f"Dataset Loaded. Feature Dimensionality: 300, Realized Sparsity: {sparsity * 100:.2f}%")
    
    strategies = ['none', 'standard_bn', 'proposed_sc_zpbn']
    results = {}
    
    criterion = nn.CrossEntropyLoss()
    
    for strat in strategies:
        print(f"\n--- Training Strategy: {strat.upper()} ---")
        model = SparseClassifier(input_dim=300, hidden_dim=128, num_classes=5, norm_type=strat).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'grad_norm_mean': [],
            'grad_norm_var': []
        }
        
        start_time = time.time()
        for epoch in range(1, epochs + 1):
            train_loss, g_mean, g_var = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            history['grad_norm_mean'].append(g_mean)
            history['grad_norm_var'].append(g_var)
            
            if epoch % 5 == 0 or epoch == epochs:
                print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1:.4f} | Grad Var: {g_var:.4f}")
                
        elapsed = time.time() - start_time
        test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)
        print(f"Final Test Accuracy: {test_acc*100:.2f}% | Test F1: {test_f1:.4f} | Time: {elapsed:.2f}s")
        
        results[strat] = {
            'history': history,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'elapsed_sec': elapsed
        }
        
    return results


def run_sparsity_sensitivity_experiment(seed: int = 42):
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Sensitivity across Sparsity Regimes (70% to 98% Zeros)")
    print("=" * 70)
    
    device = torch.device("cpu")
    sparsity_levels = [0.70, 0.80, 0.90, 0.95, 0.98]
    strategies = ['none', 'standard_bn', 'proposed_sc_zpbn']
    
    sensitivity_results = {strat: {'sparsity': [], 'test_acc': [], 'density_explosion': [], 'grad_var': []} for strat in strategies}
    
    for sp in sparsity_levels:
        print(f"\n>> Evaluating at Sparsity Level: {sp * 100:.1f}%")
        train_loader, val_loader, test_loader, actual_sp = get_dataloaders(
            n_samples=5000, n_features=300, n_classes=5, base_sparsity=sp, batch_size=64, random_state=seed
        )
        
        # Test sample input to measure density fill-in
        sample_batch = next(iter(test_loader))[0] # (64, 300)
        
        for strat in strategies:
            torch.manual_seed(seed)
            model = SparseClassifier(input_dim=300, hidden_dim=128, num_classes=5, norm_type=strat).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()
            
            # Train for 20 epochs
            all_g_vars = []
            for ep in range(20):
                _, _, g_var = train_epoch(model, train_loader, criterion, optimizer, device)
                all_g_vars.append(g_var)
                
            _, test_acc, _ = evaluate(model, test_loader, criterion, device)
            
            # Measure output density (non-zero fill-in) after first layer normalization
            with torch.no_grad():
                h1 = model.fc1(sample_batch)
                h1_norm = model.bn1(h1)
                non_zero_pct = (h1_norm != 0).float().mean().item() * 100.0
                
            sensitivity_results[strat]['sparsity'].append(actual_sp * 100.0)
            sensitivity_results[strat]['test_acc'].append(test_acc * 100.0)
            sensitivity_results[strat]['density_explosion'].append(non_zero_pct)
            sensitivity_results[strat]['grad_var'].append(float(np.mean(all_g_vars)))
            
            print(f"  [{strat.upper()}] Acc: {test_acc*100:.2f}% | Output Density: {non_zero_pct:.2f}% | Avg Grad Var: {np.mean(all_g_vars):.4f}")
            
    return sensitivity_results


if __name__ == "__main__":
    t0 = time.time()
    exp1_results = run_training_experiment(epochs=35, lr=0.01, seed=42)
    exp2_results = run_sparsity_sensitivity_experiment(seed=42)
    
    combined_data = {
        'convergence_experiment': exp1_results,
        'sensitivity_experiment': exp2_results
    }
    
    out_path = "/home/sagar-jadhav/.gemini/antigravity/scratch/sparse_batchnorm_project/results.json"
    with open(out_path, 'w') as f:
        json.dump(combined_data, f, indent=2)
        
    print(f"\nAll experiments complete in {time.time() - t0:.2f}s! Results saved to {out_path}")
