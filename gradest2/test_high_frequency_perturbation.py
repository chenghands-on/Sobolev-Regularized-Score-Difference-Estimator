"""
Test high-frequency oscillatory perturbation example
This demonstrates why Sobolev regularization is needed:
Even if log density ratio is estimated accurately, its gradient can diverge.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from numpy import mean

# Import from gradest2
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wgf import gradest_classification, grad_from_classifier, sobolev_penalty

# Use the same Classifier structure as simulation.py
class Classifier(nn.Module):
    def __init__(self, d=1, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, 1)  # Logit = log q/p
        )
    def forward(self, x): 
        return self.net(x)

def grad_from_model(model, X):
    """Extract gradient from model (same as simulation.py)"""
    X = X.clone().detach().requires_grad_(True)
    f = model(X)  # [B, 1]
    g = torch.autograd.grad(f.sum(), X)[0]  # [B, d]
    return g.detach()

def sobolev_penalty_standard(model, xb):
    """Sobolev penalty for standard Classifier (same as log_classifier_compare.py)"""
    xb = xb.clone().detach().requires_grad_(True)
    f = model(xb)  # [B, 1]
    grad_f = torch.autograd.grad(f.sum(), xb, create_graph=True)[0]  # [B, d]
    return (grad_f**2).sum(dim=1).mean()

# ============ Global Settings ============
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ============ Core Modification: Signal + High Frequency Noise ============

# 1. Define the underlying smooth ground truth (Signal)
def get_smooth_signal(x_np):
    """This is a non-zero, beautiful low-frequency curve"""
    return 1.5 * np.sin(x_np)

def get_smooth_gradient(x_np):
    """Gradient of the underlying ground truth"""
    return 1.5 * np.cos(x_np)

# 2. Define superimposed high-frequency noise (Perturbation)
def get_perturbation(x_np, freq=12, amp=0.3):
    """Frequency freq should not be too high to avoid underfitting, nor too low to be mistaken for a signal
    Amplitude amp should be moderate"""
    return amp * np.sin(freq * x_np)

def get_perturbation_grad(x_np, freq=12, amp=0.3):
    """Gradient of noise - derivative is amplified by frequency!"""
    return amp * freq * np.cos(freq * x_np)

# 3. Combination: "Dirty" distribution used for data generation
def true_log_ratio_noisy(x, freq, amp):
    """This is the distribution actually seen by the model (including noise)"""
    x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
    return torch.tensor(get_smooth_signal(x_np) + get_perturbation(x_np, freq, amp), 
                       dtype=torch.float32, device=device)

# ============ Data Sampling Logic ============
# p(x) is still Uniform
def sample_p(n):
    """Sample from source distribution p(x) = 1/(2π) on [0, 2π]"""
    return np.random.uniform(0, 2*np.pi, size=(n, 1))

# q(x) is sampled based on noisy log ratio
def sample_q_noisy(n, freq, amp):
    """q(x) \propto p(x) * exp(f(x))
    Using Rejection Sampling"""
    x_samples = []
    # Estimate maximum weight for rejection sampling
    x_test = np.linspace(0, 2*np.pi, 1000)
    log_r = get_smooth_signal(x_test) + get_perturbation(x_test, freq, amp)
    max_weight = np.exp(np.max(log_r)) * 1.2  # Give some margin
    
    max_iterations = n * 20
    iterations = 0
    
    while len(x_samples) < n and iterations < max_iterations:
        iterations += 1
        x_cand = np.random.uniform(0, 2*np.pi)
        # Calculate log ratio
        val = get_smooth_signal(x_cand) + get_perturbation(x_cand, freq, amp)
        weight = np.exp(val)
        
        if np.random.rand() < weight / max_weight:
            x_samples.append(x_cand)
    
    # If sampling is insufficient, fill with uniform distribution
    while len(x_samples) < n:
        x_samples.append(np.random.uniform(0, 2*np.pi))
    
    return np.array(x_samples[:n]).reshape(-1, 1)

# ============ Extreme Test Parameters: L2 Converges but H1 Diverges ============
# Goal: Left plot three lines merge, right plot blue line explodes

freq_param = 30       # High frequency
amp_param = 0.05      # Low amplitude (invisible to the naked eye)

num_train = 1000      # Sampling theorem: requires extremely high density sampling points
num_test = 1000
classifier_hidden = 256 # Powerful network
nepochs = 1000        # Sufficient overfitting

lambda_sobolev = 1e-3 # Extremely weak regularization, only kills high frequencies, does not harm low frequencies

print("\n" + "="*80)
print("Testing 'Invisible' Noise: Signal(1.5) + Noise({amp_param}*sin({freq_param}x))")
print("="*80)
print(f"Signal: 1.5 * sin(x)")
print(f"Noise: {amp_param} * sin({freq_param}*x)")
print(f"Noise Gradient Amplitude approx: {amp_param * freq_param:.2f}")
print(f"Training samples: {num_train}")
print(f"Test samples: {num_test}")
print(f"Classifier hidden: {classifier_hidden}")
print(f"Training epochs: {nepochs}")
print(f"Sobolev lambda: {lambda_sobolev}")

# ============ Generate Data ============
print("\nGenerating training data...")
# Training data comes from "Noisy" distribution
p_train = sample_p(num_train)
q_train = sample_q_noisy(num_train, freq_param, amp_param)

p_x_torch = torch.tensor(p_train, dtype=torch.float32, device=device)
q_x_torch = torch.tensor(q_train, dtype=torch.float32, device=device)

# Prepare training data: xp (target q) -> label 1, xq (source p) -> label 0
# Same structure as simulation.py
X_all = torch.cat([p_x_torch, q_x_torch], dim=0)
y_all = torch.cat([
    torch.zeros(p_x_torch.shape[0], 1, device=device, dtype=torch.float32),  # source p = 0
    torch.ones(q_x_torch.shape[0], 1, device=device, dtype=torch.float32)   # target q = 1
], dim=0)

# Shuffle (same as simulation.py)
perm = torch.randperm(len(X_all))
X_all = X_all[perm]
y_all = y_all[perm]

dataset = TensorDataset(X_all, y_all)
train_loader = DataLoader(dataset, batch_size=256, shuffle=True)

# ============ Key Modification: Evaluation Metric ============
# Our generated test points
print("Generating test data...")
X_test_np = np.linspace(0, 2*np.pi, num_test).reshape(-1, 1)
X_test = torch.tensor(X_test_np, dtype=torch.float32, device=device)

# *** Core point: Metric compares against Smooth Truth ***
# We assume the high-frequency oscillation is annoying noise, and our goal is to recover the smooth signal
G_target_smooth = torch.tensor(get_smooth_gradient(X_test_np), device=device, dtype=torch.float32)
Log_target_smooth = torch.tensor(get_smooth_signal(X_test_np), device=device, dtype=torch.float32)

# ============ Method 1: Classification (Baseline) ============
print("\n" + "="*80)
print("Method 1: Classification (Baseline, no Sobolev)")
print("="*80)

classifier_base = Classifier(d=1, h=classifier_hidden).to(device)
optimizer_base = optim.Adam(classifier_base.parameters(), lr=1e-3)
scheduler_base = optim.lr_scheduler.StepLR(optimizer_base, step_size=2000, gamma=0.5)

print("Training baseline classifier...")
# Use the same training approach as simulation.py
loss_fn = nn.BCEWithLogitsLoss()
for ep in range(nepochs):
    for xb, yb in train_loader:
        optimizer_base.zero_grad()
        logits = classifier_base(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer_base.step()
    scheduler_base.step()  # Learning rate decay, helps to fine-tune details in the final stage

# When calculating error, compare model output vs Smooth Target
G_base = grad_from_model(classifier_base, X_test)
mse_base = torch.mean((G_base - G_target_smooth)**2).item()
error_base = torch.norm(G_base - G_target_smooth).item()
log_ratio_base = classifier_base(X_test)
log_ratio_error = torch.mean((log_ratio_base - Log_target_smooth)**2).item()

print(f"\nBaseline Results:")
print(f"  Log ratio MSE: {log_ratio_error:.6f}")
print(f"  Gradient MSE: {mse_base:.6f}")
print(f"  Gradient L2 error: {error_base:.6f}")

# ============ Method 2: Classification + Sobolev ============
print("\n" + "="*80)
print("Method 2: Classification + Sobolev Regularization")
print("="*80)

classifier_sob = Classifier(d=1, h=classifier_hidden).to(device)
optimizer_sob = optim.Adam(classifier_sob.parameters(), lr=1e-3)
scheduler_sob = optim.lr_scheduler.StepLR(optimizer_sob, step_size=2000, gamma=0.5)

print("Training classifier with Sobolev regularization...")
# Use the same training approach as log_classifier_compare.py train_baseline_sobolev
loss_fn = nn.BCEWithLogitsLoss()
step = 0
for ep in range(nepochs):
    for xb, yb in train_loader:
        step += 1
        logits = classifier_sob(xb)
        loss_ce = loss_fn(logits, yb)
        sob = sobolev_penalty_standard(classifier_sob, xb)
        # Sobolev annealing: lambda from 0 to lambda_sobolev over 100 steps
        lam0, lam1, T = 0.0, lambda_sobolev, 100
        lam_t = lam0 + (lam1 - lam0) * min(step / T, 1.0)
        loss = loss_ce + lam_t * sob
        optimizer_sob.zero_grad()
        loss.backward()
        optimizer_sob.step()
    scheduler_sob.step()  # Learning rate decay

# When calculating error, compare model output vs Smooth Target
G_sob = grad_from_model(classifier_sob, X_test)
mse_sob = torch.mean((G_sob - G_target_smooth)**2).item()
error_sob = torch.norm(G_sob - G_target_smooth).item()
log_ratio_sob = classifier_sob(X_test)
log_ratio_error_sob = torch.mean((log_ratio_sob - Log_target_smooth)**2).item()

print(f"\nSobolev Results:")
print(f"  Log ratio MSE: {log_ratio_error_sob:.6f}")
print(f"  Gradient MSE: {mse_sob:.6f}")
print(f"  Gradient L2 error: {error_sob:.6f}")

# ============ Comparison ============
print("\n" + "="*80)
print("COMPARISON")
print("="*80)
print(f"{'Metric':<30} {'Baseline':<20} {'Sobolev':<20} {'Improvement':<20}")
print("-" * 90)
improvement_lr = ((log_ratio_error - log_ratio_error_sob)/log_ratio_error*100) if log_ratio_error > 0 else 0
improvement_grad = ((mse_base - mse_sob)/mse_base*100) if mse_base > 0 else 0
improvement_l2 = ((error_base - error_sob)/error_base*100) if error_base > 0 else 0
print(f"{'Log Ratio MSE':<30} {log_ratio_error:<20.6f} {log_ratio_error_sob:<20.6f} {improvement_lr:<20.2f}%")
print(f"{'Gradient MSE':<30} {mse_base:<20.6f} {mse_sob:<20.6f} {improvement_grad:<20.2f}%")
print(f"{'Gradient L2 Error':<30} {error_base:<20.6f} {error_sob:<20.6f} {improvement_l2:<20.2f}%")

# ============ Fix Visualization: Calculate Normalization Constant Z ============
print("\nGenerating visualization...")
print("Computing normalizing constant Z...")

# Estimate Z using Monte Carlo integration
# Z = E_p[exp(f(x))]
# We generate a large number of p(x) samples to calculate this expectation
mc_samples = 100000
x_mc = np.random.uniform(0, 2*np.pi, size=mc_samples)

# Note: Z should be based on the distribution actually used to generate the data (including noise)
# But to draw the "ideal smooth target", we are mainly concerned with the offset caused by the smooth signal
# Because the noise is high-frequency and has a mean of 0, its contribution to Z is relatively small; the main offset comes from sin(x)
exponent = get_smooth_signal(x_mc) + get_perturbation(x_mc, freq_param, amp_param)
Z_est = np.mean(np.exp(exponent))
log_Z = np.log(Z_est)

print(f"Estimated log Z: {log_Z:.4f}") 
# This value is approximately 0.2~0.3, which is exactly the vertical gap you see in the figure

# ============ Modify Plotting Code ============
# Create figure similar to the reference image: 2 subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Set global font size
plt.rcParams.update({'font.size': 14})

# Calculate model prediction
log_ratio_base_plot = classifier_base(X_test).detach().cpu().numpy().flatten()
log_ratio_sob_plot = classifier_sob(X_test).detach().cpu().numpy().flatten()
grad_base_plot = G_base.detach().cpu().numpy().flatten()
grad_sob_plot = G_sob.detach().cpu().numpy().flatten()

X_test_flat = X_test_np.flatten()

# Left Plot: Log Ratio
ax1 = axes[0]

# *** Correction point: True value must subtract log Z ***
target_smooth_normalized = get_smooth_signal(X_test_flat) - log_Z

# Plot ideal true value (corrected)
ax1.plot(X_test_flat, target_smooth_normalized, 'k-', linewidth=3, label='Target', alpha=0.9)

# Plot model (model doesn't need to be changed, it learned correctly)
ax1.plot(X_test_flat, log_ratio_base_plot, 'b-', label='Baseline', linewidth=1.5, alpha=0.8)
ax1.plot(X_test_flat, log_ratio_sob_plot, 'r-', label='Sobolev', linewidth=2, alpha=0.8)
ax1.set_xlabel('x', fontsize=18)
ax1.set_ylabel('log r(x)', fontsize=18)
ax1.set_title('Log Density Ratio Estimates', fontsize=20, fontweight='bold')
ax1.legend(loc='best', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.grid(True, alpha=0.3, linestyle=':')
ax1.set_xlim([0, 2*np.pi])

# Right Plot: Gradient
ax2 = axes[1]
# Plot ideal gradient (thick black line) - this is cos(x)
ax2.plot(X_test_flat, get_smooth_gradient(X_test_flat), 'k-', linewidth=3, label='Target Grad', alpha=0.9)
# Plot model
ax2.plot(X_test_flat, grad_base_plot, 'b-', label='Baseline', linewidth=1, alpha=0.8)
ax2.plot(X_test_flat, grad_sob_plot, 'r-', label='Sobolev', linewidth=2, alpha=0.8)
ax2.set_xlabel('x', fontsize=18)
ax2.set_ylabel('∇ log r(x)', fontsize=18)
ax2.set_title('Gradient Estimates', fontsize=20, fontweight='bold')
ax2.legend(loc='best', fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.grid(True, alpha=0.3, linestyle=':')
ax2.set_xlim([0, 2*np.pi])

plt.tight_layout()
save_path = 'high_frequency_perturbation_comparison.png'
plt.savefig(save_path, dpi=200, bbox_inches='tight')
print(f"Visualization saved to: {save_path}")

print("\n" + "="*80)
print("Test completed!")
print("="*80)
