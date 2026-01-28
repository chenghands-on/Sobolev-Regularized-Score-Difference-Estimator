import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# ============ Global Settings ============
torch.manual_seed(0); np.random.seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============ Method B: Rotated Single-Ridge Data ============
# In (t,s) space: p has t=±a, q has t=±b; s ~ N(0, cov2) identical for both
a, b = 0.5, 1.0               # Separation of p/q in ridge direction (b>a)
cov1, cov2 = 0.1, 0.1         # Variance in t,s directions (same for both classes)
theta = np.pi/6.0             # Angle between single ridge direction and x1 axis (30°)

R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])         # (t,s) -> (x1,x2)

# Covariance after rotation: Sigma = R diag(cov1, cov2) R^T
Sigma_np = R @ np.diag([cov1, cov2]) @ R.T
Sigma = torch.tensor(Sigma_np, dtype=torch.float, device=device)
Sigma_inv = torch.linalg.inv(Sigma)

def sample_ts_mix(center_abs, n):
    """
    Sample n points from symmetric mixture of {+center_abs, -center_abs} centers in (t,s) space,
    t ~ N(±center_abs, cov1), s ~ N(0, cov2), then rotate to x space.
    """
    signs = np.random.choice([-1.0, +1.0], size=(n,1))
    t = np.random.normal(loc=signs * center_abs, scale=np.sqrt(cov1), size=(n,1))
    s = np.random.normal(loc=0.0, scale=np.sqrt(cov2), size=(n,1))
    ts = np.concatenate([t, s], axis=1)                   # [n,2] in (t,s)
    x = ts @ R.T                                          # Transform to (x1,x2)
    return x

# Training set
num_per_mode = 1000
p_x = np.concatenate([sample_ts_mix(+a, num_per_mode),
                      sample_ts_mix(+a, num_per_mode)], axis=0)  # Equivalent to ±a mixture
q_x = np.concatenate([sample_ts_mix(+b, num_per_mode),
                      sample_ts_mix(+b, num_per_mode)], axis=0)

x = np.concatenate([p_x, q_x], axis=0)
y = np.concatenate([np.zeros((2*num_per_mode,1)), np.ones((2*num_per_mode,1))], axis=0)

perm = np.random.permutation(len(x))
x = torch.tensor(x[perm], dtype=torch.float, device=device)
y = torch.tensor(y[perm], dtype=torch.float, device=device)

dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

# Test set
def make_test(n=4000):
    p_te = np.concatenate([sample_ts_mix(+a, n//2),
                           sample_ts_mix(+a, n//2)], axis=0)
    q_te = np.concatenate([sample_ts_mix(+b, n//2),
                           sample_ts_mix(+b, n//2)], axis=0)
    X = np.concatenate([p_te, q_te], axis=0)
    return torch.tensor(X, dtype=torch.float, device=device)

Xtest = make_test(4000)

# Means after rotation (in x space)
mean_t_pos = np.array([+1.0, 0.0])   # ±1 ridge direction basis vector in (t,s) space
mean_t_neg = np.array([-1.0, 0.0])
mu_a_pos = (a * mean_t_pos) @ R.T    # t=+a -> x
mu_a_neg = (a * mean_t_neg) @ R.T    # t=-a -> x
mu_b_pos = (b * mean_t_pos) @ R.T
mu_b_neg = (b * mean_t_neg) @ R.T

means_p = torch.tensor(np.stack([mu_a_pos, mu_a_neg], axis=0), dtype=torch.float, device=device)  # [2,2]
means_q = torch.tensor(np.stack([mu_b_pos, mu_b_neg], axis=0), dtype=torch.float, device=device)  # [2,2]

# ============ Analytical True Gradient ∇log(q/p) ============
def grad_log_mixture(X, means):  # X:[B,2], means:[K,2]
    # N_i(x) ∝ exp(-0.5 (x-mu)^T Sigma^{-1} (x-mu)), weight gamma_i normalized by unnormalized terms
    Xm = X[:, None, :] - means[None, :, :]                               # [B,K,2]
    exp_term = torch.exp(-0.5 * torch.einsum('bki,ij,bkj->bk', Xm, Sigma_inv, Xm))  # [B,K]
    weights = exp_term / (exp_term.sum(dim=1, keepdim=True) + 1e-12)                 # [B,K]
    grad = torch.einsum('bk,ij,bkj->bi', weights, Sigma_inv, (means[None,:,:]-X[:,None,:]))
    return grad

def true_grad_log_ratio(X):
    return grad_log_mixture(X, means_q) - grad_log_mixture(X, means_p)

G_true = true_grad_log_ratio(Xtest).detach()

# ============ Method A: Baseline Classifier (directly learn logit) ============
class Classifier(nn.Module):
    def __init__(self, d=2, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, 1)  # Logit = log q/p
        )
    def forward(self, x): return self.net(x)

def train_baseline(epochs=300):
    model = Classifier().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward(); opt.step()
    return model

def grad_from_model(model, X):
    X = X.clone().detach().requires_grad_(True)
    f = model(X)                       # [B,1]
    g = torch.autograd.grad(f.sum(), X)[0]
    return g.detach()

baseline = train_baseline(epochs=300)
G_base = grad_from_model(baseline, Xtest)
mse_base = torch.mean((G_base - G_true)**2).item()
print(f"[Baseline] Grad-MSE vs truth: {mse_base:.6f}")

# ============ Method B: U-Bottleneck (k=1) + Optional Sobolev + Alternating U Update ============
class UGuidance(nn.Module):
    def __init__(self, d=2, k=1, h=64):
        super().__init__()
        self.U = nn.Parameter(torch.randn(d, k))       # Learn single ridge direction
        self.g = nn.Sequential(
            nn.Linear(k, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, 1)
        )
    def forward_with_latent(self, x, detach_U=False):
        U = self.U.detach() if detach_U else self.U
        z = x @ U          # [B,1]
        f = self.g(z)      # [B,1]  - Directly used as logit for log-ratio
        return f, z, U
    def forward(self, x, detach_U=False):
        f, _, _ = self.forward_with_latent(x, detach_U)
        return f

def sobolev_penalty(model: UGuidance, xb, detach_U: bool):
    xb = xb.clone().detach().requires_grad_(True)
    f, z, U = model.forward_with_latent(xb, detach_U=detach_U)
    grad_z = torch.autograd.grad(f.sum(), z, create_graph=True)[0]     # [B,1]
    grad_x = grad_z @ U.T                                              # [B,2]
    return (grad_x**2).sum(dim=1).mean()

def sobolev_perp_penalty(model: UGuidance, xb, detach_U: bool):
    xb = xb.clone().detach().requires_grad_(True)
    f, z, U = model.forward_with_latent(xb, detach_U=detach_U)  # U:[d,k]
    grad_z = torch.autograd.grad(f.sum(), z, create_graph=True)[0]   # [B,k]
    grad_x = grad_z @ U.T                                            # [B,d]
    # Project onto orthogonal complement of U
    if U.shape[1] == 1:
        u = U / (U.norm(dim=0, keepdim=True) + 1e-12)               # [d,1]
        proj_perp = grad_x - (grad_x @ u) @ u.T                     # [B,d]
    else:
        UtU_inv = torch.linalg.inv(U.T @ U)
        P_perp = torch.eye(U.shape[0], device=U.device) - U @ UtU_inv @ U.T
        proj_perp = grad_x @ P_perp.T
    return (proj_perp**2).sum(dim=1).mean()


def orthonormalize_U_(U: torch.Tensor):
    with torch.no_grad():
        Q = U / (U.norm(dim=0, keepdim=True) + 1e-12)
        U.copy_(Q)

def train_U_sobolev(epochs=300, lambda_grad=1e-4, u_update_every=2, qr_every=50, beta_orth=1e-3):
    model = UGuidance().to(device)
    # Only main network parameters go into opt_main; U has its own optimizer
    main_params = [p for n,p in model.named_parameters() if n != 'U']
    opt_main = optim.Adam(main_params, lr=1e-3)
    opt_U    = optim.Adam([model.U], lr=2e-3)
    bce = nn.BCEWithLogitsLoss()

    step = 0
    for ep in range(epochs):
        for xb, yb in loader:
            step += 1
            update_U = (step % u_update_every == 0)

            # When not updating U, use detach_U=True for forward/regularization to avoid building graph for U
            logits = model.forward(xb, detach_U=not update_U)
            loss_ce = bce(logits, yb)

            sob = sobolev_penalty(model, xb, detach_U=not update_U)

            # Soft orthogonal regularization: only compute and backprop when updating U; else set to zero to avoid unnecessary graph/accumulation
            if update_U:
                I = torch.eye(model.U.shape[1], device=device)
                orth_pen = ((model.U.T @ model.U - I)**2).sum()
            else:
                orth_pen = torch.tensor(0.0, device=device)

            # Sobolev annealing (first 100 steps from 0 to lambda_grad)
            lam0, lam1, T = 0.0, lambda_grad, 100
            lam_t = lam0 + (lam1 - lam0) * min(step / T, 1.0)

            loss = loss_ce + lam_t * sob + beta_orth * orth_pen

            # Optimization
            opt_main.zero_grad(set_to_none=True)
            if update_U:
                opt_U.zero_grad(set_to_none=True)
            loss.backward()
            opt_main.step()
            if update_U:
                opt_U.step()
                # Periodically normalize columns, more stable inside no_grad
                if step % qr_every == 0:
                    with torch.no_grad():
                        orthonormalize_U_(model.U.data)
    return model


# You can set lambda_grad=0 as a control group "without Sobolev"
u_model = train_U_sobolev(epochs=300, lambda_grad=1e-4, u_update_every=10, qr_every=50)

def grad_from_U(model, X):
    X = X.clone().detach().requires_grad_(True)
    f, z, U = model.forward_with_latent(X, detach_U=True)  # Don't differentiate w.r.t. U
    grad_z = torch.autograd.grad(f.sum(), z)[0]            # [B,1]
    grad_x = grad_z @ U.T
    return grad_x.detach()

G_U = grad_from_U(u_model, Xtest)
mse_U = torch.mean((G_U - G_true)**2).item()
print(f"[U+Sobolev(k=1, rotated ridge)] Grad-MSE vs truth: {mse_U:.6f}")

improve = (mse_base - mse_U) / (mse_base + 1e-12)
print(f"Relative improvement (MSE drop): {improve*100:.2f}%")

# ============ Method C: Local Linear (Kernel-based gradient estimation) ============
# Note: In gradest2 convention, p is target and q is source, estimating ∇log(p/q)
# In simulation.py, p is source and q is target, estimating ∇log(q/p)
# So we swap p_x and q_x to get the correct gradient direction

def rbfkernel(x, y, sigma):
    """RBF kernel: k(x,y) = exp(-||x-y||^2 / (2*sigma^2))"""
    x = x.view(x.shape[0], -1)
    y = y.view(y.shape[0], -1)
    t1 = torch.tile(torch.sum(x**2, dim=1, keepdim=True), (1, y.shape[0]))
    t2 = -2 * torch.matmul(x, y.T)
    t3 = torch.tile(torch.sum(y**2, dim=1, keepdim=True).T, (x.shape[0], 1))
    return torch.exp(-(t1 + t2 + t3) / (2 * sigma**2))

psicon = lambda d: d**2/2 + d  # ψ(d) = d²/2 + d

def obj_local_linear(W, b, xp, xq, psicon, kpx=None, kqx=None):
    """Objective function for Local Linear method"""
    A = torch.mean(kpx * (torch.matmul(W, xp.T) + b), 1, keepdim=True)
    B = torch.mean(kqx * psicon((torch.matmul(W, xq.T) + b)), 1, keepdim=True)
    return torch.mean(-A + B, 0)

class NPnet(nn.Module):
    """Parameter matrix network for Local Linear method"""
    def __init__(self, n, m):
        super().__init__()
        self.Wb = nn.Parameter(torch.zeros(n, m+1))  # [n_samples, d+1]
    
    def forward(self, x):
        return self.Wb  # Directly return parameter matrix

def train_local_linear(xp, xq, x, sigma, epochs=500, batch_size=256, lr=1e-1):
    """
    Train Local Linear gradient estimator.
    
    Args:
        xp: target domain samples (q in simulation.py convention)
        xq: source domain samples (p in simulation.py convention)
        x: points where gradient is estimated
        sigma: kernel bandwidth
        epochs: training epochs
        batch_size: batch size
        lr: learning rate
    
    Returns:
        trained network
    """
    d = xp.shape[1]
    n = x.shape[0]
    
    # Compute kernel matrices
    kpx = rbfkernel(x, xp, sigma)  # [n, n_p]
    kqx = rbfkernel(x, xq, sigma)  # [n, n_q]
    
    # Create data loader
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xp, xq, kpx.T, kqx.T),
        batch_size=batch_size, shuffle=True
    )
    
    # Initialize network
    net = NPnet(n, d).to(device)
    optimizer = optim.Adagrad(net.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        for xpi, xqi, kpxi, kqxi in train_loader:
            kpxi = kpxi.T  # [n, batch_size]
            kqxi = kqxi.T  # [n, batch_size]
            
            optimizer.zero_grad()
            
            # Forward pass
            predicted = net(x)  # [n, d+1]
            Wx = predicted[:, :d]  # [n, d] - gradient estimate
            bx = predicted[:, d:]   # [n, 1] - bias
            
            # Compute objective
            output = obj_local_linear(Wx, bx, xpi, xqi, psicon, kpxi, kqxi)
            
            # Backward and optimize
            output.backward()
            optimizer.step()
    
    return net

def grad_from_local_linear(net, X):
    """Extract gradient from trained Local Linear network"""
    predicted = net(X)  # [B, d+1]
    grad = predicted[:, :X.shape[1]]  # [B, d] - only gradient part, no bias
    return grad.detach()

# Prepare data for Local Linear
# Note: In simulation.py, p_x is source, q_x is target
# Local Linear estimates ∇log(p/q) when p is target and q is source
# So we swap: use q_x as xp (target) and p_x as xq (source)
# This gives us ∇log(q/p) which is what we need
p_x_torch = torch.tensor(p_x, dtype=torch.float, device=device)
q_x_torch = torch.tensor(q_x, dtype=torch.float, device=device)

# Compute sigma (median distance)
def comp_median(X):
    """Compute median pairwise distance"""
    dists = torch.cdist(X, X)
    return torch.median(dists[dists > 0])

sigma_ll = comp_median(torch.cat([p_x_torch, q_x_torch], dim=0)).item()

# Train Local Linear model
print("\n[Local Linear] Training...")
ll_net = train_local_linear(
    xp=q_x_torch,  # target (q in simulation.py)
    xq=p_x_torch,  # source (p in simulation.py)
    x=Xtest,       # points where gradient is estimated
    sigma=sigma_ll,
    epochs=500,
    batch_size=256,
    lr=1e-1
)

# Extract gradient
G_ll = grad_from_local_linear(ll_net, Xtest)
mse_ll = torch.mean((G_ll - G_true)**2).item()
print(f"[Local Linear] Grad-MSE vs truth: {mse_ll:.6f}")

# ============ Summary Comparison ============
print("\n" + "="*60)
print("SUMMARY: Gradient Estimation MSE Comparison")
print("="*60)
print(f"1. Baseline (Classification):        {mse_base:.6f}")
print(f"2. U+Sobolev (k=1):                  {mse_U:.6f}")
print(f"3. Local Linear (Kernel-based):      {mse_ll:.6f}")
print("="*60)

# Find best method
methods = {
    "Baseline": mse_base,
    "U+Sobolev": mse_U,
    "Local Linear": mse_ll
}
best_method = min(methods, key=methods.get)
best_mse = methods[best_method]
print(f"\nBest method: {best_method} (MSE: {best_mse:.6f})")

# Relative improvements
improve_ll_vs_base = (mse_base - mse_ll) / (mse_base + 1e-12) * 100
improve_ll_vs_U = (mse_U - mse_ll) / (mse_U + 1e-12) * 100
print(f"\nLocal Linear vs Baseline: {improve_ll_vs_base:+.2f}% improvement")
print(f"Local Linear vs U+Sobolev: {improve_ll_vs_U:+.2f}% improvement")
