import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import csv
import os

# ============ Global Settings ============
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============ Create Output Directory ============
output_dir = 'sim_exp_result'
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}/")

# ============ Distribution Type List ============
DISTRIBUTION_TYPES = ['rotated_ridge', 'orthogonal_gmm', 'bounded']

# Store all results
results = []

# ============ Distribution Generation Functions ============
def setup_distribution(dist_type, seed=0):
    """
    Setup distribution type, return sampling functions, means, covariances, etc.
    
    Returns:
        sample_p: Function to sample from source distribution (n) -> [n, 2]
        sample_q: Function to sample from target distribution (n) -> [n, 2]
        means_p_np: Mean of source distribution [K, 2]
        means_q_np: Mean of target distribution [K, 2]
        Sigma_np: Covariance matrix [2, 2]
        title: Distribution type title
        dist_params: Distribution parameter dictionary (for visualization)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if dist_type == 'rotated_ridge':
        # In (t,s) space: p has t=±a, q has t=±b; s ~ N(0, cov2) identical for both
        a, b = 0.5, 1.0
        cov1, cov2 = 0.1, 0.1
        theta = np.pi/6.0
        
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        
        Sigma_np = R @ np.diag([cov1, cov2]) @ R.T
        Sigma = torch.tensor(Sigma_np, dtype=torch.float, device=device)
        Sigma_inv = torch.linalg.inv(Sigma)
        
        def sample_mixture(center_abs, n):
            signs = np.random.choice([-1.0, +1.0], size=(n,1))
            t = np.random.normal(loc=signs * center_abs, scale=np.sqrt(cov1), size=(n,1))
            s = np.random.normal(loc=0.0, scale=np.sqrt(cov2), size=(n,1))
            ts = np.concatenate([t, s], axis=1)
            x = ts @ R.T
            return x
        
        def sample_p(n):
            return np.concatenate([sample_mixture(a, n//2), sample_mixture(a, n//2)], axis=0)
        
        def sample_q(n):
            return np.concatenate([sample_mixture(b, n//2), sample_mixture(b, n//2)], axis=0)
        
        mean_t_pos = np.array([+1.0, 0.0])
        mean_t_neg = np.array([-1.0, 0.0])
        mu_a_pos = (a * mean_t_pos) @ R.T
        mu_a_neg = (a * mean_t_neg) @ R.T
        mu_b_pos = (b * mean_t_pos) @ R.T
        mu_b_neg = (b * mean_t_neg) @ R.T
        
        means_p_np = np.stack([mu_a_pos, mu_a_neg], axis=0)
        means_q_np = np.stack([mu_b_pos, mu_b_neg], axis=0)
        
        title = f"Rotated Ridge (θ={np.degrees(theta):.1f}°)"
        dist_params = {'a': a, 'b': b, 'theta': theta}
        
        return sample_p, sample_q, means_p_np, means_q_np, Sigma_np, Sigma_inv, title, dist_params
    
    elif dist_type == 'orthogonal_gmm':
        mu_S = np.array([0.5, 0.5])
        mu_T = np.array([0.5, -0.5])
        sigma_sq = 0.1
        sigma = np.sqrt(sigma_sq)
        
        Sigma_np = sigma_sq * np.eye(2)
        Sigma = torch.tensor(Sigma_np, dtype=torch.float, device=device)
        Sigma_inv = torch.linalg.inv(Sigma)
        
        def sample_mixture(mu, n):
            signs = np.random.choice([-1.0, +1.0], size=(n,))
            means = signs[:, None] * mu[None, :]
            x = np.random.normal(loc=means, scale=sigma, size=(n, 2))
            return x
        
        def sample_p(n):
            return np.concatenate([sample_mixture(mu_S, n//2), sample_mixture(mu_S, n//2)], axis=0)
        
        def sample_q(n):
            return np.concatenate([sample_mixture(mu_T, n//2), sample_mixture(mu_T, n//2)], axis=0)
        
        means_p_np = np.stack([mu_S, -mu_S], axis=0)
        means_q_np = np.stack([mu_T, -mu_T], axis=0)
        
        title = "Orthogonal GMM (μSᵀμT = 0)"
        dist_params = {'mu_S': mu_S, 'mu_T': mu_T, 'sigma_sq': sigma_sq}
        
        return sample_p, sample_q, means_p_np, means_q_np, Sigma_np, Sigma_inv, title, dist_params
    
    elif dist_type == 'bounded':
        # Bounded source to bounded target
        # Source: Uniform distribution on [-1, 1]^2, p(x) = 1/4
        # Target: q(x)/p(x) = c * (1 + b * cos(πx₁) * cos(πx₂))
        a = 5  # ratio range parameter: q/p ∈ [1/a, a]
        r_min, r_max = 1 / a, a
        c = 0.5 * (r_max + r_min)
        b = (r_max - r_min) / (r_max + r_min)
        
        # For uniform distribution, covariance matrix not needed, but we create identity matrix for compatibility
        Sigma_np = np.eye(2) * 0.01  # Small dummy covariance, only for visualization
        Sigma = torch.tensor(Sigma_np, dtype=torch.float, device=device)
        Sigma_inv = torch.linalg.inv(Sigma)
        
        def generate_source_cosine(n):
            """Generate uniform source samples on [-1, 1]^2"""
            return np.random.uniform(-1, 1, size=(n, 2))
        
        def generate_target_cosine(n):
            """Generate target samples using rejection sampling"""
            # Use PyTorch for rejection sampling
            N_prop = int(n * 3)  # oversample factor
            x = torch.rand(N_prop, 2, device=device) * 2 - 1  # [-1, 1]^2
            ratio = c * (1 + b * torch.cos(torch.pi * x[:, 0]) * torch.cos(torch.pi * x[:, 1]))
            accept_prob = ratio / ratio.max()
            u = torch.rand_like(accept_prob)
            accepted = x[u < accept_prob].cpu().numpy()
            
            # If accepted samples are insufficient, recursive call (increase oversample factor)
            if len(accepted) < n:
                # Increase oversample factor and retry
                N_prop = int(n * 5)
                x = torch.rand(N_prop, 2, device=device) * 2 - 1
                ratio = c * (1 + b * torch.cos(torch.pi * x[:, 0]) * torch.cos(torch.pi * x[:, 1]))
                accept_prob = ratio / ratio.max()
                u = torch.rand_like(accept_prob)
                accepted = x[u < accept_prob].cpu().numpy()
            
            if len(accepted) < n:
                # If still insufficient, return all accepted samples (may be less than n)
                return accepted[:n] if len(accepted) > 0 else generate_source_cosine(n)
            
            return accepted[:n]
        
        def sample_p(n):
            return generate_source_cosine(n)
        
        def sample_q(n):
            return generate_target_cosine(n)
        
        # For uniform distribution, we use dummy means for visualization (uniform distribution has no mean)
        # But for compatibility with visualization functions, we use center and corners of [-1, 1]^2
        means_p_np = np.array([[-0.5, -0.5], [0.5, 0.5]])  # Dummy values, only for visualization
        means_q_np = np.array([[0.0, 0.0], [0.0, 0.0]])  # Dummy values, only for visualization
        
        title = f"Bounded Source → Bounded Target (a={a})"
        dist_params = {'a': a, 'c': c, 'b': b}
        
        return sample_p, sample_q, means_p_np, means_q_np, Sigma_np, Sigma_inv, title, dist_params
    
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")

# ============ Visualization Functions ============
def plot_distributions(sample_p, sample_q, means_p_np, means_q_np, Sigma_np, title, save_path=None, n_samples_plot=1000, grid_resolution=100, dist_type=None, dist_params=None):
    """Plot source and target distributions (without overlay)"""
    np.random.seed(42)
    p_samples = sample_p(n_samples_plot)
    q_samples = sample_q(n_samples_plot)
    
    # Create grid
    x_min = min(p_samples[:, 0].min(), q_samples[:, 0].min()) - 0.5
    x_max = max(p_samples[:, 0].max(), q_samples[:, 0].max()) + 0.5
    y_min = min(p_samples[:, 1].min(), q_samples[:, 1].min()) - 0.5
    y_max = max(p_samples[:, 1].max(), q_samples[:, 1].max()) + 0.5
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_resolution),
                         np.linspace(y_min, y_max, grid_resolution))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Calculate density
    if dist_type == 'bounded':
        # Source: Uniform distribution, density is constant 1/4
        density_p = np.ones(grid_points.shape[0]) * 0.25
        
        # Target: q(x) = (c/4) * (1 + b * cos(πx₁) * cos(πx₂))
        a = dist_params['a']
        r_min, r_max = 1 / a, a
        c = 0.5 * (r_max + r_min)
        b = (r_max - r_min) / (r_max + r_min)
        
        x1 = grid_points[:, 0]
        x2 = grid_points[:, 1]
        # Only has values within [-1, 1]^2, zero elsewhere
        mask = (x1 >= -1) & (x1 <= 1) & (x2 >= -1) & (x2 <= 1)
        density_q = np.zeros(grid_points.shape[0])
        density_q[mask] = (c / 4.0) * (1 + b * np.cos(np.pi * x1[mask]) * np.cos(np.pi * x2[mask]))
        density_q = np.maximum(density_q, 0)  # Ensure non-negative
    else:
        # 高斯混合分布
        def mixture_density(points, means, cov):
            density = np.zeros(points.shape[0])
            for mean in means:
                rv = multivariate_normal(mean, cov)
                density += 0.5 * rv.pdf(points)
            return density
        
        density_p = mixture_density(grid_points, means_p_np, Sigma_np)
        density_q = mixture_density(grid_points, means_q_np, Sigma_np)
    
    density_p = density_p.reshape(xx.shape)
    density_q = density_q.reshape(xx.shape)
    
    # Create figure (two subplots: source and target)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Source distribution
    ax1 = axes[0]
    ax1.scatter(p_samples[:, 0], p_samples[:, 1], alpha=0.4, s=10, c='blue', label='Source (p)')
    contour1 = ax1.contour(xx, yy, density_p, levels=8, colors='blue', alpha=0.6, linewidths=1.5)
    ax1.clabel(contour1, inline=True, fontsize=14)
    ax1.set_xlabel('x1', fontsize=16)
    ax1.set_ylabel('x2', fontsize=16)
    ax1.set_title(f'Source Distribution (p)\n{title}', fontsize=16, fontweight='bold')
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.legend(fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # Target distribution
    ax2 = axes[1]
    ax2.scatter(q_samples[:, 0], q_samples[:, 1], alpha=0.4, s=10, c='red', label='Target (q)')
    contour2 = ax2.contour(xx, yy, density_q, levels=8, colors='red', alpha=0.6, linewidths=1.5)
    ax2.clabel(contour2, inline=True, fontsize=14)
    ax2.set_xlabel('x1', fontsize=16)
    ax2.set_ylabel('x2', fontsize=16)
    ax2.set_title(f'Target Distribution (q)\n{title}', fontsize=16, fontweight='bold')
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.legend(fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Distribution plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

# ============ Model Definitions ============
class Classifier(nn.Module):
    def __init__(self, d=2, h=64, use_smooth_activation=False):
        super().__init__()
        if use_smooth_activation:
            # Use smooth activation function (SiLU/Swish) to obtain smooth gradients
            self.net = nn.Sequential(
                nn.Linear(d, h), nn.SiLU(),
                nn.Linear(h, h), nn.SiLU(),
                nn.Linear(h, 1)
            )
        else:
            # Use ReLU (keep original for 2D experiments)
            self.net = nn.Sequential(
                nn.Linear(d, h), nn.ReLU(),
                nn.Linear(h, h), nn.ReLU(),
                nn.Linear(h, 1)
            )
    def forward(self, x): return self.net(x)
    
    def forward_1d(self, x): 
        # For 1D input, ensure correct dimensions
        if x.dim() == 1:
            x = x.unsqueeze(1)
        return self.net(x)

class UGuidance(nn.Module):
    def __init__(self, d=2, k=1, h=64):
        super().__init__()
        self.U = nn.Parameter(torch.randn(d, k))
        self.g = nn.Sequential(
            nn.Linear(k, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, 1)
        )
    def forward_with_latent(self, x, detach_U=False):
        U = self.U.detach() if detach_U else self.U
        z = x @ U
        f = self.g(z)
        return f, z, U
    def forward(self, x, detach_U=False):
        f, _, _ = self.forward_with_latent(x, detach_U)
        return f

# ============ Training Functions ============
def train_baseline(loader, epochs=300, seed=0, d=2, use_smooth_activation=False):
    torch.manual_seed(seed)
    model = Classifier(d=d, use_smooth_activation=use_smooth_activation).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
    return model

def sobolev_penalty_standard(model, xb):
    xb = xb.clone().detach().requires_grad_(True)
    f = model(xb)
    grad_x = torch.autograd.grad(f.sum(), xb, create_graph=True)[0]
    return (grad_x**2).sum(dim=1).mean()

def train_baseline_sobolev(loader, epochs=300, lambda_grad=1e-3, seed=0, d=2, use_smooth_activation=False):
    torch.manual_seed(seed)
    model = Classifier(d=d, use_smooth_activation=use_smooth_activation).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    step = 0
    for ep in range(epochs):
        for xb, yb in loader:
            step += 1
            logits = model(xb)
            loss_ce = bce(logits, yb)
            sob = sobolev_penalty_standard(model, xb)
            # Lambda gradually decreases from 1e-3 to 5e-4
            lam0, lam1, T = 1e-3, 5e-4, 100
            lam_t = lam0 + (lam1 - lam0) * min(step / T, 1.0)
            loss = loss_ce + lam_t * sob
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model

def sobolev_perp_penalty(model, xb, detach_U):
    xb = xb.clone().detach().requires_grad_(True)
    f, z, U = model.forward_with_latent(xb, detach_U=detach_U)
    grad_z = torch.autograd.grad(f.sum(), z, create_graph=True)[0]
    grad_x = grad_z @ U.T
    if U.shape[1] == 1:
        u = U / (U.norm(dim=0, keepdim=True) + 1e-12)
        proj_perp = grad_x - (grad_x @ u) @ u.T
    else:
        UtU_inv = torch.linalg.inv(U.T @ U)
        P_perp = torch.eye(U.shape[0], device=U.device) - U @ UtU_inv @ U.T
        proj_perp = grad_x @ P_perp.T
    return (proj_perp**2).sum(dim=1).mean()

def orthonormalize_U_(U):
    with torch.no_grad():
        Q = U / (U.norm(dim=0, keepdim=True) + 1e-12)
        U.copy_(Q)

def train_U_sobolev(loader, epochs=300, lambda_grad=1e-4, u_update_every=10, qr_every=50, beta_orth=1e-3, seed=0):
    torch.manual_seed(seed)
    model = UGuidance().to(device)
    main_params = [p for n,p in model.named_parameters() if n != 'U']
    opt_main = optim.Adam(main_params, lr=1e-3)
    opt_U = optim.Adam([model.U], lr=2e-3)
    bce = nn.BCEWithLogitsLoss()
    step = 0
    for ep in range(epochs):
        for xb, yb in loader:
            step += 1
            update_U = (step % u_update_every == 0)
            logits = model.forward(xb, detach_U=not update_U)
            loss_ce = bce(logits, yb)
            sob = sobolev_perp_penalty(model, xb, detach_U=not update_U)
            if update_U:
                I = torch.eye(model.U.shape[1], device=device)
                orth_pen = ((model.U.T @ model.U - I)**2).sum()
            else:
                orth_pen = torch.tensor(0.0, device=device)
            lam0, lam1, T = 0.0, lambda_grad, 100
            lam_t = lam0 + (lam1 - lam0) * min(step / T, 1.0)
            loss = loss_ce + lam_t * sob + beta_orth * orth_pen
            opt_main.zero_grad(set_to_none=True)
            if update_U:
                opt_U.zero_grad(set_to_none=True)
            loss.backward()
            opt_main.step()
            if update_U:
                opt_U.step()
                if step % qr_every == 0:
                    with torch.no_grad():
                        orthonormalize_U_(model.U.data)
    return model

# ============ Evaluation Functions ============
def grad_from_model(model, X):
    X = X.clone().detach().requires_grad_(True)
    f = model(X)
    g = torch.autograd.grad(f.sum(), X)[0]
    return g.detach()

def grad_from_U(model, X):
    X = X.clone().detach().requires_grad_(True)
    f, z, U = model.forward_with_latent(X, detach_U=True)
    grad_z = torch.autograd.grad(f.sum(), z)[0]
    grad_x = grad_z @ U.T
    return grad_x.detach()

# ============ Local Linear (Kernel-based gradient estimation) ============
# Helper functions matching gradest2.core.util implementation
def comp_dist(x, y):
    """Compute squared Euclidean distance matrix between x and y"""
    x = x.view(x.shape[0], -1)
    y = y.view(y.shape[0], -1)
    t1 = torch.tile(torch.sum(x**2, dim=1, keepdim=True), (1, y.shape[0]))
    t2 = -2 * torch.matmul(x, y.T)
    t3 = torch.tile(torch.sum(y**2, dim=1, keepdim=True).T, (x.shape[0], 1))
    return t1 + t2 + t3

def kernel_comp(x, y, sigma):
    """RBF kernel: k(x,y) = exp(-||x-y||^2 / (2*sigma^2))"""
    x = x.view(x.shape[0], -1)
    y = y.view(y.shape[0], -1)
    t1 = torch.tile(torch.sum(x**2, dim=1, keepdim=True), (1, y.shape[0]))
    t2 = -2 * torch.matmul(x, y.T)
    t3 = torch.tile(torch.sum(y**2, dim=1, keepdim=True).T, (x.shape[0], 1))
    return torch.exp(-(t1 + t2 + t3) / (2 * sigma**2))

rbfkernel = kernel_comp  # Alias for consistency

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

# Remove comp_median - we'll compute sigma directly using comp_dist

def train_local_linear(xp, xq, x, sigma, epochs=500, batch_size=256, lr=1e-1, seed=0):
    """
    Train Local Linear gradient estimator.
    
    Args:
        xp: target domain samples (q in convention)
        xq: source domain samples (p in convention)
        x: points where gradient is estimated
        sigma: kernel bandwidth
        epochs: training epochs
        batch_size: batch size
        lr: learning rate
        seed: random seed
    
    Returns:
        trained network
    """
    torch.manual_seed(seed)
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

def compute_true_grad(X, means_p, means_q, Sigma_inv, dist_type=None, dist_params=None):
    """
    Compute true gradient ∇log(q/p)
    
    For bounded distribution:
    - p(x) = 1/4 (uniform distribution)
    - q(x)/p(x) = c * (1 + b * cos(πx₁) * cos(πx₂))
    - ∇log(q/p) = [-πb sin(πx₁) cos(πx₂) / (1 + b * cos(πx₁) * cos(πx₂)),
                    -πb cos(πx₁) sin(πx₂) / (1 + b * cos(πx₁) * cos(πx₂))]
    """
    if dist_type == 'bounded':
        # Analytical gradient formula
        a = dist_params['a']
        r_min, r_max = 1 / a, a
        c = 0.5 * (r_max + r_min)
        b = (r_max - r_min) / (r_max + r_min)
        
        x1 = X[:, 0]
        x2 = X[:, 1]
        cos_x1 = torch.cos(torch.pi * x1)
        cos_x2 = torch.cos(torch.pi * x2)
        sin_x1 = torch.sin(torch.pi * x1)
        sin_x2 = torch.sin(torch.pi * x2)
        
        denominator = 1 + b * cos_x1 * cos_x2
        grad_x1 = -torch.pi * b * sin_x1 * cos_x2 / (denominator + 1e-12)
        grad_x2 = -torch.pi * b * cos_x1 * sin_x2 / (denominator + 1e-12)
        
        return torch.stack([grad_x1, grad_x2], dim=1)
    else:
        # Gradient for Gaussian mixture distribution
        def grad_log_mixture(X, means):
            Xm = X[:, None, :] - means[None, :, :]
            exp_term = torch.exp(-0.5 * torch.einsum('bki,ij,bkj->bk', Xm, Sigma_inv, Xm))
            weights = exp_term / (exp_term.sum(dim=1, keepdim=True) + 1e-12)
            grad = torch.einsum('bk,ij,bkj->bi', weights, Sigma_inv, (means[None,:,:]-X[:,None,:]))
            return grad
        return grad_log_mixture(X, means_q) - grad_log_mixture(X, means_p)

def plot_error_vs_samples(n_list, errors_baseline, errors_sobolev, errors_local_linear,
                          std_errors_baseline=None, std_errors_sobolev=None, std_errors_local_linear=None, save_path=None):
    """Plot estimation error vs sample size (similar to Figure 9 right plot)"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    n_array = np.array(n_list)
    err_base = np.array(errors_baseline)
    err_sob = np.array(errors_sobolev)
    err_ll = np.array(errors_local_linear)
    
    # Plot error lines
    ax.plot(n_array, err_base, 'o-', color='orange', linewidth=2, markersize=8, label='Baseline')
    ax.plot(n_array, err_sob, 's-', color='yellow', linewidth=2, markersize=8, label='Baseline+Sobolev')
    ax.plot(n_array, err_ll, '^-', color='green', linewidth=2, markersize=8, label='Local Linear')
    
    # Plot error bars (if available)
    if std_errors_baseline is not None:
        std_base = np.array(std_errors_baseline)
        ax.errorbar(n_array, err_base, yerr=std_base, fmt='o', color='orange', 
                   capsize=5, capthick=2, alpha=0.6)
    if std_errors_sobolev is not None:
        std_sob = np.array(std_errors_sobolev)
        ax.errorbar(n_array, err_sob, yerr=std_sob, fmt='s', color='yellow', 
                   capsize=5, capthick=2, alpha=0.6)
    if std_errors_local_linear is not None:
        std_ll = np.array(std_errors_local_linear)
        ax.errorbar(n_array, err_ll, yerr=std_ll, fmt='^', color='green', 
                   capsize=5, capthick=2, alpha=0.6)
    
    ax.set_xlabel('n', fontsize=12)
    ax.set_ylabel('||∇ log r - ŵ||', fontsize=12)
    ax.set_title('Estimation error with standard error', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Error vs samples plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()

def plot_gradient_heatmap_2d(xx, yy, grad_true, grad_base, grad_sob, title, save_path=None):
    """Plot 2D gradient heatmaps (ground truth, Baseline, and Baseline+Sobolev)"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Compute gradient magnitude
    if torch.is_tensor(grad_true):
        grad_true_np = grad_true.cpu().numpy()
    else:
        grad_true_np = grad_true
    if torch.is_tensor(grad_base):
        grad_base_np = grad_base.cpu().numpy()
    else:
        grad_base_np = grad_base
    if torch.is_tensor(grad_sob):
        grad_sob_np = grad_sob.cpu().numpy()
    else:
        grad_sob_np = grad_sob
    
    # Compute gradient vector magnitude
    grad_true_mag = np.sqrt(grad_true_np[:, 0]**2 + grad_true_np[:, 1]**2).reshape(xx.shape)
    grad_base_mag = np.sqrt(grad_base_np[:, 0]**2 + grad_base_np[:, 1]**2).reshape(xx.shape)
    grad_sob_mag = np.sqrt(grad_sob_np[:, 0]**2 + grad_sob_np[:, 1]**2).reshape(xx.shape)
    
    # Use unified color range
    vmin = min(grad_true_mag.min(), grad_base_mag.min(), grad_sob_mag.min())
    vmax = max(grad_true_mag.max(), grad_base_mag.max(), grad_sob_mag.max())
    
    # Ground truth gradient heatmap
    im1 = axes[0].contourf(xx, yy, grad_true_mag, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_xlabel('x1', fontsize=12)
    axes[0].set_ylabel('x2', fontsize=12)
    axes[0].set_title(f'Ground Truth\n{title}', fontsize=13, fontweight='bold')
    axes[0].set_aspect('equal', adjustable='box')
    plt.colorbar(im1, ax=axes[0], label='||∇ log r||')
    
    # Baseline estimated gradient heatmap
    im2 = axes[1].contourf(xx, yy, grad_base_mag, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_xlabel('x1', fontsize=12)
    axes[1].set_ylabel('x2', fontsize=12)
    axes[1].set_title(f'Baseline\n{title}', fontsize=13, fontweight='bold')
    axes[1].set_aspect('equal', adjustable='box')
    plt.colorbar(im2, ax=axes[1], label='||∇ log r||')
    
    # Baseline+Sobolev estimated gradient heatmap
    im3 = axes[2].contourf(xx, yy, grad_sob_mag, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[2].set_xlabel('x1', fontsize=12)
    axes[2].set_ylabel('x2', fontsize=12)
    axes[2].set_title(f'Baseline+Sobolev\n{title}', fontsize=13, fontweight='bold')
    axes[2].set_aspect('equal', adjustable='box')
    plt.colorbar(im3, ax=axes[2], label='||∇ log r||')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Gradient heatmap saved to: {save_path}")
    else:
        plt.show()
    plt.close()

# ============ Main Test Loop ============
print(f"\n{'='*80}")
print("Testing three distribution types with three methods (Baseline, Baseline+Sobolev, Local Linear)")
print("Training set sizes: 10, 100, 1000 samples per mode (Total: 20, 200, 2000 samples)")
print(f"{'='*80}\n")

# Test different training set sizes (samples per distribution type, total samples is 2x)
NUM_PER_MODE_LIST = [10, 100, 1000]  # Corresponding to total samples: 20, 200, 2000

# Store error vs sample data for each distribution type (for plotting error vs sample graphs)
error_vs_sample_data = {dist_type: {'n_list': [], 'errors_baseline': [], 'errors_sobolev': [], 
                                     'errors_local_linear': [], 'std_errors_baseline': [], 
                                     'std_errors_sobolev': [], 'std_errors_local_linear': []} 
                        for dist_type in DISTRIBUTION_TYPES}

for num_per_mode in NUM_PER_MODE_LIST:
    print(f"\n{'='*80}")
    print(f"Training Set Size: {num_per_mode} samples per mode (Total: {2*num_per_mode} samples)")
    print(f"{'='*80}\n")
    
    for dist_type in DISTRIBUTION_TYPES:
        print(f"\n{'-'*80}")
        print(f"Distribution: {dist_type} | Training samples: {2*num_per_mode}")
        print(f"{'-'*80}\n")
        
        # Setup distribution (use different seed each time to ensure independence)
        base_seed = num_per_mode  # Use num_per_mode as seed base
        
        # Generate test data (fixed size, only need to generate once)
        sample_p_temp, sample_q_temp, means_p_np, means_q_np, Sigma_np, Sigma_inv, title, dist_params = setup_distribution(dist_type, seed=base_seed)
        Xtest = torch.tensor(np.concatenate([sample_p_temp(100), sample_q_temp(100)], axis=0), dtype=torch.float, device=device)
        
        means_p = torch.tensor(means_p_np, dtype=torch.float, device=device)
        means_q = torch.tensor(means_q_np, dtype=torch.float, device=device)
        Sigma = torch.tensor(Sigma_np, dtype=torch.float, device=device)
        Sigma_inv_torch = torch.linalg.inv(Sigma)
        
        # Compute ground truth gradient
        G_true = compute_true_grad(Xtest, means_p, means_q, Sigma_inv_torch, dist_type=dist_type, dist_params=dist_params).detach()
        
        # Visualization (only generate on first run, i.e., when num_per_mode=10)
        if num_per_mode == 10:
            plot_save_path = f'{output_dir}/distributions_{dist_type}.png'
            plot_distributions(sample_p_temp, sample_q_temp, means_p_np, means_q_np, Sigma_np, title, 
                              save_path=plot_save_path, dist_type=dist_type, dist_params=dist_params)
        
        # Repeat experiments to compute standard deviation
        n_repeats_2d = 3  # Number of experiment repeats
        mse_base_repeats = []
        mse_sob_repeats = []
        mse_ll_repeats = []
        error_base_repeats = []
        error_sob_repeats = []
        error_ll_repeats = []
        
        # Store last gradient results for heatmap plotting (use results when num_per_mode=10)
        grad_base_final = None
        grad_sob_final = None
        grid_points_final = None
        
        print(f"Running {n_repeats_2d} repeats for statistical significance...")
        
        for repeat in range(n_repeats_2d):
            seed_repeat = base_seed * 1000 + repeat
            
            # Setup distribution and generate training data
            sample_p, sample_q, _, _, _, _, _, _ = setup_distribution(dist_type, seed=seed_repeat)
            p_x = sample_p(num_per_mode)
            q_x = sample_q(num_per_mode)
            x = np.concatenate([p_x, q_x], axis=0)
            y = np.concatenate([np.zeros((num_per_mode,1)), np.ones((num_per_mode,1))], axis=0)
            
            perm = np.random.permutation(len(x))
            x = torch.tensor(x[perm], dtype=torch.float, device=device)
            y = torch.tensor(y[perm], dtype=torch.float, device=device)
            
            dataset = TensorDataset(x, y)
            loader = DataLoader(dataset, batch_size=min(256, len(x)), shuffle=True)
            
            # Train model (using ReLU activation)
            baseline = train_baseline(loader, epochs=300, seed=seed_repeat, d=2, use_smooth_activation=False)
            G_base = grad_from_model(baseline, Xtest)
            mse_base = torch.mean((G_base - G_true)**2).item()
            mse_base_repeats.append(mse_base)
            error_base = torch.norm(G_base - G_true).item()
            error_base_repeats.append(error_base)
            
            baseline_sob = train_baseline_sobolev(loader, epochs=300, lambda_grad=1e-3, seed=seed_repeat, d=2, use_smooth_activation=False)
            G_base_sob = grad_from_model(baseline_sob, Xtest)
            mse_base_sob = torch.mean((G_base_sob - G_true)**2).item()
            mse_sob_repeats.append(mse_base_sob)
            error_sob = torch.norm(G_base_sob - G_true).item()
            error_sob_repeats.append(error_sob)
            
            # Train Local Linear model
            p_x_torch = torch.tensor(p_x, dtype=torch.float, device=device)
            q_x_torch = torch.tensor(q_x, dtype=torch.float, device=device)
            
            # Compute sigma using only training data (matching wgf_da.py convention)
            # In wgf_da.py, sigma = (.5*comp_dist(Xq, Xp).flatten().median()).sqrt()
            # All hyperparameters should be estimated from training data only
            sigma_ll = (0.5 * comp_dist(q_x_torch, p_x_torch).flatten().median()).sqrt().item()
            
            # Note: Local Linear estimates ∇log(p/q) when p is target and q is source
            # So we swap: use q_x as xp (target) and p_x as xq (source)
            # This gives us ∇log(q/p) which is what we need
            ll_net = train_local_linear(
                xp=q_x_torch,  # target (q)
                xq=p_x_torch,  # source (p)
                x=Xtest,       # points where gradient is estimated
                sigma=sigma_ll,
                epochs=500,
                batch_size=256,
                lr=1e-1,
                seed=seed_repeat
            )
            G_ll = grad_from_local_linear(ll_net, Xtest)
            mse_ll = torch.mean((G_ll - G_true)**2).item()
            mse_ll_repeats.append(mse_ll)
            error_ll = torch.norm(G_ll - G_true).item()
            error_ll_repeats.append(error_ll)
            
            # Save last results for heatmap plotting (temporarily commented out)
            # if num_per_mode == 100 and repeat == n_repeats_2d - 1:
            #     # Create grid for heatmap
            #     p_samples = sample_p(1000)
            #     q_samples = sample_q(1000)
            #     x_min = min(p_samples[:, 0].min(), q_samples[:, 0].min()) - 0.5
            #     x_max = max(p_samples[:, 0].max(), q_samples[:, 0].max()) + 0.5
            #     y_min = min(p_samples[:, 1].min(), q_samples[:, 1].min()) - 0.5
            #     y_max = max(p_samples[:, 1].max(), q_samples[:, 1].max()) + 0.5
            #     
            #     xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
            #                          np.linspace(y_min, y_max, 100))
            #     grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float, device=device)
            #     
            #     # Compute ground truth and estimated gradients
            #     G_true_grid = compute_true_grad(grid_points, means_p, means_q, Sigma_inv_torch, 
            #                                     dist_type=dist_type, dist_params=dist_params).detach()
            #     grad_base_final = grad_from_model(baseline, grid_points)
            #     grad_sob_final = grad_from_model(baseline_sob, grid_points)
            #     grid_points_final = (xx, yy, G_true_grid)
            
            if (repeat + 1) % 5 == 0:
                print(f"  Completed {repeat + 1}/{n_repeats_2d} repeats")
        
        # Compute mean and standard error
        mean_base = np.mean(mse_base_repeats)
        std_base = np.std(mse_base_repeats) / np.sqrt(n_repeats_2d)
        mean_sob = np.mean(mse_sob_repeats)
        std_sob = np.std(mse_sob_repeats) / np.sqrt(n_repeats_2d)
        mean_ll = np.mean(mse_ll_repeats)
        std_ll = np.std(mse_ll_repeats) / np.sqrt(n_repeats_2d)
        
        error_base = np.mean(error_base_repeats)
        error_sob = np.mean(error_sob_repeats)
        error_ll = np.mean(error_ll_repeats)
        error_std_base = np.std(error_base_repeats) / np.sqrt(n_repeats_2d)
        error_std_sob = np.std(error_sob_repeats) / np.sqrt(n_repeats_2d)
        error_std_ll = np.std(error_ll_repeats) / np.sqrt(n_repeats_2d)
        
        # Store error vs sample data
        error_vs_sample_data[dist_type]['n_list'].append(2 * num_per_mode)
        error_vs_sample_data[dist_type]['errors_baseline'].append(error_base)
        error_vs_sample_data[dist_type]['errors_sobolev'].append(error_sob)
        error_vs_sample_data[dist_type]['errors_local_linear'].append(error_ll)
        error_vs_sample_data[dist_type]['std_errors_baseline'].append(error_std_base)
        error_vs_sample_data[dist_type]['std_errors_sobolev'].append(error_std_sob)
        error_vs_sample_data[dist_type]['std_errors_local_linear'].append(error_std_ll)
        
        print(f"  Baseline MSE: {mean_base:.6f} ± {std_base:.6f}")
        print(f"  Baseline+Sobolev MSE: {mean_sob:.6f} ± {std_sob:.6f}")
        print(f"  Local Linear MSE: {mean_ll:.6f} ± {std_ll:.6f}")
        
        # Save heatmap data (temporarily commented out)
        # if num_per_mode == 100 and grad_base_final is not None:
        #     error_vs_sample_data[dist_type]['heatmap_data'] = {
        #         'xx': grid_points_final[0],
        #         'yy': grid_points_final[1],
        #         'grad_true': grid_points_final[2],
        #         'grad_base': grad_base_final,
        #         'grad_sob': grad_sob_final,
        #         'title': title
        #     }
        
        # Store results
        results.append({
            'Distribution': dist_type,
            'Title': title,
            'Num_Per_Mode': num_per_mode,
            'Train_Size': 2 * num_per_mode,
            'Baseline': mean_base,
            'Baseline_Std': std_base,
            'Baseline+Sobolev': mean_sob,
            'Baseline+Sobolev_Std': std_sob,
            'Local_Linear': mean_ll,
            'Local_Linear_Std': std_ll,
            'Improvement (Base+ Sob)': (mean_base - mean_sob) / (mean_base + 1e-12) * 100,
            'Improvement (Local Linear)': (mean_base - mean_ll) / (mean_base + 1e-12) * 100
        })
        
        print(f"\n{title} (Train: {2*num_per_mode} samples) Results:")
        print(f"  Baseline:           {mean_base:.6f} ± {std_base:.6f}")
        print(f"  Baseline+Sobolev:   {mean_sob:.6f} ± {std_sob:.6f} ({((mean_base-mean_sob)/mean_base*100):.2f}% improvement)")
        print(f"  Local Linear:       {mean_ll:.6f} ± {std_ll:.6f} ({((mean_base-mean_ll)/mean_base*100):.2f}% improvement)")

# ============ Generate Results Table ============
print(f"\n{'='*80}")
print("Final Results Summary")
print(f"{'='*80}\n")

# Print formatted table
print("="*180)
print("MSE Results Table:")
print("="*180)
print(f"{'Distribution':<25} {'Train Size':<12} {'Baseline':<20} {'Base+Sob':<20} {'Local Linear':<20} {'Imp (Base+Sob)':<18} {'Imp (LL)':<18}")
print("-"*180)
for r in results:
    baseline_str = f"{r['Baseline']:.6f}"
    sob_str = f"{r['Baseline+Sobolev']:.6f}"
    ll_str = f"{r['Local_Linear']:.6f}"
    if 'Baseline_Std' in r:
        baseline_str = f"{r['Baseline']:.6f} ± {r['Baseline_Std']:.6f}"
        sob_str = f"{r['Baseline+Sobolev']:.6f} ± {r['Baseline+Sobolev_Std']:.6f}"
        ll_str = f"{r['Local_Linear']:.6f} ± {r['Local_Linear_Std']:.6f}"
    print(f"{r['Title']:<25} {r['Train_Size']:<12} {baseline_str:<20} {sob_str:<20} {ll_str:<20} "
          f"{r['Improvement (Base+ Sob)']:<18.2f}% {r['Improvement (Local Linear)']:<18.2f}%")
print("="*180)

# Print grouped by distribution type
print(f"\n{'='*180}")
print("Results Grouped by Distribution:")
print("="*180)
for dist_type in DISTRIBUTION_TYPES:
    print(f"\n{dist_type.upper()}:")
    print(f"{'Train Size':<12} {'Baseline':<20} {'Base+Sob':<20} {'Local Linear':<20} {'Imp (Base+Sob)':<18} {'Imp (LL)':<18}")
    print("-"*130)
    dist_results = [r for r in results if r['Distribution'] == dist_type]
    for r in sorted(dist_results, key=lambda x: x['Train_Size']):
        baseline_str = f"{r['Baseline']:.6f}"
        sob_str = f"{r['Baseline+Sobolev']:.6f}"
        ll_str = f"{r['Local_Linear']:.6f}"
        if 'Baseline_Std' in r:
            baseline_str = f"{r['Baseline']:.6f} ± {r['Baseline_Std']:.6f}"
            sob_str = f"{r['Baseline+Sobolev']:.6f} ± {r['Baseline+Sobolev_Std']:.6f}"
            ll_str = f"{r['Local_Linear']:.6f} ± {r['Local_Linear_Std']:.6f}"
        print(f"{r['Train_Size']:<12} {baseline_str:<20} {sob_str:<20} {ll_str:<20} "
              f"{r['Improvement (Base+ Sob)']:<18.2f}% {r['Improvement (Local Linear)']:<18.2f}%")

# Save table as CSV
csv_path = f'{output_dir}/results_comparison.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    # Check if standard deviation data exists
    has_std = 'Baseline_Std' in results[0] if results else False
    if has_std:
        writer.writerow(['Distribution', 'Title', 'Num_Per_Mode', 'Train_Size', 'Baseline', 'Baseline_Std', 
                         'Baseline+Sobolev', 'Baseline+Sobolev_Std', 'Local_Linear', 'Local_Linear_Std',
                         'Improvement (Base+ Sob)', 'Improvement (Local Linear)'])
        for r in results:
            writer.writerow([r['Distribution'], r['Title'], r['Num_Per_Mode'], r['Train_Size'], 
                            r['Baseline'], r.get('Baseline_Std', ''), r['Baseline+Sobolev'], 
                            r.get('Baseline+Sobolev_Std', ''), r['Local_Linear'], r.get('Local_Linear_Std', ''),
                            r['Improvement (Base+ Sob)'], r['Improvement (Local Linear)']])
    else:
        writer.writerow(['Distribution', 'Title', 'Num_Per_Mode', 'Train_Size', 'Baseline', 'Baseline+Sobolev', 
                         'Local_Linear', 'Improvement (Base+ Sob)', 'Improvement (Local Linear)'])
        for r in results:
            writer.writerow([r['Distribution'], r['Title'], r['Num_Per_Mode'], r['Train_Size'], 
                            r['Baseline'], r['Baseline+Sobolev'], r['Local_Linear'], 
                            r['Improvement (Base+ Sob)'], r['Improvement (Local Linear)']])
print(f"\nResults saved to: {csv_path}")

# ============ Plot Error vs Sample and Heatmaps ============
print(f"\n{'='*80}")
print("Generating Error vs Sample plots and Heatmaps")
print(f"{'='*80}\n")

# Plot error vs sample for each distribution type
for dist_type in DISTRIBUTION_TYPES:
    data = error_vs_sample_data[dist_type]
    if len(data['n_list']) > 0:
        print(f"Generating error vs samples plot for {dist_type}...")
        plot_error_vs_samples(
            data['n_list'], 
            data['errors_baseline'], 
            data['errors_sobolev'],
            data['errors_local_linear'],
            data['std_errors_baseline'],
            data['std_errors_sobolev'],
            data['std_errors_local_linear'],
            save_path=f'{output_dir}/error_vs_samples_{dist_type}.png'
        )
    
    # Plot heatmap (combine Truth, Baseline and Baseline+Sobolev into one figure)
    # Heatmap plotting temporarily commented out
    # if 'heatmap_data' in data:
    #     hd = data['heatmap_data']
    #     print(f"Generating combined heatmap for {dist_type}...")
    #     plot_gradient_heatmap_2d(
    #         hd['xx'], hd['yy'], 
    #         hd['grad_true'], hd['grad_base'], hd['grad_sob'],
    #         hd['title'],
    #         save_path=f'gradient_heatmap_{dist_type}.png'
    #     )

print("\nAll plots generated!")
