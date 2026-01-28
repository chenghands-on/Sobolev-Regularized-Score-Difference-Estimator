import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from core.util import comp_dist, comp_median, kernel_comp, dKernel_comp

# psicon = lambda d: torch.exp(d - 1)
psicon = lambda d: d**2/2 + d
rbfkernel = lambda x, y, sigma: kernel_comp(x, y, sigma)

def obj(W, b, xp, xq, psicon, kpx=None, kqx=None):
    A = torch.mean(kpx * (torch.matmul(W, xp.T) + b), 1, keepdim=True)
    B = torch.mean(kqx * psicon( (torch.matmul(W, xq.T) + b)), 1, keepdim=True)
    return torch.mean(- A + B, 0)

def gradest(net, xp, xq, x, sigma, optimizer, kernel = rbfkernel, batch_size = 256, nepochs = 100, penalty = None):
    
    d = xp.shape[1]

    kpx = kernel(x, xp, sigma)
    kqx = kernel(x, xq, sigma)
    
    # Create a generator for deterministic DataLoader shuffling
    generator = torch.Generator()
    generator.manual_seed(42)  # Fixed seed for DataLoader
    
    # create the neural network
    train_loader = torch.utils.data.DataLoader( 
                torch.utils.data.TensorDataset(xp, xq, kpx.T, kqx.T), batch_size=batch_size, shuffle=True, generator=generator)
    
    # start the training loop
    for epoch in range(nepochs):
        # iterate over the data
        for i, data in enumerate(train_loader):
            # get the batch
            xpi, xqi, kpxi, kqxi = data
            kpxi = kpxi.T; kqxi = kqxi.T
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            predicted = net(x)
            Wx = predicted[:, :d]
            bx = predicted[:, d:]
            
            output = obj(Wx, bx, xpi, xqi, psicon, kpxi, kqxi)
            
            if penalty is not None:
                output = output + penalty(x)
            
            output.backward()
            optimizer.step()

            # # print statistics
            # if epoch % 100 == 0:
            #     print('[%d, %5d] loss: %f' %(epoch, i, output.item()))
                
    return net

def MMD_flow(zp, zq, kernel = rbfkernel, sigma = None):
    zq.requires_grad = True
    kqq = kernel(zq, zq, sigma)
    kpq = kernel(zp, zq, sigma)
    
    MMDobj = torch.mean(kqq) - 2*torch.mean(kpq)
    
    # compute the gradient with respect to xq using automatic differentiation
    grad = torch.autograd.grad(-MMDobj, zq)[0]
    return grad * 1000000

def sobolev_penalty(model, x, detach_model=False):
    """
    Compute Sobolev penalty: ||∇f(x)||^2 where f is the model output (logit).
    
    Args:
        model: neural network model
        x: input tensor [B, d]
        detach_model: if True, detach model parameters (for alternating updates)
                      Currently not used, kept for future extension
    
    Returns:
        penalty: mean of squared gradient norm
    """
    x = x.clone().detach().requires_grad_(True)
    f = model(x)  # [B, 1] - logit = log(p/q)
    
    # Compute gradient of f w.r.t. x
    grad_f = torch.autograd.grad(f.sum(), x, create_graph=True)[0]  # [B, d]
    # Return mean of squared norm
    return (grad_f**2).sum(dim=1).mean()

def grad_from_classifier(model, x, batch_size=10000):
    """
    Extract gradient of log(p/q) from trained classifier.
    
    Args:
        model: trained classifier that outputs logit = log(p/q)
        x: input tensor [B, d] where gradient is computed
        batch_size: batch size for processing large datasets (default: 10000)
    
    Returns:
        grad: gradient tensor [B, d] = ∇log(p/q)
    """
    # Process in batches if dataset is large to avoid memory issues and improve speed
    if x.shape[0] > batch_size:
        from torch.utils.data import DataLoader, TensorDataset
        grad_list = []
        dataloader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)
        
        for batch_data in dataloader:
            x_batch = batch_data[0].clone().detach().requires_grad_(True)
            f_batch = model(x_batch)  # [batch_size, 1] - logit = log(p/q)
            grad_batch = torch.autograd.grad(f_batch.sum(), x_batch, create_graph=False)[0]  # [batch_size, d]
            grad_list.append(grad_batch.detach())
        
        grad = torch.cat(grad_list, dim=0)
    else:
        # Small dataset: process all at once
        x = x.clone().detach().requires_grad_(True)
        f = model(x)  # [B, 1] - logit = log(p/q)
        grad = torch.autograd.grad(f.sum(), x, create_graph=False)[0]  # [B, d]
        grad = grad.detach()
    
    return grad

def gradest_classification(model, xp, xq, x, optimizer, 
                          batch_size=256, nepochs=100, 
                          lambda_sobolev=0.0, use_sobolev=False,
                          sobolev_anneal=True, sobolev_anneal_steps=100,
                          early_stopping=True, patience=20, min_delta=1e-6):
    """
    Estimate density ratio gradient using classification approach.
    
    In gradest2 convention:
    - xp: target domain data (p)
    - xq: source domain data (q)
    - We estimate ∇log(p/q)
    
    Args:
        model: classifier network (outputs logit = log(p/q))
        xp: target domain samples [Np, d]
        xq: source domain samples [Nq, d]
        x: points where gradient is estimated [N, d] (usually xq)
        optimizer: optimizer for training
        batch_size: batch size for training
        nepochs: number of training epochs
        lambda_sobolev: coefficient for Sobolev regularization
        use_sobolev: whether to use Sobolev regularization
        sobolev_anneal: whether to use annealing for Sobolev regularization
        sobolev_anneal_steps: number of steps for annealing (from 0 to lambda_sobolev)
    
    Returns:
        model: trained model
    """
    # Prepare data: xp (target) -> label 1, xq (source) -> label 0
    # This is opposite to simulation.py convention
    # Ensure balanced dataset by resampling the smaller one
    n_p, n_q = xp.shape[0], xq.shape[0]
    if n_p > n_q:
        # Resample xq to match xp
        idx_q = torch.randint(0, n_q, (n_p,), device=xq.device)
        xq_balanced = xq[idx_q]
        xp_balanced = xp
    elif n_p < n_q:
        # Resample xp to match xq
        idx_p = torch.randint(0, n_p, (n_q,), device=xp.device)
        xp_balanced = xp[idx_p]
        xq_balanced = xq
    else:
        # Already balanced
        xp_balanced = xp
        xq_balanced = xq
    
    X_all = torch.cat([xp_balanced, xq_balanced], dim=0)
    y_all = torch.cat([
        torch.ones(xp_balanced.shape[0], 1, device=xp.device, dtype=torch.float32),  # target = 1
        torch.zeros(xq_balanced.shape[0], 1, device=xq.device, dtype=torch.float32)   # source = 0
    ], dim=0)
    
    # Create a generator for deterministic DataLoader shuffling
    generator = torch.Generator()
    generator.manual_seed(42)  # Fixed seed for DataLoader
    
    # Create data loader
    dataset = torch.utils.data.TensorDataset(X_all, y_all)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    
    # Loss function: BCE with logits
    loss_fn = nn.BCEWithLogitsLoss()
    
    # Early stopping setup
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    step = 0
    for epoch in range(nepochs):
        epoch_losses = []
        
        for xb, yb in train_loader:
            step += 1
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(xb)  # [B, 1]
            loss_ce = loss_fn(logits, yb)
            
            # Sobolev regularization (optional)
            if use_sobolev and lambda_sobolev > 0:
                sob = sobolev_penalty(model, xb, detach_model=False)
                
                # Annealing: gradually increase lambda from 0 to lambda_sobolev
                if sobolev_anneal:
                    lam0, lam1, T = 0.0, lambda_sobolev, sobolev_anneal_steps
                    lam_t = lam0 + (lam1 - lam0) * min(step / T, 1.0)
                else:
                    lam_t = lambda_sobolev
                
                loss = loss_ce + lam_t * sob
            else:
                loss = loss_ce
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        # Compute average loss for this epoch
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        
        # Early stopping check
        if early_stopping:
            if avg_epoch_loss < best_loss - min_delta:
                # Loss improved
                best_loss = avg_epoch_loss
                patience_counter = 0
                # Save best model state
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                # Loss did not improve
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}/{nepochs} (best loss: {best_loss:.6f})")
                    # Restore best model
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break
    
    return model
    
    
    
    
    
