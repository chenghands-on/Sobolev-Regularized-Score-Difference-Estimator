import sklearn
from scipy.spatial.distance import cdist
import torch
from core.util import comp_dist
from numpy import *
import numpy as np
import torch.optim as optim
import classif
import ot
import time
from wgf import gradest, MMD_flow, gradest_classification, grad_from_classifier
from core.nn import NPnet, DensityRatioClassifier

def OTYphat(xq, yq, xp, Chinge = None, lmb = 1e1):
    yyq,Yb=classif.get_label_matrix(yq)
    
    wa=torch.ones((xq.shape[0],))/xq.shape[0]
    wb=torch.ones((xp.shape[0],))/xp.shape[0]
    C0=torch.tensor(cdist(xq.cpu(), xp.cpu(),metric='sqeuclidean'))
    
    if Chinge is None:
        Chinge=torch.zeros_like(torch.tensor(C0))
        
    C=.1*C0+Chinge
    G = ot.emd(wa.cpu().numpy(),wb.cpu().numpy(),C.cpu().numpy())
    # Kt=sklearn.metrics.pairwise.rbf_kernel(Xp,gamma=gam)
    Kt = sklearn.metrics.pairwise.linear_kernel(xp)
    Yst=xp.shape[0]*G.T.dot((yyq+1)/2.)
    g = classif.SVMClassifier(lmb)
    
    g.fit(Kt,Yst)
    ypred=g.predict(Kt)
    yphat = (ypred.argmax(1))+1
    Chinge=classif.loss_hinge(yyq,ypred)
    
    return yphat, Chinge

def concat(xp, yp, xq, yq, device):
    Zp = torch.cat([xp, torch.tensor(yp, dtype=torch.float32, device=device).view(-1,1)], 1)
    # idx = random.choice(Zp.shape[0], xq.shape[0])
    # Zp = Zp[idx, :]

    Zq = torch.cat([xq, torch.tensor(yq, dtype=torch.float32, device=device).view(-1,1)], 1)
    
    return Zp, Zq

def train_classifier_for_domain_adaptation(Xp, yp, Xq, yq, device, classifier_hidden=64, 
                                          classifier_lr=1e-3, lambda_sobolev=1e-4, 
                                          use_sobolev=False, sobolev_anneal=True, 
                                          sobolev_anneal_steps=100, batch_size=500):
    """
    Train classifier once before WGF epochs.
    Returns: classifier_net, initial yphat, initial Chinge, initial Zp, initial Zq, initial Xq
    """
    yphat, Chinge = OTYphat(Xq.cpu(), yq, Xp.cpu())
    Zp, Zq = concat(Xp, yphat, Xq, yq, device)
    
    d = Zq.shape[1]  # dimension of Z (including label)
    classifier_net = DensityRatioClassifier(d, h=classifier_hidden, activation='silu').to(device)
    classifier_opt = optim.Adam(classifier_net.parameters(), lr=classifier_lr)
    
    # Train classifier using all data (1000 epochs with early stopping)
    print("Training classifier (gradient estimator)...")
    gradest_classification(
        classifier_net, Zp, Zq, Zq, classifier_opt,
        batch_size=batch_size, nepochs=1000,
        lambda_sobolev=lambda_sobolev, use_sobolev=use_sobolev,
        sobolev_anneal=sobolev_anneal, sobolev_anneal_steps=sobolev_anneal_steps,
        early_stopping=True, patience=20, min_delta=1e-6
    )
    print("Classifier training completed.")
    
    # Convert Chinge to tensor if it's numpy array
    if Chinge is not None and not isinstance(Chinge, torch.Tensor):
        Chinge = torch.tensor(Chinge)
    
    return classifier_net, yphat, Chinge, Zp, Zq, Xq.clone()

def apply_gradient_steps(classifier_net, Xp, yp, Xq, yq, Zp, Zq, Chinge, device, 
                        num_steps, step_size=0.01, classifier_lr=2e-4, 
                        lambda_sobolev=1e-4, use_sobolev=True, 
                        sobolev_anneal=True, sobolev_anneal_steps=100,
                        retrain_classifier=True, batch_size=500,
                        adaptive_lambda=False, initial_lambda=None,
                        lambda_min=1e-6, lambda_max=1e-3):
    """
    Apply num_steps gradient updates. 
    If retrain_classifier=True, retrain classifier at each step starting from previous parameters.
    Returns: final yphat, final Xq, trajectory of Xq, lambda_history, timing_info
    timing_info: dict with 'total_time', 'gradient_times' (list of per-step gradient computation times)
    """
    start_time = time.time()
    Xq_traj = [Xq.clone()]
    current_Xq = Xq.clone()
    current_Zq = Zq.clone()
    current_Zp = Zp.clone()
    # Chinge might be numpy array or tensor, handle both cases
    if Chinge is not None:
        if isinstance(Chinge, torch.Tensor):
            current_Chinge = Chinge.clone()
        else:
            current_Chinge = torch.tensor(Chinge)
    else:
        current_Chinge = None
    current_yphat = None
    
    # Create optimizer for classifier (will be reused if retraining)
    classifier_opt = optim.Adam(classifier_net.parameters(), lr=classifier_lr)
    
    # Initialize lambda for adaptive mode
    if adaptive_lambda:
        current_lambda = initial_lambda if initial_lambda is not None else lambda_sobolev
        lambda_history = [current_lambda]
    else:
        current_lambda = lambda_sobolev
        lambda_history = None
    
    # Track finetune (retrain) and gradient computation times separately
    finetune_times = []  # Time for retraining classifier at each step
    estimate_gradient_times = []  # Time for computing gradient (forward+backward)
    
    for step_idx in range(num_steps):
        if retrain_classifier:
            # Retrain classifier on current joint distribution (starting from previous parameters)
            print(f"Retraining classifier at step {step_idx+1}/{num_steps}...")
            finetune_start_time = time.time()
            gradest_classification(
                classifier_net, current_Zp, current_Zq, current_Zq, classifier_opt,
                batch_size=batch_size, nepochs=1000,
                lambda_sobolev=current_lambda, use_sobolev=use_sobolev,
                sobolev_anneal=sobolev_anneal, sobolev_anneal_steps=sobolev_anneal_steps,
                early_stopping=True, patience=20, min_delta=1e-6
            )
            finetune_time = time.time() - finetune_start_time
            finetune_times.append(finetune_time)
            print(f"Classifier retraining completed at step {step_idx+1}/{num_steps}.")
        else:
            finetune_times.append(0.0)  # No finetune if retrain_classifier=False
        
        # Compute gradient using (re)trained classifier - measure time (estimate gradient only)
        grad_start_time = time.time()
        grad = grad_from_classifier(classifier_net, current_Zq)[:, :current_Xq.shape[1]].detach()
        grad_time = time.time() - grad_start_time
        estimate_gradient_times.append(grad_time)
        
        # Gradient update
        current_Xq = current_Xq + step_size * grad
        
        # Update yphat and Zq (joint distribution changes after Xq update)
        current_yphat, current_Chinge = OTYphat(current_Xq.cpu(), yq, Xp.cpu(), Chinge=current_Chinge)
        current_Zp, current_Zq = concat(Xp, current_yphat, current_Xq, yq, device)
        
        # Record trajectory
        Xq_traj.append(current_Xq.clone())
        
        # Print accuracy
        accuracy = mean(yp==current_yphat)
        print(f"Gradient step {step_idx+1}/{num_steps} accuracy:", accuracy)
    
    total_time = time.time() - start_time
    timing_info = {
        'total_time': total_time,
        'finetune_times': finetune_times,
        'estimate_gradient_times': estimate_gradient_times,
        'avg_finetune_time': np.mean(finetune_times) if finetune_times else 0.0,
        'avg_estimate_gradient_time': np.mean(estimate_gradient_times) if estimate_gradient_times else 0.0,
        # Keep backward compatibility
        'gradient_times': estimate_gradient_times,
        'avg_gradient_time': np.mean(estimate_gradient_times) if estimate_gradient_times else 0.0
    }
    
    return current_yphat, current_Xq, Xq_traj, lambda_history, timing_info

def WGF_DomainAdaptation(Xp, yp, Xq, yq, kernel, nepoch = 5, VGD_batchsize = 500, device = 'cpu',
                         grad_method='kernel', lambda_sobolev=1e-4, use_sobolev=False,
                         classifier_hidden=64, classifier_lr=1e-3, sobolev_anneal=True, sobolev_anneal_steps=100,
                         classification_single_update=False, gradient_step_size=0.05, num_gradient_steps=None):
    import time
    # Create a generator for deterministic DataLoader shuffling
    generator = torch.Generator()
    generator.manual_seed(42)  # Fixed seed for DataLoader
    
    yphat, Chinge = OTYphat(Xq.cpu(), yq, Xp.cpu())
    Zp, Zq = concat(Xp, yphat, Xq, yq, device)

    idxp = torch.tensor(range(Zp.shape[0]))
    idxq = torch.tensor(range(Zq.shape[0]))
    
    # if two datasets are not the same, resample the small dataset to match the big dataset. 
    if Zp.shape[0] > Zq.shape[0]:
        idxq = random.choice(Zq.shape[0], Zp.shape[0])
    elif Zp.shape[0] < Zq.shape[0]:
        idxp = random.choice(Zp.shape[0], Zq.shape[0])

    train_loader = torch.utils.data.DataLoader( 
                torch.utils.data.TensorDataset(torch.tensor(idxp), 
                torch.tensor(idxq)), batch_size=VGD_batchsize, shuffle=True, generator=generator)
    Xq_traj = [Xq]

    # For classification method: train classifier once before WGF epochs
    if grad_method == 'classification':
        d = Zq.shape[1]  # dimension of Z (including label)
        classifier_net = DensityRatioClassifier(d, h=classifier_hidden, activation='silu').to(device)
        classifier_opt = optim.Adam(classifier_net.parameters(), lr=classifier_lr)
        
        # Train classifier using all data (1000 epochs with early stopping)
        print("Training classifier (gradient estimator) before WGF epochs...")
        gradest_classification(
            classifier_net, Zp, Zq, Zq, classifier_opt,
            batch_size=500, nepochs=1000,
            lambda_sobolev=lambda_sobolev, use_sobolev=use_sobolev,
            sobolev_anneal=sobolev_anneal, sobolev_anneal_steps=sobolev_anneal_steps,
            early_stopping=True, patience=20, min_delta=1e-6
        )
        print("Classifier training completed.")

    # Initialize timing info for kernel method
    if grad_method == 'kernel':
        WGF_DomainAdaptation.kernel_times = []
        WGF_DomainAdaptation.grad_extract_times = []
    
    # WGF epoch loop
    for epoch in range(nepoch):
        if grad_method == 'classification':
            # Classification method: use pre-trained classifier for gradient updates
            # No training in WGF epochs
            
            # For single update mode: compute gradient once and update
            if classification_single_update:
                grad = grad_from_classifier(classifier_net, Zq)[:, :Xq.shape[1]].detach()
                Xq = Xq + gradient_step_size * grad  # Configurable step size
                yphat, Chinge = OTYphat(Xq.cpu(), yq, Xp.cpu(), Chinge = Chinge)
                Zp, Zq = concat(Xp, yphat, Xq, yq, device)
                Xq_traj.append(Xq)
                accuracy = mean(yp==yphat)
                print(f"WGF epoch {epoch+1} accuracy:", accuracy)
            elif num_gradient_steps is not None:
                # Multi-step update mode: use pre-trained classifier for multiple gradient updates
                # Each WGF epoch: do num_gradient_steps gradient updates (step size 0.01)
                for step_idx in range(num_gradient_steps):
                    grad = grad_from_classifier(classifier_net, Zq)[:, :Xq.shape[1]].detach()
                    Xq = Xq + 0.01 * grad  # Fixed step size 0.01 for multi-step
                    yphat, Chinge = OTYphat(Xq.cpu(), yq, Xp.cpu(), Chinge = Chinge)
                    Zp, Zq = concat(Xp, yphat, Xq, yq, device)
                    Xq_traj.append(Xq)
                    accuracy = mean(yp==yphat)
                    print(f"WGF epoch {epoch+1}, gradient step {step_idx+1}/{num_gradient_steps} accuracy:", accuracy)
            else:
                # Multiple updates per epoch (one per batch)
                for i, data in enumerate(train_loader):
                    print("iteration:", i, "...")
                    idxp, idxq = data
                    Zpi = Zp[idxp, :]; Zqi = Zq[idxq, :]
                    
                    # Use the pre-trained classifier to compute gradient at current Zq
                    grad = grad_from_classifier(classifier_net, Zq)[:, :Xq.shape[1]].detach()
                    
                    # gradient variational descent
                    Xq = Xq + 0.01 * grad
                    
                    yphat, Chinge = OTYphat(Xq.cpu(), yq, Xp.cpu(), Chinge = Chinge)
                    Zp, Zq = concat(Xp, yphat, Xq, yq, device)
                    
                    Xq_traj.append(Xq)
                    
                    accuracy = mean(yp==yphat)
                    print("accuracy:", accuracy)
        else:
            # Kernel method: original logic (train per batch)
            for i, data in enumerate(train_loader):
                print("iteration:", i, "...")
                idxp, idxq = data
                Zpi = Zp[idxp, :]; Zqi = Zq[idxq, :]
                
                lmbd = 0.0000
                sigma = (.5*comp_dist(Xq, Xp).flatten().median()).sqrt()
                        
                if grad_method == 'kernel':
                    # Original kernel-based method
                    net = NPnet(Xq.shape[0], Xp.shape[1] + 1).to(device)
                    optimizer = optim.Adagrad(net.parameters(), lr=1e-1)
                    
                    # Measure kernel computation and training time
                    kernel_batch_start_time = time.time()
                    # Reverse KL flow (includes kernel computation + network training)
                    gradnet = gradest(net, Zpi, Zqi, Zq, sigma, optimizer, batch_size=500, nepochs = 500, kernel=kernel)
                    kernel_batch_train_time = time.time() - kernel_batch_start_time
                    
                    # Measure gradient extraction time (just forward pass)
                    grad_extract_start_time = time.time()
                    grad = gradnet(Zq)[:, :Xq.shape[1]].detach()
                    grad_extract_time = time.time() - grad_extract_start_time
                    
                    # Store timing info (will be accumulated)
                    if not hasattr(WGF_DomainAdaptation, 'kernel_times'):
                        WGF_DomainAdaptation.kernel_times = []
                        WGF_DomainAdaptation.grad_extract_times = []
                    WGF_DomainAdaptation.kernel_times.append(kernel_batch_train_time)
                    WGF_DomainAdaptation.grad_extract_times.append(grad_extract_time)
                else:
                    raise ValueError(f"Unknown grad_method: {grad_method}. Choose 'kernel' or 'classification'.")
                
                # gradient variational descent
                Xq = Xq + .01*grad
                
                yphat, Chinge = OTYphat(Xq.cpu(), yq, Xp.cpu(), Chinge = Chinge)
                Zp, Zq = concat(Xp, yphat, Xq, yq, device)

                Xq_traj.append(Xq)
                
                accuracy = mean(yp==yphat)
                print("accuracy:", accuracy)
                        
        print("done!\n")
    
    # Return timing info if kernel method
    if grad_method == 'kernel' and hasattr(WGF_DomainAdaptation, 'kernel_times'):
        timing_info = {
            'kernel_times': WGF_DomainAdaptation.kernel_times,
            'grad_extract_times': WGF_DomainAdaptation.grad_extract_times,
            'avg_kernel_time': np.mean(WGF_DomainAdaptation.kernel_times) if WGF_DomainAdaptation.kernel_times else 0.0,
            'avg_grad_extract_time': np.mean(WGF_DomainAdaptation.grad_extract_times) if WGF_DomainAdaptation.grad_extract_times else 0.0
        }
        # Clean up
        delattr(WGF_DomainAdaptation, 'kernel_times')
        delattr(WGF_DomainAdaptation, 'grad_extract_times')
        return yphat, Xq, Xq_traj, timing_info
    else:
        return yphat, Xq, Xq_traj, None