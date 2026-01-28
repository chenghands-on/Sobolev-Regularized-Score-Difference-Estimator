# %%

import os
import matplotlib.pyplot as plt
import torch
from numpy import *
from core.util import svm
from core.util import comp_dist, kernel_comp
import classif
from wgf_da import WGF_DomainAdaptation
import time

from IPython import display

import pylab as pl
import matplotlib.pyplot as plt

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
decaf_dir = os.path.join(script_dir, 'decaf6')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def kernel(x,y,sigma):
    # # the first till the second last, use gaussian kernel
    x1 = x[:, :-1]
    y1 = y[:, :-1]
    k1 = kernel_comp(x1, y1, sigma)
    
    # the last dimension, use delta kernel
    x2 = x[:, -1:]
    y2 = y[:, -1:]
    k2 = x2 - y2.T
    k2 = (k2 == 0)
    
    return k1*k2
    # return kernel_comp(x, y, sigma)

# %%
seed = 42  # Changed from 123 to 42
# Set all random seeds for reproducibility
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
# Fix numpy seed
random.seed(seed)
# Fix Python random seed (if used)
import random as py_random
py_random.seed(seed)
# Set deterministic behavior for PyTorch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define all 12 dataset pairs
datasets = ['amazon', 'caltech', 'dslr', 'webcam']
dataset_pairs = []
for ds_s in datasets:
    for ds_t in datasets:
        if ds_s != ds_t:  # Exclude same dataset pairs
            dataset_pairs.append((ds_s, ds_t))
# Use all dataset pairs (12 pairs total)

all_results = {}

# Define lambda_sobolev values to test (fixed lambda values)
lambda_sobolev_list = [1e-5]  # Only test 1e-5
# lambda_sobolev_list = [1e-5, 3e-5, 1e-4]  # All values

# Loop over all dataset pairs
for pair_idx, (ds_s, ds_t) in enumerate(dataset_pairs):
    print("\n" + "="*80)
    print(f"Dataset Pair {pair_idx+1}/{len(dataset_pairs)}: {ds_s} -> {ds_t}")
    print("="*80)
    
    # Reset random seeds for each dataset pair to ensure reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    py_random.seed(seed)
    
    # Load data
    from scipy.io import loadmat
    mat = loadmat(os.path.join(decaf_dir, ds_t + '_decaf.mat'))
    Xp = mat['feas']
    yp = mat['labels'].flatten()

    Xp = Xp - mean(Xp, axis=0)
    # do PCA on Xp
    from sklearn.decomposition import PCA
    pca = PCA(n_components=100)
    pca.fit(Xp)
    Xp = pca.transform(Xp)
    Xp = Xp / 100

    Xp_tor = torch.tensor(Xp, dtype=torch.float32).to(device)

    mat = loadmat(os.path.join(decaf_dir, ds_s + '_decaf.mat'))
    Xq = mat['feas']
    yq = mat['labels'].flatten()
    Xq = Xq - mean(Xq, axis=0)
    # do PCA on Xq
    Xq = pca.transform(Xq)
    Xq = Xq / 100
    Xq_tor = torch.tensor(Xq, dtype=torch.float32).to(device)

    results = {}
    
    # Save original Xq_tor for fair comparison
    Xq_tor_original = Xq_tor.clone()

    # Calculate number of steps to match kernel method
    # Kernel method: 5 epochs, batch_size=500, each batch updates once
    # So: num_steps = ceil(data_size / batch_size_for_steps) * num_epochs
    batch_size_for_steps = 500  # Used for calculating number of steps (to match kernel method)
    batch_size_for_training = 256  # Used for training classifier
    num_epochs = 5
    data_size = Xq_tor_original.shape[0]
    num_batches_per_epoch = (data_size + batch_size_for_steps - 1) // batch_size_for_steps  # Ceiling division
    num_steps = num_batches_per_epoch * num_epochs
    
    print(f"\nData size: {data_size}, Batches per epoch (for steps calc): {num_batches_per_epoch}, Total steps: {num_steps}")
    print(f"Training batch size: {batch_size_for_training}")
    
    # Fixed parameters
    classifier_hidden = 256
    # Test different regularizer terms (lambda_sobolev) - defined above

    from wgf_da import train_classifier_for_domain_adaptation, apply_gradient_steps, WGF_DomainAdaptation
    from core.nn import DensityRatioClassifier
    
    # ============ Baseline: No adaptation ============
    # Reset random seeds before baseline method
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    py_random.seed(seed)
    
    print("\n" + "="*80)
    print("Baseline: No adaptation (train on source, test on target)")
    print("="*80)
    
    gamma_baseline = 1/comp_dist(Xq_tor_original, Xq_tor_original).flatten().median().item()
    yt_baseline, acc_baseline = svm(Xq_tor_original.cpu(), yq, Xp, yp, gamma=gamma_baseline)
    print(f"Baseline accuracy (no adaptation): {acc_baseline:.4f}")
    
    results['baseline'] = {
        'acc_svm': acc_baseline,
        'avg_gradient_time': 0.0
    }
    
    # ============ Test different sobolev regularization coefficients ============
    for lambda_sobolev in lambda_sobolev_list:
        # Reset random seeds before each lambda test
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        py_random.seed(seed)
        
        print("\n" + "="*80)
        print(f"Testing lambda_sobolev = {lambda_sobolev} (steps = {num_steps}, hidden = {classifier_hidden})")
        print("="*80)
        
        # ============ Train classifier once (WITH Sobolev) ============
        print("\n" + "="*80)
        print(f"Training classifier once WITH Sobolev (hidden={classifier_hidden}, lambda_sobolev={lambda_sobolev})")
        print("="*80)
        
        # Train classifier once and get initial state (WITH Sobolev)
        train_classifier_start_time = time.time()
        classifier_net_sob, initial_yphat_sob, initial_Chinge_sob, initial_Zp_sob, initial_Zq_sob, initial_Xq_sob = \
        train_classifier_for_domain_adaptation(
            Xp_tor, yp, Xq_tor_original, yq, device,
                classifier_hidden=classifier_hidden,
            classifier_lr=2e-4,
                lambda_sobolev=lambda_sobolev,
            use_sobolev=True,
            sobolev_anneal=True,
                sobolev_anneal_steps=100,
                batch_size=batch_size_for_training
        )
        train_classifier_time = time.time() - train_classifier_start_time
        print(f"Classifier training time: {train_classifier_time:.2f} seconds")

        # ============ Classification WITH Sobolev ============
        print("\n" + "="*80)
        print(f"Classification WITH Sobolev: {num_steps} gradient steps (step size: 0.01, hidden={classifier_hidden}, lambda_sobolev={lambda_sobolev})")
        print("="*80)
        
        # Reset to initial state
        Xq_tor = initial_Xq_sob.clone()
        Zp = initial_Zp_sob.clone()
        Zq = initial_Zq_sob.clone()
        # Handle Chinge (might be numpy array or tensor)
        if initial_Chinge_sob is not None:
            if isinstance(initial_Chinge_sob, torch.Tensor):
                Chinge = initial_Chinge_sob.clone()
            else:
                import numpy as np
                Chinge = np.array(initial_Chinge_sob).copy() if isinstance(initial_Chinge_sob, np.ndarray) else initial_Chinge_sob
        else:
            Chinge = None
        
        # Apply gradient steps
        # Note: retrain_classifier=True means retraining at each step (slow but more accurate)
        #       retrain_classifier=False means only forward pass (fast but classifier may become outdated)
        # For fair comparison with kernel method, we use retrain_classifier=True
        result = apply_gradient_steps(
            classifier_net_sob, Xp_tor, yp, Xq_tor, yq, Zp, Zq, Chinge, device,
            num_steps=num_steps, step_size=0.01,
            classifier_lr=2e-4, lambda_sobolev=lambda_sobolev, use_sobolev=True,
            sobolev_anneal=True, sobolev_anneal_steps=100,
            retrain_classifier=True, batch_size=batch_size_for_training,  # Retrain at each step
            adaptive_lambda=False  # Use fixed lambda with annealing
        )
        yphat_class_sob, Xq_T_class_sob, Xq_traj_class_sob, lambda_history_sob, timing_info_sob = result

        acc_proposed_class_sob = mean(yp==yphat_class_sob)
        print(f"OT+SVM accuracy (trained on target domain): {acc_proposed_class_sob:.4f}")

        gamma = 1/comp_dist(Xq_T_class_sob, Xq_T_class_sob).flatten().median().item()
        yt, acc_svm_class_sob = svm(Xq_T_class_sob.cpu(), yq, Xp, yp, gamma=gamma)
        print(f"SVM accuracy (train on adapted source, test on target): {acc_svm_class_sob:.4f}")
        
        # Print timing information (avg gradient time per step)
        # Only count forward+backward time, not finetune time
        # Sobolev advantage: once classifier is trained, gradient = one forward + one backward pass
        avg_gradient_time = timing_info_sob['avg_estimate_gradient_time']
        print(f"Average gradient time per step (forward+backward only): {avg_gradient_time:.4f} seconds")
        print(f"  (Note: finetune time not included, avg finetune per step: {timing_info_sob['avg_finetune_time']:.4f}s)")

        results[f'lambda_sob_{lambda_sobolev}'] = {
            'acc_proposed': acc_proposed_class_sob,
            'acc_svm': acc_svm_class_sob,
            'classifier_hidden': classifier_hidden,
            'lambda_sobolev': lambda_sobolev,
            'lambda_history': lambda_history_sob,
            'avg_gradient_time': avg_gradient_time
        }
        
        # ============ Classification WITH Sobolev (Adaptive Lambda) ============
        # COMMENTED OUT - Adaptive lambda method temporarily disabled
        # # Test adaptive lambda method for all lambda values
        # # Reset random seeds before adaptive lambda method
        # torch.manual_seed(seed)
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed(seed)
        #     torch.cuda.manual_seed_all(seed)
        # random.seed(seed)
        # py_random.seed(seed)
        # 
        # print("\n" + "="*80)
        # print(f"Classification WITH Sobolev (Adaptive Lambda, initial={lambda_sobolev}): {num_steps} gradient steps")
        # print("="*80)
        # 
        # # Train classifier separately for adaptive lambda method (to get independent timing)
        # print("\n" + "="*80)
        # print(f"Training classifier once WITH Sobolev for Adaptive Lambda (hidden={classifier_hidden}, lambda_sobolev={lambda_sobolev})")
        # print("="*80)
        # 
        # train_classifier_start_time_adaptive = time.time()
        # classifier_net_sob_adaptive, initial_yphat_sob_adaptive, initial_Chinge_sob_adaptive, initial_Zp_sob_adaptive, initial_Zq_sob_adaptive, initial_Xq_sob_adaptive = \
        # train_classifier_for_domain_adaptation(
        #     Xp_tor, yp, Xq_tor_original, yq, device,
        #     classifier_hidden=classifier_hidden,
        #     classifier_lr=2e-4,
        #     lambda_sobolev=lambda_sobolev,
        #     use_sobolev=True,
        #     sobolev_anneal=True,
        #     sobolev_anneal_steps=100,
        #     batch_size=batch_size_for_training
        # )
        # train_classifier_time_adaptive = time.time() - train_classifier_start_time_adaptive
        # print(f"Classifier training time (for adaptive lambda): {train_classifier_time_adaptive:.2f} seconds")
        # 
        # # Reset to initial state
        # Xq_tor = initial_Xq_sob_adaptive.clone()
        # Zp = initial_Zp_sob_adaptive.clone()
        # Zq = initial_Zq_sob_adaptive.clone()
        # 
        # # Handle Chinge
        # if initial_Chinge_sob_adaptive is not None:
        #     if isinstance(initial_Chinge_sob_adaptive, torch.Tensor):
        #         Chinge = initial_Chinge_sob_adaptive.clone()
        #     else:
        #         import numpy as np
        #         Chinge = np.array(initial_Chinge_sob_adaptive).copy() if isinstance(initial_Chinge_sob_adaptive, np.ndarray) else initial_Chinge_sob_adaptive
        # else:
        #     Chinge = None
        # 
        # # Apply gradient steps with adaptive lambda
        # result_adaptive = apply_gradient_steps(
        #     classifier_net_sob_adaptive, Xp_tor, yp, Xq_tor, yq, Zp, Zq, Chinge, device,
        #     num_steps=num_steps, step_size=0.01,
        #     classifier_lr=2e-4, lambda_sobolev=lambda_sobolev, use_sobolev=True,
        #     sobolev_anneal=True, sobolev_anneal_steps=100,
        #     retrain_classifier=True, batch_size=batch_size_for_training,
        #     adaptive_lambda=True,  # Use adaptive lambda
        #     initial_lambda=lambda_sobolev,
        #     lambda_min=1e-6, lambda_max=1e-3
        # )
        # yphat_class_sob_adaptive, Xq_T_class_sob_adaptive, Xq_traj_class_sob_adaptive, lambda_history_adaptive, timing_info_adaptive = result_adaptive
        # 
        # acc_proposed_class_sob_adaptive = mean(yp==yphat_class_sob_adaptive)
        # print(f"OT+SVM accuracy (trained on target domain): {acc_proposed_class_sob_adaptive:.4f}")
        # 
        # gamma = 1/comp_dist(Xq_T_class_sob_adaptive, Xq_T_class_sob_adaptive).flatten().median().item()
        # yt, acc_svm_class_sob_adaptive = svm(Xq_T_class_sob_adaptive.cpu(), yq, Xp, yp, gamma=gamma)
        # print(f"SVM accuracy (train on adapted source, test on target): {acc_svm_class_sob_adaptive:.4f}")
        # 
        # if lambda_history_adaptive:
        #     print(f"Lambda history (final): {lambda_history_adaptive[-1]:.2e}, "
        #           f"min: {min(lambda_history_adaptive):.2e}, max: {max(lambda_history_adaptive):.2e}")
        # 
        # # Print timing information
        # total_time_adaptive = train_classifier_time_adaptive + timing_info_adaptive['total_time']
        # print(f"Total time (classifier training + gradient steps): {total_time_adaptive:.2f} seconds")
        # print(f"  - Classifier training: {train_classifier_time_adaptive:.2f} seconds")
        # print(f"  - Gradient steps: {timing_info_adaptive['total_time']:.2f} seconds")
        # print(f"  - Average gradient computation time per step: {timing_info_adaptive['avg_gradient_time']:.4f} seconds")
        # print(f"  - Number of gradient steps: {num_steps}")
        # 
        # results[f'lambda_sob_adaptive_{lambda_sobolev}'] = {
        #     'acc_proposed': acc_proposed_class_sob_adaptive,
        #     'acc_svm': acc_svm_class_sob_adaptive,
        #     'num_steps': num_steps,
        #     'classifier_hidden': classifier_hidden,
        #     'initial_lambda': lambda_sobolev,
        #     'lambda_history': lambda_history_adaptive,
        #     'train_classifier_time': train_classifier_time_adaptive,
        #     'gradient_steps_time': timing_info_adaptive['total_time'],
        #     'total_time': total_time_adaptive,
        #     'avg_gradient_time': timing_info_adaptive['avg_gradient_time'],
        #     'gradient_times': timing_info_adaptive['gradient_times']
        # }

    # ============ Train classifier once (WITHOUT Sobolev) ============
    # COMMENTED OUT - Only testing Baseline and Sobolev methods
    # print("\n" + "="*80)
    # print(f"Training classifier once WITHOUT Sobolev (hidden={classifier_hidden})")
    # print("="*80)
    # 
    # # Train classifier once and get initial state (WITHOUT Sobolev)
    # classifier_net_nosob, initial_yphat_nosob, initial_Chinge_nosob, initial_Zp_nosob, initial_Zq_nosob, initial_Xq_nosob = \
    #     train_classifier_for_domain_adaptation(
    #         Xp_tor, yp, Xq_tor_original, yq, device,
    #         classifier_hidden=classifier_hidden,
    #         classifier_lr=2e-4,
    #         lambda_sobolev=1e-4,
    #         use_sobolev=False,
    #         sobolev_anneal=True,
    #         sobolev_anneal_steps=100,
    #         batch_size=batch_size_for_training
    #     )

    # ============ Classification WITHOUT Sobolev ============
    # COMMENTED OUT - Only testing Baseline and Sobolev methods
    # print("\n" + "="*80)
    # print(f"Classification WITHOUT Sobolev: {num_steps} gradient steps (step size: 0.01, hidden={classifier_hidden})")
    # print("="*80)
    # 
    # # Reset to initial state
    # Xq_tor = initial_Xq_nosob.clone()
    # Zp = initial_Zp_nosob.clone()
    # Zq = initial_Zq_nosob.clone()
    # # Handle Chinge (might be numpy array or tensor)
    # if initial_Chinge_nosob is not None:
    #     if isinstance(initial_Chinge_nosob, torch.Tensor):
    #         Chinge = initial_Chinge_nosob.clone()
    #     else:
    #         import numpy as np
    #         Chinge = np.array(initial_Chinge_nosob).copy() if isinstance(initial_Chinge_nosob, np.ndarray) else initial_Chinge_nosob
    # else:
    #     Chinge = None
    # 
    # # Apply gradient steps (retrain classifier at each step)
    # result = apply_gradient_steps(
    #     classifier_net_nosob, Xp_tor, yp, Xq_tor, yq, Zp, Zq, Chinge, device,
    #     num_steps=num_steps, step_size=0.01,
    #     classifier_lr=2e-4, lambda_sobolev=1e-4, use_sobolev=False,
    #     sobolev_anneal=True, sobolev_anneal_steps=100,
    #     retrain_classifier=True, batch_size=batch_size_for_training,
    #     adaptive_lambda=False  # Not using Sobolev, so adaptive lambda not applicable
    # )
    # yphat_class_nosob, Xq_T_class_nosob, Xq_traj_class_nosob, lambda_history_nosob = result
    # 
    # acc_proposed_class_nosob = mean(yp==yphat_class_nosob)
    # print(f"OT+SVM accuracy (trained on target domain): {acc_proposed_class_nosob:.4f}")
    # 
    # gamma = 1/comp_dist(Xq_T_class_nosob, Xq_T_class_nosob).flatten().median().item()
    # yt, acc_svm_class_nosob = svm(Xq_T_class_nosob.cpu(), yq, Xp, yp, gamma=gamma)
    # print(f"SVM accuracy (train on adapted source, test on target): {acc_svm_class_nosob:.4f}")
    # 
    # results['classification_without_sob'] = {
    #     'acc_proposed': acc_proposed_class_nosob,
    #     'acc_svm': acc_svm_class_nosob,
    #     'num_steps': num_steps,
    #     'classifier_hidden': classifier_hidden
    # }

    # ============ Kernel method (Local Linear) - 5 epochs ============
    # Reset random seeds before kernel method
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    py_random.seed(seed)
    
    print("\n" + "="*80)
    print("Kernel method (Local Linear): 5 epochs")
    print("="*80)
    
    # Reset to original source data for kernel method
    Xq_tor_kernel = Xq_tor_original.clone()
    
    # Apply kernel method using WGF_DomainAdaptation with timing
    result_kernel = WGF_DomainAdaptation(
        Xp_tor, yp, Xq_tor_kernel, yq, kernel, 
        nepoch=5, 
        VGD_batchsize=500, 
        device=device,
        grad_method='kernel'
    )
    
    # Handle return value (with or without timing_info)
    if len(result_kernel) == 4:
        yphat_kernel, Xq_T_kernel, Xq_traj_kernel, kernel_timing_info = result_kernel
    else:
        yphat_kernel, Xq_T_kernel, Xq_traj_kernel = result_kernel
        kernel_timing_info = None
    
    acc_proposed_kernel = mean(yp==yphat_kernel)
    print(f"OT+SVM accuracy (trained on target domain): {acc_proposed_kernel:.4f}")

    gamma_kernel = 1/comp_dist(Xq_T_kernel, Xq_T_kernel).flatten().median().item()
    yt_kernel, acc_svm_kernel = svm(Xq_T_kernel.cpu(), yq, Xp, yp, gamma=gamma_kernel)
    print(f"SVM accuracy (train on adapted source, test on target): {acc_svm_kernel:.4f}")
    
    avg_kernel_time = 0.0
    if kernel_timing_info:
        avg_kernel_time = kernel_timing_info['avg_kernel_time']
        print(f"Average gradient time per step (kernel training + compute): {avg_kernel_time:.4f} seconds")
    
    results['kernel'] = {
        'acc_proposed': acc_proposed_kernel,
        'acc_svm': acc_svm_kernel,
        'nepoch': 5,
        'avg_gradient_time': avg_kernel_time
    }
    
    # Store results for this pair
    all_results[f"{ds_s}->{ds_t}"] = results

# ============ Overall Summary ============
print("\n" + "="*80)
print("OVERALL SUMMARY: Sobolev Regularization Coefficient Comparison (dynamic steps, hidden=256)")
print("="*80)

# Use the same lambda_sobolev_list defined above

# Summary for each lambda_sobolev
for lambda_sobolev in lambda_sobolev_list:
    print(f"\n{'='*80}")
    print(f"Lambda Sobolev: {lambda_sobolev} (steps = dynamic, hidden = 256)")
    print(f"{'='*80}")
    print(f"\n{'Dataset Pair':<25} {'SVM Acc':<12}")
    print("-"*40)
    
    lambda_results = []
    for pair_name, pair_results in all_results.items():
        lambda_key = f'lambda_sob_{lambda_sobolev}'
        if lambda_key in pair_results:
            svm_acc = pair_results[lambda_key]['acc_svm']
            lambda_results.append(svm_acc)
            print(f"{pair_name:<25} {svm_acc:<12.4f}")
    
    if lambda_results:
        avg_lambda = mean(lambda_results)
        print("-"*40)
        print(f"{'Average':<25} {avg_lambda:<12.4f}")

# Final comparison table
print("\n" + "="*80)
print("FINAL COMPARISON: Average Performance by Lambda Sobolev (dynamic steps matching kernel method, hidden=256)")
print("="*80)
print(f"\n{'Lambda Sobolev':<20} {'Avg SVM Acc':<15}")
print("-"*40)

lambda_averages = {}
for lambda_sobolev in lambda_sobolev_list:
    lambda_results = []
    for pair_name, pair_results in all_results.items():
        lambda_key = f'lambda_sob_{lambda_sobolev}'
        if lambda_key in pair_results:
            lambda_results.append(pair_results[lambda_key]['acc_svm'])
    
    if lambda_results:
        avg_lambda = mean(lambda_results)
        lambda_averages[lambda_sobolev] = avg_lambda
        print(f"{lambda_sobolev:<20} {avg_lambda:<15.4f}")

# Find best lambda_sobolev
if lambda_averages:
    best_lambda = max(lambda_averages.keys(), key=lambda k: lambda_averages[k])
    print(f"\nBest Lambda Sobolev: {best_lambda} (Avg SVM Acc: {lambda_averages[best_lambda]:.4f})")

# ============ All Methods Comparison ============
print("\n" + "="*80)
print("ALL METHODS COMPARISON (Baseline + Sobolev: Fixed Lambda + Kernel)")
print("="*80)

# Summary for all methods (baseline, sobolev, and kernel methods)
methods = ['baseline', 'kernel']
# Add fixed lambda methods
for lambda_val in lambda_sobolev_list:
    methods.append(f'lambda_sob_{lambda_val}')
# Add adaptive lambda methods (for all lambda values) - COMMENTED OUT
# for lambda_val in lambda_sobolev_list:
#     methods.append(f'lambda_sob_adaptive_{lambda_val}')

print(f"\n{'Dataset Pair':<25}", end='')
for method in methods:
    method_name = method.replace('lambda_sob_', 'λ=').replace('classification_without_sob', 'Class w/o Sob').replace('_', ' ').title()
    print(f"{method_name:<25}", end='')
print()

print("-" * (25 + 25 * len(methods)))

method_averages = {method: [] for method in methods}
for pair_name, pair_results in all_results.items():
    print(f"{pair_name:<25}", end='')
    for method in methods:
        if method in pair_results:
            acc = pair_results[method]['acc_svm']
            method_averages[method].append(acc)
            print(f"{acc:<25.4f}", end='')
        else:
            print(f"{'N/A':<25}", end='')
    print()

print("-" * (25 + 25 * len(methods)))
print(f"{'Average':<25}", end='')
for method in methods:
    if method_averages[method]:
        avg = mean(method_averages[method])
        print(f"{avg:<25.4f}", end='')
    else:
        print(f"{'N/A':<25}", end='')
print()

# ============ Timing Statistics ============
print("\n" + "="*80)
print("TIMING STATISTICS: Average Gradient Time")
print("="*80)

# Gradient time comparison by dataset pair
print("\n" + "="*80)
print("GRADIENT TIME BY DATASET PAIR")
print("="*80)
print(f"\n{'Method':<35} {'Avg Gradient Time (s)':<25}")
print("-" * 60)

gradient_times_summary = []
for pair_name, pair_results in all_results.items():
    print(f"\n{pair_name}:")
    for method in methods:
        if method in pair_results:
            result = pair_results[method]
            method_name = method.replace('lambda_sob_', 'λ=').replace('classification_without_sob', 'Class w/o Sob').replace('_', ' ').title()
            avg_grad_time = result.get('avg_gradient_time', 0.0)
            print(f"{method_name:<35} {avg_grad_time:<25.4f}")
            gradient_times_summary.append((method, avg_grad_time))

# Summary across all pairs
if len(all_results) > 1:
    print("\n" + "="*80)
    print("AVERAGE GRADIENT TIME ACROSS ALL DATASET PAIRS")
    print("="*80)
    print(f"\n{'Method':<35} {'Avg Gradient Time (s)':<25}")
    print("-" * 60)
    
    method_avg_times = {}
    for method_name, grad_time in gradient_times_summary:
        if method_name not in method_avg_times:
            method_avg_times[method_name] = []
        method_avg_times[method_name].append(grad_time)
    
    for method_name, times in method_avg_times.items():
        method_label = method_name.replace('lambda_sob_', 'λ=').replace('classification_without_sob', 'Class w/o Sob').replace('_', ' ').title()
        avg_time = mean(times)
        print(f"{method_label:<35} {avg_time:<25.4f}")

# Save all results
result_dir = os.path.join(script_dir, 'result')
os.makedirs(result_dir, exist_ok=True)
result_path = os.path.join(result_dir, 'results_baseline_sob_kernel_1pair_gradtime.pt')
torch.save(all_results, result_path)
print(f"\nResults saved to: {result_path}")
print("="*80)
# %%
