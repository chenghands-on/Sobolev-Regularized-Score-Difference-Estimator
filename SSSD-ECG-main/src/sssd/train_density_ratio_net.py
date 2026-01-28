
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from density_ratio_guidance import Bandit_Critic_Guide

import json
import numpy as np
import os
import argparse

import sys
sys.path.append('utils')
from util_generation import bandit_get_args, calc_diffusion_hyperparams, std_normal
from load_ptbxl import load_ptbxl_data
from load_icbeb import load_icbeb_data

import math
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import time
from datetime import timedelta
import random


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    This sets seeds for Python random, NumPy, PyTorch (CPU and CUDA), and CUDNN.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed} for reproducibility")

# Parse command line arguments for mode selection
parser = argparse.ArgumentParser(description='Train density ratio network (classifier)')
parser.add_argument('--lambda_grad', type=float, default=5.0, 
                    help='Maximum weight for Sobolev penalty (default: 5.0, will be scheduled)')
parser.add_argument('--use_sobolev', action='store_true',
                    help='Use Sobolev penalty in standard guidance mode')
parser.add_argument('--no_decay', action='store_true',
                    help='If set, keep Sobolev lambda at 5.0. Otherwise, decay from 5.0 to 1e-2 (default: decay enabled)')

parser.add_argument('--device', type=str, default='cuda:1',
                    help='Main device to use (default: cuda:1). If multiple GPUs specified, will use DataParallel.')
parser.add_argument('--gpu_ids', type=str, default=None,
                    help='Comma-separated GPU IDs to use for multi-GPU training (e.g., "1,2,3"). If not specified, uses single GPU with --device.')

parser.add_argument('--consistence_weight', type=float, default=0.0,
                    help='Weight for consistence regularization (default: 0.0, disabled to keep original behavior). Set to 0.1-0.5 for improved version.')
parser.add_argument('--improved_schedule', action='store_true',
                    help='If set, use improved schedule with T_decay1=1500 instead of original 800 (default: False, keeps original schedule)')
parser.add_argument('--data_noise_scale', type=float, default=0.0,
                    help='Scale of additional noise to add to training data (default: 0.0, disabled). Higher values (0.02-0.05) increase noise and help Sobolev regularization.')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='Weight decay (L2 regularization) for optimizer (default: 0.0, promotes overfitting). Set to 1e-5 to 1e-3 to prevent overfitting.')
parser.add_argument('--experiment_name', type=str, default=None,
                    help='Experiment name suffix to avoid overwriting results. If None, uses default naming. (default: None)')
parser.add_argument('--total_iters', type=int, default=1000,
                    help='Total training iterations (default: 1000)')
args_cmd = parser.parse_args()

# Fixed seed for reproducibility
SEED = 42
set_seed(SEED)

# Define total_iterations early for use in model initialization
total_iterations = args_cmd.total_iters  # Total training iterations

args = bandit_get_args()

# Parse GPU IDs for multi-GPU training
if args_cmd.gpu_ids:
    gpu_ids = [int(x.strip()) for x in args_cmd.gpu_ids.split(',')]
    if len(gpu_ids) > 1:
        # Multi-GPU mode
        args.device = f'cuda:{gpu_ids[0]}'  # Main device
        use_multi_gpu = True
        print(f"Multi-GPU training enabled. Using GPUs: {gpu_ids}")
        print(f"Main device: {args.device}")
    else:
        # Single GPU mode (only one GPU specified in gpu_ids)
        args.device = f'cuda:{gpu_ids[0]}'
        use_multi_gpu = False
        gpu_ids = None  # Clear gpu_ids for single GPU mode
        print(f"Single-GPU training. Using device: {args.device}")
else:
    # No gpu_ids specified, use --device argument
    args.device = args_cmd.device
    use_multi_gpu = False
    gpu_ids = None
    print(f"Single-GPU training. Using device: {args.device}")

# Initialize guidance network based on mode
# Model prefix naming: clf (naive), clf_sob (Sobolev no decay), clf_sobd (Sobolev decay - default)
if args_cmd.use_sobolev:
    print("Using Bandit_Critic_Guide mode with Sobolev penalty")
    guidance_net = Bandit_Critic_Guide(3904, 0, args, weight_decay=args_cmd.weight_decay)
    # Add suffix based on no_decay option (decay is now default)
    decay_to_min = not args_cmd.no_decay  # Default: decay enabled
    if decay_to_min:
        model_prefix = 'clf_sobd'  # Sobolev with decay (default)
        # Use actual total_iterations to determine decay steps
        T_decay1 = 1500 if args_cmd.improved_schedule else 800
        decay_end = min(T_decay1, total_iterations)
        decay_steps = f"500-{decay_end}"
        print(f"  Sobolev lambda will decay from {args_cmd.lambda_grad} to 1e-2 ({decay_steps} steps)")
    else:
        model_prefix = 'clf_sob'  # Sobolev without decay
        print(f"  Sobolev lambda will stay at {args_cmd.lambda_grad} (no decay)")
    if args_cmd.consistence_weight > 0:
        print(f"  Consistency loss enabled with weight: {args_cmd.consistence_weight}")
    if args_cmd.improved_schedule:
        print("  Using improved schedule (T_decay1=1500)")
    else:
        print("  Using original schedule (T_decay1=800)")
    if args_cmd.data_noise_scale > 0:
        print(f"  Additional data noise enabled with scale: {args_cmd.data_noise_scale}")
    if args_cmd.weight_decay > 0:
        print(f"  Weight decay enabled: {args_cmd.weight_decay}")
    else:
        print("  Weight decay disabled (promotes overfitting for Sobolev regularization)")
else:
    print("Using Bandit_Critic_Guide mode (default)")
    guidance_net = Bandit_Critic_Guide(3904, 0, args)
    model_prefix = 'clf'  # Naive classifier

X_train_PTBXL, X_val_PTBXL, X_test_PTBXL, y_train_PTBXL, y_val_PTBXL, y_test_PTBXL = load_ptbxl_data()
X_train_ICBEB, X_val_ICBEB, X_test_ICBEB, y_train_ICBEB, y_val_ICBEB, y_test_ICBEB = load_icbeb_data()

X_train_ICBEB = X_test_ICBEB
y_train_ICBEB = y_test_ICBEB

label_map = [46, 4, 0, 11, 12, 49, 54, 63, 64]
# first filter out the corresponding labeled data
indices = (y_train_PTBXL[:, label_map] == 1).any(axis=1)

# Filter data and labels based on indices
X_train_PTBXL = X_train_PTBXL[indices]
y_train_PTBXL = y_train_PTBXL[indices]

# Standardize the data of icbeb
indices = [i for i, d in enumerate(X_train_ICBEB) if d.shape[0] >= 1000]
X_train_ICBEB = [d[:1000,:] for d in X_train_ICBEB if d.shape[0] >= 1000]
y_train_ICBEB = y_train_ICBEB[indices]


# Standardize label of icbeb 
new_labels = np.zeros((y_train_ICBEB.shape[0], 71))

# Iterate through each example and update new_labels
for i in range(y_train_ICBEB.shape[0]):
    for j in range(y_train_ICBEB.shape[1]):
        new_labels[i, label_map[j]] = y_train_ICBEB[i, j]

y_train_ICBEB = new_labels
data_ptbxl = np.transpose(X_train_PTBXL,(0,2,1))

# Print shapes (X_train_ICBEB is a list, not numpy array)
print(f"X_train_ICBEB: {len(X_train_ICBEB)} samples (list)")
if len(X_train_ICBEB) > 0:
    print(f"  First sample shape: {X_train_ICBEB[0].shape}")
print(f"X_train_PTBXL shape: {X_train_PTBXL.shape}")
print(f"y_train_ICBEB shape: {y_train_ICBEB.shape}")
print(f"y_train_PTBXL shape: {y_train_PTBXL.shape}")


''' class balance processing'''
train_data = []
for i in range(len(data_ptbxl)):
    train_data.append([data_ptbxl[i], y_train_PTBXL[i]])

trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=128, drop_last=True)

    
# training
n_iter = 1
start_time = time.time()
iter_times = []  # Store time for each iteration to calculate average

print(f"Starting training for {total_iterations} iterations ...")
print("=" * 80)

while n_iter < total_iterations + 1:
    for audio, label in trainloader:
        # Record start time for this batch/iteration
        iter_start_time = time.time()
        
        # we left the selection for diffusion model, we need to use 12 leads for classifier
        audio = audio.float().to(args.device)  # Move audio to device
        label = label.float().to(args.device)
        
        # Verbose mode for first 100 steps and every 100 steps after that
        verbose_lambda = args_cmd.use_sobolev and (n_iter <= 100 or n_iter % 100 == 0)
        decay_to_min = not args_cmd.no_decay if args_cmd.use_sobolev else False
        loss = guidance_net.update_qt(
            audio, label,
            lambda_grad=args_cmd.lambda_grad if args_cmd.use_sobolev else 0.0,
            step=n_iter,
            use_sobolev=args_cmd.use_sobolev,
            decay_to_min=decay_to_min,
            verbose=verbose_lambda,
            joint_gradient=False,  # Always False now
            consistency_weight=args_cmd.consistence_weight,
            improved_schedule=args_cmd.improved_schedule,
            data_noise_scale=args_cmd.data_noise_scale,
            total_iterations=total_iterations
        )
        
        # Calculate iteration time (time for this single batch)
        iter_time = time.time() - iter_start_time
        iter_times.append(iter_time)
        
        # Progress display every 20 iterations
        if n_iter % 20 == 0 or n_iter == 1:
            # Calculate statistics
            avg_iter_time = sum(iter_times[-100:]) / len(iter_times[-100:]) if len(iter_times) > 0 else iter_time  # Average of last 100 iterations
            elapsed_time = time.time() - start_time
            remaining_iterations = total_iterations - n_iter
            estimated_remaining_time = avg_iter_time * remaining_iterations
            
            # Get loss value (handle both tensor and numpy array)
            if isinstance(loss, torch.Tensor):
                loss_value = loss.item()
            else:
                loss_value = float(loss) if isinstance(loss, np.ndarray) else loss
            
            # Progress display
            progress = (n_iter / total_iterations) * 100
            print(f"[{n_iter}/{total_iterations}] ({progress:.1f}%) | "
                  f"Loss: {loss_value:.6f} | "
                  f"Elapsed: {timedelta(seconds=int(elapsed_time))} | "
                  f"ETA: {timedelta(seconds=int(estimated_remaining_time))} | "
                  f"Avg iter time: {avg_iter_time:.3f}s")

        # save checkpoint (save at end for small iterations, every 1000 for full training)
        save_interval = total_iterations if total_iterations <= 100 else 1000
        if n_iter > 0 and (n_iter % save_interval == 0 or n_iter == total_iterations):
            # Add experiment name suffix if provided
            if args_cmd.experiment_name:
                checkpoint_name = '{}_{}_{}.pkl'.format(model_prefix, args_cmd.experiment_name, n_iter)
            else:
                checkpoint_name = '{}_{}.pkl'.format(model_prefix, n_iter)
            save_dir = "Diffusion_RL/ecg_ptbxl_benchmarking/model_weight/"
            os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists
            save_path = os.path.join(save_dir, checkpoint_name)
            
            # Save different state dicts based on mode
            # Handle DataParallel: get underlying model if wrapped
            model_to_save = guidance_net.qt.module if isinstance(guidance_net.qt, torch.nn.DataParallel) else guidance_net.qt
            
            torch.save({
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': guidance_net.qt_optimizer.state_dict()
            }, save_path)
            
            print('=' * 80)
            print(f'Model at iteration {n_iter} is saved to: {save_path}')
            print('=' * 80)

        n_iter += 1
        
        if n_iter > total_iterations:
            break

print("=" * 80)
print(f"Training completed! Total time: {timedelta(seconds=int(time.time() - start_time))}")
