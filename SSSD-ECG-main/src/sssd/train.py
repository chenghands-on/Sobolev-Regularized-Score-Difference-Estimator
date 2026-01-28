import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import random

from utils.util_generation import find_max_epoch, print_size, training_loss_label, calc_diffusion_hyperparams
from models_new.SSSD_ECG import SSSD_ECG


import sys
sys.path.append("Diffusion_RL/ecg_ptbxl_benchmarking/code/models")
sys.path.append("Diffusion_RL/ecg_ptbxl_benchmarking/code/")
from load_icbeb import load_icbeb_data


# Fixed seed for reproducibility
SEED = 42

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


def train(output_directory,
          ckpt_iter,
          n_iters,
          iters_per_ckpt,
          iters_per_logging,
          learning_rate,
         batch_size,
         use_pretrained=True,
         experiment_name=None):
  
    """
    Train Diffusion Models

    Parameters:
    output_directory (str):         save model checkpoints to this path
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded; 
                                    automatically selects the maximum iteration if 'max' is selected
    data_path (str):                path to dataset, numpy array.
    n_iters (int):                  number of iterations to train
    iters_per_ckpt (int):           number of iterations to save checkpoint, 
    iters_per_logging (int):        number of iterations to save training log and compute validation loss, default is 100
    learning_rate (float):          learning rate
    """
    
    # Declare global variables at the beginning of the function
    global model_config, diffusion_config, diffusion_hyperparams

    # generate experiment (local) path
    local_path = "ch{}_T{}_betaT{}".format(model_config["res_channels"], 
                                           diffusion_config["T"], 
                                           diffusion_config["beta_T"])

    # Get shared output_directory ready (for other outputs, not model weights)
    output_directory = os.path.join(output_directory, local_path)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    # Use unified model weight directory (same as inference_guided_ptbxl.py)
    # Get script directory to resolve relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_weight_dir = os.path.join(script_dir, 'Diffusion_RL', 'ecg_ptbxl_benchmarking', 'model_weight')
    if not os.path.isdir(model_weight_dir):
        os.makedirs(model_weight_dir)
        os.chmod(model_weight_dir, 0o775)
    print("model weight directory", model_weight_dir, flush=True)

    # Set random seed first, before any random operations
    set_seed(SEED)
    
    # Set device (can be overridden by environment variable CUDA_DEVICE or default to cuda:5)
    device_str = os.getenv('CUDA_DEVICE', 'cuda:5')
    device = torch.device(device_str)
    print(f'Using device: {device}', flush=True)

    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].to(device)
            
    # predefine model
    net = SSSD_ECG(**model_config).to(device)

    # define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Helper function to find max epoch with diff_ prefix
    def find_max_diff_epoch(path, exp_name=None, scratch=False):
        """Find maximum epoch in diff_*.pkl files, optionally with experiment_name"""
        files = os.listdir(path)
        epoch = -1
        prefix = 'diff_scratch_' if scratch else 'diff_'
        
        for f in files:
            if not f.startswith(prefix) or not f.endswith('.pkl'):
                continue
            
            # Remove prefix to get remaining part
            remaining = f[len(prefix):-4]
            
            # If experiment_name is specified, check if it matches
            if exp_name:
                # Pattern: diff_{exp_name}_{number}.pkl or diff_scratch_{exp_name}_{number}.pkl
                if remaining.startswith(exp_name + '_'):
                    try:
                        epoch_str = remaining[len(exp_name) + 1:]  # Skip exp_name and underscore
                        epoch = max(epoch, int(epoch_str))
                    except:
                        continue
            else:
                # No experiment_name: look for files without experiment_name
                # Pattern: diff_{number}.pkl or diff_scratch_{number}.pkl
                # Check if remaining part is just a number (no underscores except in exp_name)
                try:
                    # Try to parse as integer directly
                    epoch = max(epoch, int(remaining))
                except:
                    # If it contains underscores, it might have experiment_name, skip it
                    if '_' not in remaining:
                        try:
                            epoch = max(epoch, int(remaining))
                        except:
                            continue
        return epoch
    
    # load checkpoint from unified model weight directory (only if use_pretrained=True)
    if use_pretrained:
        if ckpt_iter == 'max':
            # First try to find diff_*.pkl files with experiment_name, then without
            if experiment_name:
                ckpt_iter = find_max_diff_epoch(model_weight_dir, exp_name=experiment_name, scratch=False)
                if ckpt_iter < 0:
                    ckpt_iter = find_max_diff_epoch(model_weight_dir, exp_name=None, scratch=False)
            else:
                ckpt_iter = find_max_diff_epoch(model_weight_dir, exp_name=None, scratch=False)
            
            if ckpt_iter < 0:
                # Fall back to regular *.pkl files
                ckpt_iter = find_max_epoch(model_weight_dir)
        
        if ckpt_iter >= 0:
            # Try loading with experiment_name first, then fall back
            model_path = None
            if experiment_name:
                # Try: diff_{experiment_name}_{ckpt_iter}.pkl
                model_path = os.path.join(model_weight_dir, 'diff_{}_{}.pkl'.format(experiment_name, ckpt_iter))
                if not os.path.exists(model_path):
                    # Fall back to: diff_{ckpt_iter}.pkl
                    model_path = os.path.join(model_weight_dir, 'diff_{}.pkl'.format(ckpt_iter))
            else:
                # Try: diff_{ckpt_iter}.pkl
                model_path = os.path.join(model_weight_dir, 'diff_{}.pkl'.format(ckpt_iter))
            
            if not os.path.exists(model_path):
                # Final fall back to regular name for backward compatibility
                model_path = os.path.join(model_weight_dir, '{}.pkl'.format(ckpt_iter))
            
            try:
                checkpoint = torch.load(model_path, map_location='cpu')

                # feed model dict and optimizer state
                net.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                print('Successfully loaded model at iteration {} from {}'.format(ckpt_iter, model_path))
            except Exception as e:
                ckpt_iter = -1
                print(f'No valid checkpoint model found at {model_path}: {e}')
                print('Start training from initialization.')
        else:
            ckpt_iter = -1
            print('No valid checkpoint model found, start training from initialization.')
    else:
        print('Training from scratch (no pretrained model).')
        ckpt_iter = -1
        
        
    # Since I am going to load 690 the training samples, therefore I use this (normalized data)
    X_train_ICBEB, X_val_ICBEB, X_test_ICBEB, y_train_ICBEB, y_val_ICBEB, y_test_ICBEB = load_icbeb_data()
    data_ptbxl = X_test_ICBEB
    labels_ptbxl = y_test_ICBEB

    # --- Force 71-dim mapping ---
    label_map = [46, 4, 0, 11, 12, 49, 54, 63, 64]
    
    # Force create 71-dim label matrix (matching PTB-XL label space)
    print(f"Standardizing ICBEB labels to 71-dimensional space (Original Logic)")
    new_labels = np.zeros((labels_ptbxl.shape[0], 71))
    
    # Fill ICBEB's 9 class labels into the specified 71 index positions
    for i in range(labels_ptbxl.shape[0]):
        for j in range(labels_ptbxl.shape[1]):
            new_labels[i, label_map[j]] = labels_ptbxl[i, j]
    
    labels_ptbxl = new_labels
    # --- Mapping end ---


    data_ptbxl_all = []
    for i in data_ptbxl:
        if i.shape[0]>=1000:
            data_ptbxl_all.append(i[:1000])
        else:
            exit(1)
    data_ptbxl_all = np.array(data_ptbxl_all)
    data_ptbxl = np.transpose(data_ptbxl_all,(0,2,1))
    
    
    train_data = []
    for i in range(len(data_ptbxl)):
        train_data.append([data_ptbxl[i], labels_ptbxl[i]])
        
    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=8, drop_last=True)
       
    index_8 = torch.tensor([0,2,3,4,5,6,7,11]).to(device)
    index_4 = torch.tensor([1,8,9,10]).to(device)
    
    
    # training
    # Start from iteration 0 (weight loading is independent of iteration counting)
    n_iter = 0
    
    while n_iter < n_iters + 1:
        for audio, label in trainloader:
            
            # Move audio to device first, then select indices
            audio = audio.to(device)
            audio = torch.index_select(audio, 1, index_8).float()
            label = label.to(device).float()
           
            
            # back-propagation
            optimizer.zero_grad()
            X = audio, label
            
            loss = training_loss_label(net, nn.MSELoss(), X, diffusion_hyperparams)

            loss.backward()
            optimizer.step()

            if n_iter % iters_per_logging == 0:
                print("iteration: {} \tloss: {}".format(n_iter, loss.item()))

            # save checkpoint to unified model weight directory
            # Use "diff_" prefix to avoid conflicts with other models
            # Add "scratch" suffix if training from scratch
            # Add experiment_name if provided
            if n_iter > 0 and n_iter % iters_per_ckpt == 0:
                if use_pretrained:
                    if experiment_name:
                        checkpoint_name = 'diff_{}_{}.pkl'.format(experiment_name, n_iter)
                    else:
                        checkpoint_name = 'diff_{}.pkl'.format(n_iter)
                else:
                    if experiment_name:
                        checkpoint_name = 'diff_scratch_{}_{}.pkl'.format(experiment_name, n_iter)
                    else:
                        checkpoint_name = 'diff_scratch_{}.pkl'.format(n_iter)
                checkpoint_path = os.path.join(model_weight_dir, checkpoint_name)
                torch.save({'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           checkpoint_path)
                print('model at iteration %s is saved to %s' % (n_iter, checkpoint_path))

            n_iter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/config_SSSD_ECG.json',
                        help='JSON file for configuration')
    parser.add_argument('--no_pretrained', action='store_true',
                        help='Train from scratch without loading pretrained model. Saved models will have "scratch" suffix.')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name suffix to avoid overwriting model files. If None, uses default naming. (default: None)')

    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()

    config = json.loads(data)
    print(config)
    
    train_config = config["train_config"]  # training parameters

    global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset

    global diffusion_config
    diffusion_config = config["diffusion_config"]  # basic hyperparameters

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)  # dictionary of all diffusion hyperparameters

    global model_config
    model_config = config['wavenet_config']

    # Add use_pretrained flag from command line argument
    train_config['use_pretrained'] = not args.no_pretrained
    # Add experiment_name from command line argument
    train_config['experiment_name'] = args.experiment_name

    train(**train_config)
