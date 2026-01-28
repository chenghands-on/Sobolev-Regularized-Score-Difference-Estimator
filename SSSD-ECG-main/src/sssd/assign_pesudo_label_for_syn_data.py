# from SNIP_for_reward import load_pretrained_model
import pickle
import torch
import numpy as np
import os
import sys

# Add paths based on script location, not current working directory
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Script is in sssd/, so Diffusion_RL/ is a subdirectory
models_path = os.path.join(script_dir, 'Diffusion_RL', 'ecg_ptbxl_benchmarking', 'code', 'models')
code_path = os.path.join(script_dir, 'Diffusion_RL', 'ecg_ptbxl_benchmarking', 'code')

# Add to path if they exist
if os.path.exists(models_path):
    sys.path.insert(0, models_path)
if os.path.exists(code_path):
    sys.path.insert(0, code_path)

from util_utility import utils
from util_utility.utils import apply_standardizer

import ipdb




def load_pretrained_classifier_for_label_inference():
    """
    Load pretrained classifier model (100000.pkl) with original 71 classes for label inference.
    This is different from load_pretrained_model() which loads a 2-class model for density ratio estimation.
    
    Returns:
        fastai_model: Model with 71 classes output (for multi-class classification)
    """
    from models.fastai_model import fastai_model
    
    # Get script directory to resolve relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_weight_dir = os.path.join(script_dir, 'Diffusion_RL', 'ecg_ptbxl_benchmarking', 'model_weight')
    
    # Essential parameters
    modelname = 'fastai_xresnet1d50'
    pretrained_file = os.path.join(model_weight_dir, '100000.pkl')
    outputfolder = model_weight_dir + '/'
    num_classes = 71  # Keep original 71 classes for label inference
    sampling_frequency = 100
    input_shape = [1000, 12]

    # Load model without replacing output layer (keep 71 classes for label inference)
    model = fastai_model(
        name=modelname,
        n_classes=num_classes,
        freq=sampling_frequency,
        outputfolder=outputfolder,
        input_shape=input_shape,
        pretrained=True,
        pretrained_file=pretrained_file,
        n_classes_pretrained=num_classes,
        replace_output_layer=False,  # Keep original output layer (71 classes)
        epochs_finetuning=0,  # No fine-tuning needed
    )
    return model


def infer_label(syn_x):
    """
    Infer labels for synthetic data using the original 71-class classifier.
    This uses the multi-class model (71 classes), not the 2-class density ratio model.
    Directly loads weights from 100000.pkl to avoid loading saved 2-class models.
    """
    from models.fastai_model import fastai_model
    from fastai.torch_core import to_np
    
    # Get script directory to resolve relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_weight_dir = os.path.join(script_dir, 'Diffusion_RL', 'ecg_ptbxl_benchmarking', 'model_weight')
    
    # Essential parameters
    modelname = 'fastai_xresnet1d50'
    pretrained_file = os.path.join(model_weight_dir, '100000.pkl')
    outputfolder = model_weight_dir + '/'
    num_classes = 71  # Keep original 71 classes for label inference
    sampling_frequency = 100
    input_shape = [1000, 12]

    # Load model without replacing output layer (keep 71 classes for label inference)
    model = fastai_model(
        name=modelname,
        n_classes=num_classes,
        freq=sampling_frequency,
        outputfolder=outputfolder,
        input_shape=input_shape,
        pretrained=True,
        pretrained_file=pretrained_file,
        n_classes_pretrained=num_classes,
        replace_output_layer=False,  # Keep original output layer (71 classes)
        epochs_finetuning=0,  # No fine-tuning needed
    )
    
    # Prepare data: convert to list format if needed
    if isinstance(syn_x, np.ndarray):
        if syn_x.ndim == 4:
            # (batch, samples_per_batch, channels, length) -> reshape to (batch*samples_per_batch, channels, length)
            syn_x = syn_x.reshape(-1, syn_x.shape[2], syn_x.shape[3])
            # Now it's 3D: (N, channels, length) -> convert to list of (length, channels)
            X = [syn_x[i].T for i in range(len(syn_x))]
        elif syn_x.ndim == 3:
            # (N, channels, length) -> convert to list of (length, channels)
            X = [syn_x[i].T for i in range(len(syn_x))]
        elif syn_x.ndim == 2:
            # (N, length) -> assume single channel, convert to (length, 1)
            X = [syn_x[i].reshape(-1, 1) for i in range(len(syn_x))]
        else:
            X = syn_x
    else:
        X = syn_x
    
    # Filter and prepare data
    X = [l.astype(np.float32) for l in X]
    y_dummy = [np.ones(num_classes, dtype=np.float32) for _ in range(len(X))]
    
    # Get learner
    learn = model._get_learner(X, y_dummy, X, y_dummy)
    
    # Directly load weights from 100000.pkl (not from saved model)
    checkpoint = torch.load(pretrained_file, map_location='cpu')
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Load state dict into model (strict=False to allow some mismatch if needed)
    learn.model.load_state_dict(state_dict, strict=False)
    
    # Get predictions
    with torch.no_grad():
        preds, targs = learn.get_preds()
        preds = to_np(preds)
    
    max_indices = np.argmax(preds, axis=1)

    # Create a one-hot embedding tensor
    one_hot = np.zeros_like(preds)
    one_hot[np.arange(len(max_indices)), max_indices] = 1
    return one_hot
    