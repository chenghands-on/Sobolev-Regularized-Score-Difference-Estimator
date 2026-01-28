"""
Evaluate synthetic ECG data using FID (Frechet Inception Distance) and Coverage metrics.

FID: Measures Wasserstein-2 distance between real and synthetic data in feature space.
Coverage: Ratio of real records that have at least one synthetic record within their sphere.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from scipy.linalg import sqrtm
import argparse
import random

# Add paths
sys.path.append("Diffusion_RL/ecg_ptbxl_benchmarking/code/models")
sys.path.append("Diffusion_RL/ecg_ptbxl_benchmarking/code/")
from models.fastai_model import fastai_model
from load_icbeb import load_icbeb_data


def load_pretrained_classifier_for_features(model_path):
    """
    Load pretrained classifier model (100000.pkl) and return a function to extract features.
    Features are extracted from the last layer before classification.
    """
    # Get script directory to resolve relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_weight_dir = os.path.join(script_dir, 'Diffusion_RL', 'ecg_ptbxl_benchmarking', 'model_weight')
    
    # Essential parameters
    modelname = 'fastai_xresnet1d50'
    pretrained_file = model_path if os.path.isabs(model_path) else os.path.join(model_weight_dir, model_path)
    outputfolder = model_weight_dir + '/'
    num_classes = 71  # Keep original 71 classes for feature extraction
    sampling_frequency = 100
    input_shape = [1000, 12]

    # Load model without replacing output layer (keep 71 classes for feature extraction)
    model = fastai_model(
        name=modelname,
        n_classes=num_classes,
        freq=sampling_frequency,
        outputfolder=outputfolder,
        input_shape=input_shape,
        pretrained=True,
        pretrained_file=pretrained_file,
        n_classes_pretrained=num_classes,
        replace_output_layer=False,  # Keep original output layer
        epochs_finetuning=0,  # No fine-tuning needed
    )
    
    print(f"Loaded pretrained classifier from: {pretrained_file}")
    
    # Load 100000.pkl weights once and store for reuse
    checkpoint = torch.load(pretrained_file, map_location='cpu')
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            pretrained_state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            pretrained_state_dict = checkpoint['model_state_dict']
        else:
            pretrained_state_dict = checkpoint
    else:
        pretrained_state_dict = checkpoint
    
    # Filter out output layer weights (we want features before classification)
    filtered_state_dict = {k: v for k, v in pretrained_state_dict.items() 
                          if not k.startswith('8.6.') and not k.startswith('8.8.')}
    
    def extract_features(X):
        """
        Extract features from the last layer before classification.
        Uses model.get_embedding_for_real_data but loads 100000.pkl weights.
        
        Args:
            X: Input data, list of arrays or numpy array
            
        Returns:
            features: Feature vectors [N, feature_dim]
        """
        # Handle list of arrays (real ICBEB data format)
        if isinstance(X, list):
            X_processed = X
        else:
            # Handle numpy array (synthetic data format)
            if X.ndim == 4:
                # (batch, samples_per_batch, channels, length) -> flatten first two dims
                X = X.reshape(-1, X.shape[2], X.shape[3])
            if X.ndim == 3:
                # (N, channels, length) -> convert to list format
                X_processed = [X[i].T for i in range(len(X))]  # Transpose to (length, channels)
            else:
                X_processed = X
        
        # Use model's get_embedding_for_real_data method, but we need to load 100000.pkl weights
        # Create dummy labels for the learner
        filtered_arrays = [arr for arr in X_processed if arr.shape[0] >= 1000]
        filtered_arrays = [arr.astype(np.float32)[:1000] for arr in filtered_arrays]
        
        y_dummy = [np.ones(model.num_classes, dtype=np.float32) for _ in range(len(filtered_arrays))]
        learn = model._get_embedding_learner(filtered_arrays, y_dummy, filtered_arrays, y_dummy)
        
        # Load the 100000.pkl model weights (use filtered_state_dict from outer scope)
        learn.model.load_state_dict(filtered_state_dict, strict=False)
        
        # Get predictions (which are the features before the final classification layer)
        learn.model.eval()
        # Disable dropout and other random operations
        # for module in learn.model.modules():
        #     if isinstance(module, torch.nn.Dropout):
        #         module.p = 0.0
        #     if isinstance(module, torch.nn.BatchNorm1d) or isinstance(module, torch.nn.BatchNorm2d):
        #         module.eval()
        with torch.no_grad():
            preds, _ = learn.get_preds()
            # Reshape to get features: preds shape is [N, 7*128] = [N, 896]
            # Reshape to [N, 7, 128] and take mean to get [N, 128] features
            if preds.shape[1] == 896:  # 7 * 128
                reshaped_tensor = preds.view(len(filtered_arrays), 7, 128)
                embedding = reshaped_tensor.mean(dim=1)  # [N, 128]
            else:
                # If shape is different, use as is
                embedding = preds
        
        return embedding.numpy() if isinstance(embedding, torch.Tensor) else embedding
    
    return extract_features


def calculate_fid(real_features, synthetic_features):
    """
    Calculate Frechet Inception Distance (FID).
    
    FID = ||mu_r - mu_s||^2 + Tr(C_r + C_s - 2*sqrt(C_r * C_s))
    where mu_r, mu_s are means and C_r, C_s are covariance matrices.
    
    Args:
        real_features: Feature vectors of real data [N_real, feature_dim]
        synthetic_features: Feature vectors of synthetic data [N_syn, feature_dim]
        
    Returns:
        fid: FID score (lower is better)
    """
    # Calculate means
    mu_real = np.mean(real_features, axis=0)
    mu_syn = np.mean(synthetic_features, axis=0)
    
    # Calculate covariance matrices
    C_real = np.cov(real_features, rowvar=False)
    C_syn = np.cov(synthetic_features, rowvar=False)
    
    # Calculate squared difference of means
    mean_diff = mu_real - mu_syn
    mean_diff_sq = np.sum(mean_diff ** 2)
    
    # Calculate sqrt of product of covariance matrices
    try:
        covmean = sqrtm(C_real @ C_syn)
        # Check if result is complex (shouldn't happen for valid covariances)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
    except Exception as e:
        print(f"Warning: sqrtm failed, using alternative method: {e}")
        # Alternative: use eigendecomposition
        eigvals_real, eigvecs_real = np.linalg.eigh(C_real)
        eigvals_syn, eigvecs_syn = np.linalg.eigh(C_syn)
        # Approximate sqrt
        covmean = eigvecs_real @ np.diag(np.sqrt(np.maximum(eigvals_real, 0))) @ eigvecs_real.T
    
    # Calculate FID
    fid = mean_diff_sq + np.trace(C_real + C_syn - 2 * covmean)
    
    return fid


def calculate_coverage(real_features, synthetic_features, radius_percentile=5):
    """
    Calculate Coverage metric.
    
    Coverage = ratio of real records that have at least one synthetic record 
    within their sphere (defined by radius).
    
    Args:
        real_features: Feature vectors of real data [N_real, feature_dim]
        synthetic_features: Feature vectors of synthetic data [N_syn, feature_dim]
        radius_percentile: Percentile to use for radius calculation (default: 5)
        
    Returns:
        coverage: Coverage score (higher is better, range [0, 1])
    """
    # Calculate pairwise distances between all real and synthetic samples
    distances = cdist(real_features, synthetic_features, metric='euclidean')
    
    # For each real sample, find the minimum distance to any synthetic sample
    min_distances = np.min(distances, axis=1)  # [N_real]
    
    # Calculate radius as percentile of distances between real samples
    # This gives a sense of the "local neighborhood" size
    real_pairwise_distances = cdist(real_features, real_features, metric='euclidean')
    # Remove diagonal (self-distances)
    mask = ~np.eye(real_pairwise_distances.shape[0], dtype=bool)
    real_distances = real_pairwise_distances[mask]
    radius = np.percentile(real_distances, radius_percentile)
    
    # Coverage: ratio of real samples that have at least one synthetic sample within radius
    covered = np.sum(min_distances <= radius)
    coverage = covered / len(real_features)
    
    return coverage, radius


def main():
    parser = argparse.ArgumentParser(description='Evaluate synthetic ECG data using FID and Coverage')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--synthetic_data', type=str, 
                        default=os.path.join(script_dir, 'output', 'guided', 'ch256_T200_betaT0.02', 'gen_data_clf_sobd.npy'),
                        help='Path to synthetic data file (.npy)')
    parser.add_argument('--model_path', type=str,
                        default=os.path.join(script_dir, 'Diffusion_RL', 'ecg_ptbxl_benchmarking', 'model_weight', '100000.pkl'),
                        help='Path to pretrained classifier model')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to use for evaluation (None = use all)')
    parser.add_argument('--coverage_radius_percentile', type=int, default=5,
                        help='Percentile for coverage radius calculation (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {args.seed}")
    
    print("=" * 80)
    print("ECG Synthetic Data Evaluation: FID and Coverage")
    print("=" * 80)
    
    # Load pretrained classifier for feature extraction
    print("\n1. Loading pretrained classifier...")
    extract_features = load_pretrained_classifier_for_features(args.model_path)
    
    # Load real ICBEB data for FID calculation
    # Note: To be consistent with scp_experiment.py, we use original train_data
    # (which becomes the new test_data after swap in scp_experiment.py)
    print("\n2. Loading real ICBEB data...")
    print("   Data source: ICBEB original train data (fold <= 8) from 'Diffusion_RL/ecg_ptbxl_benchmarking/data/ICBEB/'")
    print("   Note: This corresponds to the test set used in scp_experiment.py after train/test swap")
    X_train_ICBEB, X_val_ICBEB, X_test_ICBEB, y_train_ICBEB, y_val_ICBEB, y_test_ICBEB = load_icbeb_data()
    
    # Use original train data for FID calculation (which is the test set in scp_experiment.py after swap)
    X_all_ICBEB = list(X_train_ICBEB)
    y_all_ICBEB = y_train_ICBEB
    
    print(f"   Before filtering: {len(X_all_ICBEB)} samples (original train data, fold <= 8)")
    
    # Filter real data (length >= 1000)
    indices = [i for i, d in enumerate(X_all_ICBEB) if d.shape[0] >= 1000]
    X_all_ICBEB = [X_all_ICBEB[i][:1000] for i in indices]
    y_all_ICBEB = y_all_ICBEB[indices]
    
    print(f"   After filtering (length >= 1000): {len(X_all_ICBEB)} samples")
    
    # Load synthetic data
    print(f"\n3. Loading synthetic data from: {args.synthetic_data}")
    synthetic_data = np.load(args.synthetic_data, allow_pickle=True)
    print(f"   Synthetic data shape: {synthetic_data.shape}")
    
    # Handle different synthetic data formats
    if synthetic_data.ndim == 4:
        # (batch, samples_per_batch, channels, length) -> flatten first two dims
        synthetic_data = synthetic_data.reshape(-1, synthetic_data.shape[2], synthetic_data.shape[3])
        print(f"   Reshaped to: {synthetic_data.shape}")
    elif synthetic_data.ndim == 3:
        # (N, channels, length) - already in correct format
        pass
    else:
        raise ValueError(f"Unexpected synthetic data shape: {synthetic_data.shape}")
    
    # Limit samples if specified
    if args.max_samples is not None:
        n_real = min(len(X_all_ICBEB), args.max_samples)
        n_syn = min(len(synthetic_data), args.max_samples)
        X_all_ICBEB = X_all_ICBEB[:n_real]
        synthetic_data = synthetic_data[:n_syn]
        print(f"   Limited to {n_real} real and {n_syn} synthetic samples")
    
    # Extract features
    print("\n4. Extracting features from real data...")
    real_features = extract_features(X_all_ICBEB)
    print(f"   Real features shape: {real_features.shape}")
    
    print("\n5. Extracting features from synthetic data...")
    synthetic_features = extract_features(synthetic_data)
    print(f"   Synthetic features shape: {synthetic_features.shape}")
    
    # Calculate FID
    print("\n6. Calculating FID...")
    fid = calculate_fid(real_features, synthetic_features)
    print(f"   FID = {fid:.6f} (lower is better)")
    
    # # Calculate Coverage
    # print(f"\n7. Calculating Coverage (radius percentile: {args.coverage_radius_percentile})...")
    # coverage, radius = calculate_coverage(real_features, synthetic_features, 
    #                                      radius_percentile=args.coverage_radius_percentile)
    # print(f"   Coverage = {coverage:.6f} (higher is better, range [0, 1])")
    # print(f"   Radius used: {radius:.6f}")
    
    # # Summary
    # print("\n" + "=" * 80)
    # print("Evaluation Results Summary")
    # print("=" * 80)
    # print(f"FID:        {fid:.6f} (lower is better)")
    # print(f"Coverage:   {coverage:.6f} (higher is better)")
    # print(f"Radius:     {radius:.6f}")
    # print("=" * 80)
    
    # return fid, coverage


if __name__ == "__main__":
    main()

