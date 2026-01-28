from experiments.scp_experiment import SCP_Experiment
from util_utility import utils
# model configs
from configs.fastai_configs import *
from configs.wavelet_configs import *
import argparse
import random
import numpy as np
import torch
import os

import ipdb

def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    This sets seeds for Python random, NumPy, PyTorch (CPU and CUDA), and CUDNN.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For reproducibility, use deterministic algorithms (may be slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed} for reproducibility")

def main():
    # Random seed will be set later based on command line args or environment variable
    # Default seed is set to 42, but can be overridden
    
    # datafolder = 'physionet.org/files/ptb-xl/1.0.3/'
    # datafolder_icbeb = 'Diffusion_RL/ecg_ptbxl_benchmarking/data/ICBEB/' #normal ICBEB
    datafolder_icbeb = 'Diffusion_RL/ecg_ptbxl_benchmarking/data/ICBEB/'   #vanilla diffusion 
    outputfolder = 'Diffusion_RL/ecg_ptbxl_benchmarking/output/'

    models = [
        conf_fastai_xresnet1d50,
        # conf_fastai_resnet1d_wang,
        # conf_fastai_lstm,
        # conf_fastai_lstm_bidir,
        # conf_fastai_fcn_wang,
        # conf_fastai_inception1d,
        # conf_wavelet_standard_nn,
        ]

    ##########################################
    # STANDARD SCP EXPERIMENTS ON PTBXL
    ##########################################

    # experiments = [
    #     ('exp0', 'all'),
    #     # ('exp1', 'diagnostic'),
    #     # ('exp1.1', 'subdiagnostic'),
    #     # ('exp1.1.1', 'superdiagnostic'),
    #     # ('exp2', 'form'),
    #     # ('exp3', 'rhythm')
    #    ]

    # for name, task in experiments:
    #     e = SCP_Experiment(name, task, datafolder, outputfolder, models)
    #     e.prepare()
    #     # ipdb.set_trace()
    #     e.perform()
    #     e.evaluate()

    # # generate greate summary table
    # utils.generate_ptbxl_summary_table()

    ##########################################
    # EXPERIMENT BASED ICBEB DATA
    ##########################################

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run ICBEB experiment with synthetic data augmentation')
    parser.add_argument('--syn_data_suffix', type=str, default='',
                        help='Suffix for synthetic data files (e.g., "_guidance_sobolev", "_uguidance", ""). Default: "_guidance_sobolev"')
    parser.add_argument('--syn_data_subdir', type=str, default='ch256_T200_betaT0.02',
                        help='Subdirectory for synthetic data (e.g., "ch256_T200_betaT0.02", "ch256_T5_betaT0.02"). Default: "ch256_T200_betaT0.02"')
    parser.add_argument('--outputfolder', type=str, default=None,
                        help='Output folder for results (default: Diffusion_RL/ecg_ptbxl_benchmarking/output/)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: 42, or from REPRODUCE_SEED env var)')
    args = parser.parse_args()
    
    # Use command line arguments or defaults
    syn_data_suffix = args.syn_data_suffix
    syn_data_subdir = args.syn_data_subdir
    if args.outputfolder is not None:
        outputfolder = args.outputfolder
    
    # Get seed from command line, environment variable, or use default
    if args.seed is not None:
        seed = args.seed
    elif 'REPRODUCE_SEED' in os.environ:
        seed = int(os.environ['REPRODUCE_SEED'])
    else:
        seed = 42
    
    # Set random seed (override the default set_seed(42) call)
    set_seed(seed)
    
    # Generate experiment name from suffix
    experiment_name = f'exp_ICBEB{syn_data_suffix}'  # Include suffix in experiment name so CSV files have suffix
    
    print(f'Running experiment: {experiment_name}')
    print(f'  Synthetic data suffix: {syn_data_suffix}')
    print(f'  Synthetic data subdir: {syn_data_subdir}')
    
    e = SCP_Experiment(experiment_name, 'all', datafolder_icbeb, outputfolder, models,
                       syn_data_suffix=syn_data_suffix,
                       syn_data_subdir=syn_data_subdir)
    e.prepare()
    # ipdb.set_trace()
    e.perform()
    e.evaluate()

    # generate greate summary table
    utils.ICBEBE_table(experiment_name=experiment_name, folder=outputfolder)

if __name__ == "__main__":
    main()
