"""
Test classifiers trained on generated data.
Uses fixed seed=42 for reproducibility.
"""
import os
import sys
import pandas as pd
import numpy as np
import argparse

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
code_path = os.path.join(script_dir, 'Diffusion_RL', 'ecg_ptbxl_benchmarking', 'code')
sys.path.insert(0, code_path)

from experiments.scp_experiment import SCP_Experiment
from util_utility import utils
from configs.fastai_configs import *
import random
import torch

# Fixed seed for reproducibility
SEED = 42

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed} for reproducibility")

def run_experiment(syn_data_suffix, syn_data_subdir, outputfolder_base, data_experiment_name):
    """
    Run a single experiment with seed=42.
    
    Parameters:
        syn_data_suffix: Suffix for synthetic data files (e.g., '_clf_sobd')
        syn_data_subdir: Subdirectory for synthetic data (e.g., 'ch256_T200_betaT0.02')
        outputfolder_base: Base output folder
        data_experiment_name: Experiment name used to locate data files
    
    Returns:
        dict: Results dictionary with metrics
    """
    # Set seed
    set_seed(SEED)
    
    # Create experiment name
    experiment_name = f'exp_ICBEB{syn_data_suffix}_seed{SEED}'
    
    # Create output folder for this experiment
    dataset_folder = syn_data_suffix if syn_data_suffix else 'baseline'
    outputfolder = os.path.join(outputfolder_base, dataset_folder, f'seed_{SEED}')
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)
    outputfolder = outputfolder + '/'
    
    datafolder_icbeb = 'Diffusion_RL/ecg_ptbxl_benchmarking/data/ICBEB/'
    
    models = [conf_fastai_xresnet1d50]
    
    print(f'\n{"="*80}')
    print(f'Running experiment: {experiment_name}')
    print(f'  Synthetic data suffix: {syn_data_suffix}')
    print(f'  Synthetic data subdir: {syn_data_subdir}')
    print(f'  Data experiment name: {data_experiment_name}')
    print(f'  Seed: {SEED}')
    print(f'  Output folder: {outputfolder}')
    print(f'{"="*80}\n')
    
    # Create experiment
    e = SCP_Experiment(
        experiment_name, 
        'all', 
        datafolder_icbeb, 
        outputfolder, 
        models,
        syn_data_suffix=syn_data_suffix,
        syn_data_subdir=syn_data_subdir,
        syn_data_experiment_name=data_experiment_name
    )
    
    try:
        e.prepare()
        e.perform()
        e.evaluate()
        
        # Read results
        results_path = os.path.join(outputfolder.rstrip('/'), experiment_name, 'models', 'fastai_xresnet1d50', 'results', 'te_results.csv')
        
        if os.path.exists(results_path):
            results_df = pd.read_csv(results_path, index_col=0)
            print(f"\n✓ Successfully completed experiment {experiment_name}")
            return {
                'suffix': syn_data_suffix,
                'seed': SEED,
                'experiment_name': experiment_name,
                'results_df': results_df,
                'results_path': results_path
            }
        else:
            print(f"\n✗ Results file not found: {results_path}")
            return None
            
    except Exception as ex:
        print(f"\n✗ Error running experiment {experiment_name}: {ex}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='Test classifiers on generated data (seed=42)')
    parser.add_argument('--outputfolder', type=str, 
                        default='Diffusion_RL/ecg_ptbxl_benchmarking/output',
                        help='Base output folder for results')
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='Experiment name to locate generated data files (required)')
    parser.add_argument('--data_file', type=str, default=None,
                        help='Custom data file name (e.g., gen_data_clf.npy). If not specified, uses default.')
    parser.add_argument('--use_sobolev', action='store_true',
                        help='If set, use Sobolev data file (gen_data_clf_sobd.npy by default). Otherwise, use naive (gen_data_clf.npy)')
    parser.add_argument('--no_decay', action='store_true',
                        help='If set with --use_sobolev, use gen_data_clf_sob.npy instead of gen_data_clf_sobd.npy')
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Resolve output folder path
    if not os.path.isabs(args.outputfolder):
        outputfolder_base = os.path.join(script_dir, args.outputfolder)
    else:
        outputfolder_base = args.outputfolder
    
    # Add experiment name suffix if provided
    if args.experiment_name:
        outputfolder_base = os.path.join(outputfolder_base, args.experiment_name)
    
    # Ensure output folder exists
    os.makedirs(outputfolder_base, exist_ok=True)
    
    # Define the dataset
    # Data files are located at: output/guided_{experiment_name}/ch256_T200_betaT0.02/
    # Use custom data_file if provided, otherwise use default based on --use_sobolev flag
    if args.data_file:
        # User specified custom data file
        default_data_file = args.data_file
    elif args.use_sobolev:
        # Use sobolev data file when --use_sobolev is set
        if hasattr(args, 'no_decay') and args.no_decay:
            default_data_file = 'gen_data_clf_sob.npy'  # Sobolev without decay
        else:
            default_data_file = 'gen_data_clf_sobd.npy'  # Sobolev with decay (default)
    else:
        # Default to clf data file
        default_data_file = 'gen_data_clf.npy'
    data_file_name = default_data_file
    
    # Infer suffix from data_file name (new naming convention)
    # Suffixes: _clf, _clf_sob, _clf_sobd, _diff, _diff_scratch
    if '_clf_sobd' in data_file_name:
        suffix = '_clf_sobd'
        name = 'clf_sobd'
    elif '_clf_sob' in data_file_name:
        suffix = '_clf_sob'
        name = 'clf_sob'
    elif '_clf' in data_file_name:
        suffix = '_clf'
        name = 'clf'
    elif '_diff_scratch' in data_file_name:
        suffix = '_diff_scratch'
        name = 'diff_scratch'
    elif '_diff' in data_file_name:
        suffix = '_diff'
        name = 'diff'
    else:
        # Try to infer from filename
        suffix = ''
        name = data_file_name.replace('gen_data', '').replace('.npy', '')
        if name.startswith('_'):
            suffix = name
        else:
            suffix = '_' + name if name else ''
    
    dataset = {
        'name': name,
        'suffix': suffix,
        'subdir': 'ch256_T200_betaT0.02',
        'data_file': data_file_name
    }
    
    # Verify data file exists using experiment_name (new path format)
    data_base_dir = os.path.join(script_dir, f'output/guided_{args.experiment_name}', dataset['subdir'])
    print(f"\nChecking data file in: {data_base_dir}\n")
    
    data_file = os.path.join(data_base_dir, dataset['data_file'])
    if os.path.exists(data_file):
        print(f"✓ Found: {dataset['name']} - {dataset['data_file']}")
    else:
        print(f"✗ Missing: {dataset['name']} - {dataset['data_file']}")
        print(f"  Expected at: {data_file}")
        print(f"  Please ensure the data file exists at the expected location.")
        return
    
    print(f"\n{'='*80}")
    print(f"Starting experiment with seed={SEED}")
    print(f"  Dataset: {dataset['name']}")
    print(f"  Experiment name: {args.experiment_name}")
    print(f"{'='*80}\n")
    
    # Run single experiment with seed=42
    result = run_experiment(
        suffix, 
        dataset['subdir'], 
        outputfolder_base, 
        args.experiment_name
    )
    
    if result is None:
        print("Experiment failed.")
        return
    
    # Print results
    print(f"\n{'='*80}")
    print("Results Summary")
    print(f"{'='*80}")
    
    results_df = result['results_df']
    
    # Extract key metrics
    row_to_use = 'point' if 'point' in results_df.index else ('mean' if 'mean' in results_df.index else results_df.index[0])
    
    print(f"\nDataset: {dataset['name']}")
    print(f"Seed: {SEED}")
    
    # Print all metrics
    for col in results_df.columns:
        if row_to_use in results_df.index:
            value = results_df.loc[row_to_use, col]
            print(f"  {col}: {value:.4f}")
    
    # Save results
    summary_csv_path = os.path.join(outputfolder_base, 'results_summary.csv')
    results_df.to_csv(summary_csv_path)
    print(f"\nResults saved to: {summary_csv_path}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
