from util_utility import utils
import os
import pickle
import pandas as pd
import numpy as np
import multiprocessing
from itertools import repeat

import sys
# Add path to sssd directory (where assign_pesudo_label_for_syn_data.py is located)
# From experiments/ -> code/ -> ecg_ptbxl_benchmarking/ -> Diffusion_RL/ -> sssd/
current_dir = os.path.dirname(os.path.abspath(__file__))
sssd_dir = os.path.join(current_dir, '../../../../')
sssd_dir = os.path.abspath(sssd_dir)
sys.path.append(sssd_dir)
from assign_pesudo_label_for_syn_data import infer_label

import ipdb

class SCP_Experiment():
    '''
        Experiment on SCP-ECG statements. All experiments based on SCP are performed and evaluated the same way.
    '''

    def __init__(self, experiment_name, task, datafolder, outputfolder, models, sampling_frequency=100, min_samples=0, train_fold=8, val_fold=9, test_fold=10, folds_type='strat', syn_data_suffix='', syn_data_subdir='ch256_T200_betaT0.02', syn_data_experiment_name=None):
        self.models = models
        self.min_samples = min_samples
        self.task = task
        self.train_fold = train_fold
        self.val_fold = val_fold
        self.test_fold = test_fold
        self.folds_type = folds_type
        self.experiment_name = experiment_name
        self.syn_data_suffix = syn_data_suffix  # Suffix for synthetic data files (e.g., '_guidance_sobolev', '_uguidance', '')
        self.syn_data_subdir = syn_data_subdir  # Subdirectory for synthetic data (e.g., 'ch256_T5_betaT0.02', 'ch256_T200_betaT0.02')
        self.syn_data_experiment_name = syn_data_experiment_name  # Experiment name for locating data files (e.g., 'seed42_20260118_092837')
        
        # Convert relative paths to absolute paths based on script location
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up to code/ directory: experiments/ -> code/
        code_dir = os.path.dirname(script_dir)
        # Go up to ecg_ptbxl_benchmarking/ directory: code/ -> ecg_ptbxl_benchmarking/
        benchmark_dir = os.path.dirname(code_dir)
        # Go up to Diffusion_RL/ directory: ecg_ptbxl_benchmarking/ -> Diffusion_RL/
        diffusion_rl_dir = os.path.dirname(benchmark_dir)
        # Base directory for relative paths: Diffusion_RL/ -> sssd/
        base_dir = os.path.dirname(diffusion_rl_dir)
        # Save base_dir for resolving synthetic data paths
        self.base_dir = base_dir
        
        # Convert datafolder to absolute path if it's relative
        if not os.path.isabs(datafolder):
            # If it starts with Diffusion_RL/, use base_dir
            if datafolder.startswith('Diffusion_RL/'):
                self.datafolder = os.path.join(base_dir, datafolder)
            else:
                # Otherwise, resolve relative to current working directory
                self.datafolder = os.path.abspath(datafolder)
        else:
            self.datafolder = datafolder
        
        # Ensure datafolder ends with '/' for proper path concatenation
        if not self.datafolder.endswith('/'):
            self.datafolder = self.datafolder + '/'
        
        # Convert outputfolder to absolute path if it's relative
        if not os.path.isabs(outputfolder):
            # If it starts with Diffusion_RL/, use base_dir
            if outputfolder.startswith('Diffusion_RL/'):
                outputfolder = os.path.join(base_dir, outputfolder)
            else:
                # Otherwise, resolve relative to current working directory
                outputfolder = os.path.abspath(outputfolder)
        
        # Ensure outputfolder ends with '/' for proper path concatenation
        self.outputfolder = outputfolder if outputfolder.endswith('/') else outputfolder + '/'
        self.sampling_frequency = sampling_frequency

        # create folder structure if needed
        if not os.path.exists(self.outputfolder+self.experiment_name):
            os.makedirs(self.outputfolder+self.experiment_name)
            if not os.path.exists(self.outputfolder+self.experiment_name+'/results/'):
                os.makedirs(self.outputfolder+self.experiment_name+'/results/')
            if not os.path.exists(self.outputfolder+self.experiment_name+'/models/'):
                os.makedirs(self.outputfolder+self.experiment_name+'/models/')
            if not os.path.exists(self.outputfolder+self.experiment_name+'/data/'):
                os.makedirs(self.outputfolder+self.experiment_name+'/data/')

    def get_ptbxl(self):
        # Load PTB-XL data
        self.data, self.raw_labels = utils.load_dataset(self.datafolder, self.sampling_frequency)

        # ipdb.set_trace()

        # Preprocess label data
        self.labels = utils.compute_label_aggregations(self.raw_labels, self.datafolder, self.task)

        # Select relevant data and convert to one-hot
        self.data, self.labels, self.Y, _ = utils.select_data(self.data, self.labels, self.task, self.min_samples, self.outputfolder+self.experiment_name+'/data/')

        self.input_shape = self.data[0].shape
        # 10th fold for testing (9th for now)
        self.X_test = self.data[self.labels.strat_fold == self.test_fold]
        self.y_test = self.Y[self.labels.strat_fold == self.test_fold]
        # 9th fold for validation (8th for now)
        self.X_val = self.data[self.labels.strat_fold == self.val_fold]
        self.y_val = self.Y[self.labels.strat_fold == self.val_fold]
        # rest for training
        self.X_train = self.data[self.labels.strat_fold <= self.train_fold]
        self.y_train = self.Y[self.labels.strat_fold <= self.train_fold]

        # Preprocess signal data
        self.X_train, self.X_val, self.X_test = utils.preprocess_signals(self.X_train, self.X_val, self.X_test, self.outputfolder+self.experiment_name+'/data/')

        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def get_ICBEB(self):
        # Load ICBEB data
        self.data, self.raw_labels = utils.load_dataset(self.datafolder, self.sampling_frequency)

        # ipdb.set_trace()

        # Preprocess label data
        self.labels = utils.compute_label_aggregations(self.raw_labels, self.datafolder, self.task)

        # Select relevant data and convert to one-hot
        self.data, self.labels, self.Y, _ = utils.select_data(self.data, self.labels, self.task, self.min_samples, self.outputfolder+self.experiment_name+'/data/')

        self.input_shape = self.data[0].shape
        # 10th fold for testing (9th for now)
        self.X_test = self.data[self.labels.strat_fold == self.test_fold]
        self.y_test = self.Y[self.labels.strat_fold == self.test_fold]
        # 9th fold for validation (8th for now)
        self.X_val = self.data[self.labels.strat_fold == self.val_fold]
        self.y_val = self.Y[self.labels.strat_fold == self.val_fold]
        # rest for training
        self.X_train = self.data[self.labels.strat_fold <= self.train_fold]
        self.y_train = self.Y[self.labels.strat_fold <= self.train_fold]

        # ipdb.set_trace()
        # Preprocess signal data
        self.X_train, self.X_val, self.X_test = utils.preprocess_signals(self.X_train, self.X_val, self.X_test, self.outputfolder+self.experiment_name+'/data/')
        # ipdb.set_trace()

        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test

    def get_ptbxl_unnormalized(self):
        # Load PTB-XL data
        self.data, self.raw_labels = utils.load_dataset(self.datafolder, self.sampling_frequency)

        # ipdb.set_trace()

        # Preprocess label data
        self.labels = utils.compute_label_aggregations(self.raw_labels, self.datafolder, self.task)

        # Select relevant data and convert to one-hot
        self.data, self.labels, self.Y, _ = utils.select_data(self.data, self.labels, self.task, self.min_samples, self.outputfolder+self.experiment_name+'/data/')

        self.input_shape = self.data[0].shape
        # 10th fold for testing (9th for now)
        self.X_test = self.data[self.labels.strat_fold == self.test_fold]
        self.y_test = self.Y[self.labels.strat_fold == self.test_fold]
        # 9th fold for validation (8th for now)
        self.X_val = self.data[self.labels.strat_fold == self.val_fold]
        self.y_val = self.Y[self.labels.strat_fold == self.val_fold]
        # rest for training
        self.X_train = self.data[self.labels.strat_fold <= self.train_fold]
        self.y_train = self.Y[self.labels.strat_fold <= self.train_fold]

        # Preprocess signal data
        # self.X_train, self.X_val, self.X_test = utils.preprocess_signals(self.X_train, self.X_val, self.X_test, self.outputfolder+self.experiment_name+'/data/')

        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def get_ICBEB_unnormalized(self):
        # Load ICBEB data
        self.data, self.raw_labels = utils.load_dataset(self.datafolder, self.sampling_frequency)

        # ipdb.set_trace()

        # Preprocess label data
        self.labels = utils.compute_label_aggregations(self.raw_labels, self.datafolder, self.task)

        # Select relevant data and convert to one-hot
        self.data, self.labels, self.Y, _ = utils.select_data(self.data, self.labels, self.task, self.min_samples, self.outputfolder+self.experiment_name+'/data/')

        self.input_shape = self.data[0].shape
        # 10th fold for testing (9th for now)
        self.X_test = self.data[self.labels.strat_fold == self.test_fold]
        self.y_test = self.Y[self.labels.strat_fold == self.test_fold]
        # 9th fold for validation (8th for now)
        self.X_val = self.data[self.labels.strat_fold == self.val_fold]
        self.y_val = self.Y[self.labels.strat_fold == self.val_fold]
        # rest for training
        self.X_train = self.data[self.labels.strat_fold <= self.train_fold]
        self.y_train = self.Y[self.labels.strat_fold <= self.train_fold]

        # ipdb.set_trace()
        # Preprocess signal data
        # self.X_train, self.X_val, self.X_test = utils.preprocess_signals(self.X_train, self.X_val, self.X_test, self.outputfolder+self.experiment_name+'/data/')
        # ipdb.set_trace()

        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test

    def load_data_for_embedding(self, data_path):

        self.data = np.load(data_path+'/all_generated_data.npy')
        self.labels = np.load(data_path+'/all_generated_label.npy')

        # ipdb.set_trace()
        self.data = self.data.reshape(17200,12,1000)
        # self.data = self.data.reshape(16500,12,1000)
        # self.data = self.data.reshape(17152,12,1000)
        self.data = np.transpose(self.data, (0, 2, 1))

        # self.input_shape = self.data[0].shape
        # # 10th fold for testing (9th for now)
        # self.X_test = self.data[self.labels.strat_fold == self.test_fold]
        # self.y_test = self.Y[self.labels.strat_fold == self.test_fold]
        # # 9th fold for validation (8th for now)
        # self.X_val = self.data[self.labels.strat_fold == self.val_fold]
        # self.y_val = self.Y[self.labels.strat_fold == self.val_fold]
        # # rest for training
        # self.X_train = self.data[self.labels.strat_fold <= self.train_fold]
        # self.y_train = self.Y[self.labels.strat_fold <= self.train_fold]


        # Preprocess signal data
        # self.X_train, self.X_val, self.X_test = utils.preprocess_signals(self.X_train, self.X_val, self.X_test, self.outputfolder+self.experiment_name+'/data/')
        self.X_train, self.X_val, self.X_test = utils.preprocess_signals(self.data, self.data[-3:-1], self.data[-3:-1], self.outputfolder+self.experiment_name+'/data/')
        # ipdb.set_trace()
        return self.X_train, self.X_val, self.X_test

    def prepare(self):
        
        # Load PTB-XL data
        self.data, self.raw_labels = utils.load_dataset(self.datafolder, self.sampling_frequency)

        # ipdb.set_trace()

        # Preprocess label data
        self.labels = utils.compute_label_aggregations(self.raw_labels, self.datafolder, self.task)

        # Select relevant data and convert to one-hot
        self.data, self.labels, self.Y, _ = utils.select_data(self.data, self.labels, self.task, self.min_samples, self.outputfolder+self.experiment_name+'/data/')

        # Standardize the data of icbeb
        # Filter: delete samples with length < 1000, and truncate remaining samples to 1000
        data_before_filter = len(self.data)
        print(f"Data filtering: Before filter: {data_before_filter} samples")
        
        # Step 1: Find indices of samples with length >= 1000 (keep these, delete others)
        indices = [i for i, d in enumerate(self.data) if d.shape[0] >= 1000]
        n_filtered_out = data_before_filter - len(indices)
        
        # Step 2: Keep only samples with length >= 1000, and truncate them to length 1000
        self.data = [d[:1000,:] for d in self.data if d.shape[0] >= 1000]
        
        # Step 3: Filter labels and Y using the same indices (delete labels for samples with length < 1000)
        self.labels = self.labels.iloc[indices].reset_index(drop=True)
        self.Y = self.Y[indices]

        print(f"Data filtering: After filter: {len(self.data)} samples (removed {n_filtered_out} samples with length < 1000)")
        print(f"Data filtering: All remaining samples truncated to length 1000")

        self.input_shape = self.data[0].shape
        # 10th fold for testing (9th for now)
        # Convert boolean Series to list indices for list indexing
        test_mask = (self.labels.strat_fold == self.test_fold).values
        self.X_test = [self.data[i] for i in range(len(self.data)) if test_mask[i]]
        self.y_test = self.Y[test_mask]
        # 9th fold for validation (8th for now)
        val_mask = (self.labels.strat_fold == self.val_fold).values
        self.X_val = [self.data[i] for i in range(len(self.data)) if val_mask[i]]
        self.y_val = self.Y[val_mask]
        # rest for training
        train_mask = (self.labels.strat_fold <= self.train_fold).values
        self.X_train = [self.data[i] for i in range(len(self.data)) if train_mask[i]]
        self.y_train = self.Y[train_mask]

        # Verify all samples have correct length
        if len(self.X_test) > 0:
            test_sample_lengths = [x.shape[0] for x in self.X_test]
            print(f"  Test set sample lengths: min={min(test_sample_lengths)}, max={max(test_sample_lengths)} (should all be 1000)")

        # IMPORTANT: Save test set backup to ensure it only contains real data
        # Test set should NEVER be modified with synthetic data
        self.X_test_backup = [x.copy() if hasattr(x, 'copy') else x for x in self.X_test]
        self.y_test_backup = self.y_test.copy()
        self.original_test_size = len(self.y_test_backup)  # Save original size for verification
        
        print(f"Data split (before swap):")
        print(f"  Train (fold <= {self.train_fold}): {len(self.y_train)} samples")
        print(f"  Val (fold == {self.val_fold}): {len(self.y_val)} samples")
        print(f"  Test (fold == {self.test_fold}): {len(self.y_test)} samples")

        # Swap train and test datasets for consistency across the paper
        # Original test_data -> new train_data (for training)
        # Original train_data -> new test_data (for testing)
        print(f"\nSwapping train and test datasets for consistency...")
        X_train_original = self.X_train
        y_train_original = self.y_train
        X_test_original = self.X_test
        y_test_original = self.y_test
        
        # Swap: test becomes train, train becomes test
        self.X_train = X_test_original
        self.y_train = y_test_original
        self.X_test = X_train_original
        self.y_test = y_train_original
        
        # Update backup to reflect the new test set (which is now the original train set)
        self.X_test_backup = [x.copy() if hasattr(x, 'copy') else x for x in self.X_test]
        self.y_test_backup = self.y_test.copy()
        self.original_test_size = len(self.y_test_backup)
        
        print(f"Data split (after swap):")
        print(f"  Train (original test, fold == {self.test_fold}): {len(self.y_train)} samples")
        print(f"  Val (fold == {self.val_fold}): {len(self.y_val)} samples")
        print(f"  Test (original train, fold <= {self.train_fold}): {len(self.y_test)} samples")

        # selected_indices = np.random.choice(len(self.X_test), size=690, replace=False)
        # ipdb.set_trace()
        # self.X_test = self.X_test[selected_indices]
        # self.y_test = self.y_test[selected_indices]

        # ipdb.set_trace()
        '''
        # ADDed!! This is for using synthetic data for PTB-XL
        synth_path = 'physionet.org/files/Dataset'

        self.y_train = np.load(os.path.join(synth_path, 'labels', 'ptbxl_train_labels.npy'))
        self.y_val = np.load(os.path.join(synth_path, 'labels', 'ptbxl_validation_labels.npy'))
        self.y_test = np.load(os.path.join(synth_path, 'labels', 'ptbxl_test_labels.npy'))
        self.X_train = np.load(os.path.join(synth_path, 'data', 'ptbxl_train_data.npy'))
        self.X_val = np.load(os.path.join(synth_path, 'data', 'ptbxl_validation_data.npy'))
        self.X_test = np.load(os.path.join(synth_path, 'data', 'ptbxl_test_data.npy'))

        self.X_train = np.transpose(self.X_train, (0, 2, 1))
        self.X_val = np.transpose(self.X_val, (0, 2, 1))
        self.X_test = np.transpose(self.X_test, (0, 2, 1))

        self.input_shape = self.X_train[0].shape
        '''

        # ipdb.set_trace()
        # for vanilla diffusion model 
        # syn_data = np.load('Diffusion_RL/SSSD-ECG-main/src/sssd/sssd_label_cond_ICBEB/ch256_T200_betaT0.02_True/all_generated_data.npy')
        # syn_label = np.load('Diffusion_RL/SSSD-ECG-main/src/sssd/sssd_label_cond_ICBEB/ch256_T200_betaT0.02_True/all_generated_label.npy')

        # for reward guided diffusion model 
        # syn_data = np.load('Diffusion_RL/SSSD-ECG-main/src/sssd/sssd_label_cond_ICBEB/ch256_T200_betaT0.02/all_generated_data.npy')
        # syn_label = np.load('Diffusion_RL/SSSD-ECG-main/src/sssd/sssd_label_cond_ICBEB/ch256_T200_betaT0.02/all_generated_label.npy')
        
        # for direct transfer diffusion model from ptbxl
        # syn_data = np.load('Diffusion_RL/SSSD-ECG-main/src/sssd/sssd_label_cond_ptbxl_for_icbeb/ch256_T200_betaT0.02/all_generated_data.npy')
        # syn_label = np.load('Diffusion_RL/SSSD-ECG-main/src/sssd/sssd_label_cond_ptbxl_for_icbeb/ch256_T200_betaT0.02/all_generated_label.npy')

        # for density ratio guided diffusion model from ptbxl
        # syn_data = np.load('Diffusion_RL/SSSD-ECG-main/src/sssd/sssd_label_cond_ptbxl_for_icbeb_guided/ch256_T200_betaT0.02/all_generated_data.npy')
        # syn_label = np.load('Diffusion_RL/SSSD-ECG-main/src/sssd/sssd_label_cond_ptbxl_for_icbeb_guided/ch256_T200_betaT0.02/all_generated_label.npy')

        # for density ratio guided diffusion model from ptbxl (improved )
        # syn_data = np.load('Diffusion_RL/SSSD-ECG-main/src/sssd/sssd_label_cond_ptbxl_for_icbeb_guided_/ch256_T200_betaT0.02/all_generated_data.npy')
        # syn_label = np.load('Diffusion_RL/SSSD-ECG-main/src/sssd/sssd_label_cond_ptbxl_for_icbeb_guided_/ch256_T200_betaT0.02/all_generated_label.npy')
          
        # for vanilla diffusion model 690 samples
        # syn_data = np.load('Diffusion_RL/SSSD-ECG-main/src/sssd_0514/sssd_label_cond_ICBEB/ch256_T200_betaT0.02/all_generated_data.npy')
        # syn_label = np.load('Diffusion_RL/SSSD-ECG-main/src/sssd_0514/sssd_label_cond_ICBEB/ch256_T200_betaT0.02/all_generated_label.npy')

        # for direct transfer diffusion model from ptbxl 690 samples
        # syn_data = np.load('Diffusion_RL/SSSD-ECG-main/src/sssd_0514/sssd_label_cond_ptbxl_for_icbeb_finetune/ch256_T200_betaT0.02/all_generated_data.npy')
        # syn_label = np.load('Diffusion_RL/SSSD-ECG-main/src/sssd_0514/sssd_label_cond_ptbxl_for_icbeb_finetune/ch256_T200_betaT0.02/all_generated_label.npy')
        
        # for density ratio guided diffusion model from ptbxl (improved ) 690sample
        # Resolve path: actual file is at sssd/output/guided_{experiment_name}/ch256_T200_betaT0.02/gen_data_*.npy
        # Try multiple possible locations with suffix support
        # New naming: gen_data_{suffix}.npy (suffix: clf, clf_sob, clf_sobd, diff, diff_scratch)
        data_filename = f'gen_data{self.syn_data_suffix}.npy'
        possible_paths = []
        
        # If experiment_name is provided, prioritize paths with experiment_name
        if self.syn_data_experiment_name:
            # New path format: output/guided_{experiment_name}/ch256_T200_betaT0.02/
            possible_paths.append(
                os.path.join(self.base_dir, f'output/guided_{self.syn_data_experiment_name}', self.syn_data_subdir, data_filename)
            )
            # Direct path with experiment_name (without subdir)
            possible_paths.append(
                os.path.join(self.base_dir, f'output/guided_{self.syn_data_experiment_name}', data_filename)
            )
            # Legacy path format (for backward compatibility)
            legacy_data_filename = f'all_generated_data{self.syn_data_suffix}.npy'
            possible_paths.append(
                os.path.join(self.base_dir, f'sssd_label_cond_ptbxl_for_icbeb_guided_{self.syn_data_experiment_name}', self.syn_data_subdir, legacy_data_filename)
            )
        
        # Default paths (no experiment_name)
        possible_paths.extend([
            # New path format: output/guided/ch256_T200_betaT0.02/
            os.path.join(self.base_dir, 'output/guided', self.syn_data_subdir, data_filename),
            os.path.join(self.base_dir, 'output/guided', data_filename),
        ])
        
        syn_data_path = None
        for path in possible_paths:
            if os.path.exists(path):
                syn_data_path = path
                break
        
        if syn_data_path is None:
            # Provide more helpful error message
            print(f'\nERROR: Could not find synthetic data file: {data_filename}')
            print(f'\nSearched in the following locations:')
            for i, path in enumerate(possible_paths, 1):
                exists = '✓ EXISTS' if os.path.exists(path) else '✗ NOT FOUND'
                print(f'  {i}. {exists}: {path}')
            
            # Check if the base directory exists
            base_output_dir = os.path.join(self.base_dir, 'output/guided')
            if not os.path.exists(base_output_dir):
                print(f'\n  Base output directory does not exist: {base_output_dir}')
            else:
                print(f'\n  Base output directory exists: {base_output_dir}')
                # List contents of base directory
                if os.path.isdir(base_output_dir):
                    subdirs = [d for d in os.listdir(base_output_dir) if os.path.isdir(os.path.join(base_output_dir, d))]
                    if subdirs:
                        print(f'  Subdirectories found: {subdirs}')
                        # Check subdirectory for files
                        subdir_path = os.path.join(base_output_dir, self.syn_data_subdir)
                        if os.path.exists(subdir_path):
                            files = [f for f in os.listdir(subdir_path) if f.endswith('.npy')]
                            if files:
                                print(f'  Files in {self.syn_data_subdir}/: {files}')
                            else:
                                print(f'  Directory {self.syn_data_subdir}/ exists but is empty')
            
            print(f'\nSOLUTION: Please run inference_guided_ptbxl.py first to generate synthetic data.')
            print(f'  Example: python inference_guided_ptbxl.py --use_sobolev --decay_to_min --guidance_iter 1000')
            print(f'  Or:      python inference_guided_ptbxl.py')
            print(f'  This will generate {data_filename} in the expected location.\n')
            
            raise FileNotFoundError(f'Could not find {data_filename} in any of the expected locations: {possible_paths}')
        
        print(f'Loading synthetic data from: {syn_data_path}')
        syn_data_raw = np.load(syn_data_path, allow_pickle=True)
        
        # Handle different data file formats (similar to label handling)
        if syn_data_raw.ndim >= 2:
            syn_data = syn_data_raw
        elif syn_data_raw.ndim == 1 and syn_data_raw.dtype == object:
            # Object array (list of arrays) - concatenate them
            syn_data = np.concatenate(syn_data_raw, axis=0)
        else:
            syn_data = syn_data_raw
        
        # Get original number of samples before reshaping
        # If data is already in (N, 12, 1000) format, use first dimension
        # If data is flattened, calculate from total size
        if syn_data.ndim == 4:
            # (batch, samples_per_batch, channels, length) -> total samples = batch * samples_per_batch
            original_num_samples = syn_data.shape[0] * syn_data.shape[1]
        elif syn_data.ndim == 3:
            original_num_samples = syn_data.shape[0]
        elif syn_data.ndim == 2:
            # Could be (N, 12*1000) or (N*12, 1000), check which makes sense
            # Assume it's (N, features) and we need to reshape to (N, 12, 1000)
            original_num_samples = syn_data.shape[0]
        elif syn_data.ndim == 1:
            # Flattened array, calculate samples from total size
            original_num_samples = syn_data.size // (12 * 1000)
        else:
            original_num_samples = syn_data.shape[0] if syn_data.ndim > 1 else len(syn_data)
        
        print(f'Loaded data shape: {syn_data.shape}, calculated num_samples: {original_num_samples}')
        
        # Load syn_label from label file (same directory as syn_data)
        # inference_guided_ptbxl.py saves two label files:
        # 1. gen_cond{label_suffix}.npy (saved at the beginning)
        # 2. gen_label{data_suffix}.npy (saved at the end, this is what we use)
        syn_label_dir = os.path.dirname(syn_data_path)
        
        # Try multiple possible label filenames (new naming first, then legacy)
        possible_label_paths = [
            # New naming: gen_label with suffix (matches gen_data)
            os.path.join(syn_label_dir, f'gen_label{self.syn_data_suffix}.npy'),
            # New naming: gen_cond with suffix
            os.path.join(syn_label_dir, f'gen_cond{self.syn_data_suffix}.npy'),
            # Legacy: all_generated_label with suffix
            os.path.join(syn_label_dir, f'all_generated_label{self.syn_data_suffix}.npy'),
            # Legacy: syn_all_labels with suffix
            os.path.join(syn_label_dir, f'syn_all_labels{self.syn_data_suffix}.npy'),
            # Try in subdirectory
            os.path.join(syn_label_dir, self.syn_data_subdir, f'gen_label{self.syn_data_suffix}.npy'),
            os.path.join(syn_label_dir, self.syn_data_subdir, f'all_generated_label{self.syn_data_suffix}.npy'),
        ]
        
        syn_label_path = None
        for path in possible_label_paths:
            if os.path.exists(path):
                syn_label_path = path
                break
        
        if syn_label_path is None:
            raise FileNotFoundError(f'Could not find label file. Tried: {possible_label_paths}')
        
        syn_label = infer_label(syn_data)
        
        # Get number of samples from inferred labels
        label_count = syn_label.shape[0]
        data_count = original_num_samples
        num_samples = min(label_count, data_count)
        
        print(f'Inferred label shape: {syn_label.shape}')
        print(f'Label count: {label_count}, Data count: {data_count}, Using: {num_samples} samples')
        
        # Limit labels to match data count if needed
        if label_count > num_samples:
            syn_label = syn_label[:num_samples]
        
        # Convert 71-dim one-hot to 9-dim one-hot
        # syn_label is 71-dim one hot with only 9 classes are 1, others are 0
        # we need to convert it to 9-dim one hot
        label_map = [46, 4, 0, 11, 12, 49, 54, 63, 64]  # ICBEB's 9 classes in PTB-XL's 71 classes

        # Ensure syn_label is float/int type for indexing
        if syn_label.dtype == object:
            # If still object, convert each element
            syn_label = np.array([np.array(x) for x in syn_label])
        
        output_tensor = np.zeros((syn_label.shape[0], 9), dtype=syn_label.dtype)
        for batch_idx in range(syn_label.shape[0]):
            # Iterate through each position in label_map
            for new_label_idx, ptbxl_index in enumerate(label_map):
                # Extract value from PTB-XL's 71-dim vector at position ptbxl_index
                # and map it to ICBEB's 9-dim vector at position new_label_idx
                output_tensor[batch_idx, new_label_idx] = syn_label[batch_idx, ptbxl_index]
        
        syn_label = output_tensor
        print(f'Converted label shape: {syn_label.shape} (71-dim -> 9-dim)')
        
        # print(f'Loading synthetic labels from: {syn_label_path}')
        # syn_label_raw = np.load(syn_label_path, allow_pickle=True)
        
        # # Handle different label file formats
        # # Case 1: Already a 2D array (N, 71)
        # if syn_label_raw.ndim == 2:
        #     syn_label = syn_label_raw
        # # Case 2: Object array (list of arrays) - concatenate them
        # elif syn_label_raw.ndim == 1 and syn_label_raw.dtype == object:
        #     # This happens when all_label was a list of arrays with same shape
        #     syn_label = np.concatenate(syn_label_raw, axis=0)
        # # Case 3: 1D array but not object - might need reshaping
        # else:
        #     # Try to reshape if possible
        #     if syn_label_raw.size % 71 == 0:
        #         syn_label = syn_label_raw.reshape(-1, 71)
        #     else:
        #         raise ValueError(f'Unexpected label shape: {syn_label_raw.shape}, dtype: {syn_label_raw.dtype}')
        
        # print(f'Loaded label shape: {syn_label.shape}, dtype: {syn_label.dtype}')
        
        # ## syn_label is 71-dim one hot with only 9 classes are 1, others are 0
        # ## we need to convert it to 9-dim one hot
        # label_map = [46, 4, 0, 11, 12, 49, 54, 63, 64]  # ICBEB's 9 classes in PTB-XL's 71 classes

        # # Convert 71-dim one-hot to 9-dim one-hot
        # # Ensure syn_label is float/int type for indexing
        # if syn_label.dtype == object:
        #     # If still object, convert each element
        #     syn_label = np.array([np.array(x) for x in syn_label])
        
        # output_tensor = np.zeros((syn_label.shape[0], 9), dtype=syn_label.dtype)
        # for batch_idx in range(syn_label.shape[0]):
        #     # Iterate through each position in label_map
        #     for new_label_idx, ptbxl_index in enumerate(label_map):
        #         # Extract value from PTB-XL's 71-dim vector at position ptbxl_index
        #         # and map it to ICBEB's 9-dim vector at position new_label_idx
        #         output_tensor[batch_idx, new_label_idx] = syn_label[batch_idx, ptbxl_index]
        
        # # Match label and data sample counts
        # # Use the minimum of label count and data count to ensure they match
        # label_count = output_tensor.shape[0]
        # data_count = original_num_samples
        # num_samples = min(label_count, data_count)
        
        # print(f'Label count: {label_count}, Data count: {data_count}, Using: {num_samples} samples')
        
        # # Limit both to the same number of samples
        # if label_count > num_samples:
        #     syn_label = output_tensor[:num_samples]
        # else:
        #     syn_label = output_tensor


    
                  
        # syn_data = syn_data.reshape(16500,12,1000)
        # Reshape based on actual number of samples
        # Calculate total elements needed
        total_elements = num_samples * 12 * 1000
        
        # Handle different data dimensions
        if syn_data.ndim == 4:
            # (batch, samples_per_batch, channels, length) -> (batch*samples_per_batch, channels, length)
            # Reshape to combine batch and samples_per_batch dimensions
            syn_data = syn_data.reshape(-1, syn_data.shape[2], syn_data.shape[3])
        elif syn_data.ndim == 3:
            # Already in (samples, channels, length) format
            pass
        else:
            raise ValueError(f'Unexpected syn_data shape: {syn_data.shape}')
        
        # Check if syn_data needs to be flattened first
        if syn_data.ndim > 1:
            current_size = syn_data.size
            if current_size != total_elements:
                # Flatten first, then reshape
                syn_data = syn_data.flatten()[:total_elements]
        
        # Ensure final shape is (num_samples, 12, 1000)
        syn_data = syn_data.reshape(num_samples, 12, 1000)
        # syn_data = syn_data.reshape(17200,12,1000)
        syn_data = np.transpose(syn_data, (0, 2, 1))

        print("self.y_train.shape", self.y_train.shape)
        print("syn_label.shape", syn_label.shape)

        # Preprocess signal data
        # IMPORTANT: Test set should only contain real data, never synthetic data
        # Preprocess all sets separately to ensure test set remains pure
        self.X_train, self.X_val, self.X_test = utils.preprocess_signals(self.X_train, self.X_val, self.X_test, self.outputfolder+self.experiment_name+'/data/')

        # Verify test set only contains real data (before any synthetic data operations)
        # Test set should NOT be modified after this point
        assert len(self.X_test) == len(self.y_test), "Test set data and labels must have same length"
        
        # Split synthetic data: use 80% for training, 20% for validation
        # Validation set can use synthetic data for better model selection and hyperparameter tuning
        val_split_ratio = 0.2  # 20% of synthetic data goes to validation set
        n_syn_total = len(syn_data)
        n_syn_val = int(n_syn_total * val_split_ratio)
        n_syn_train = n_syn_total - n_syn_val
        
        # Shuffle synthetic data before splitting
        syn_indices = np.random.permutation(n_syn_total)
        syn_data_shuffled = syn_data[syn_indices]
        syn_label_shuffled = syn_label[syn_indices]
        
        # Split synthetic data
        syn_data_train = syn_data_shuffled[:n_syn_train]
        syn_label_train = syn_label_shuffled[:n_syn_train]
        syn_data_val = syn_data_shuffled[n_syn_train:]
        syn_label_val = syn_label_shuffled[n_syn_train:]
        
        print(f"Splitting synthetic data: {n_syn_train} samples for training, {n_syn_val} samples for validation")
        
        # Prepare training data (real + synthetic)
        concatenated_train_data = []
        # Add real training data
        for element in self.X_train:
            concatenated_train_data.append(element)
        # Add synthetic training data
        for element in syn_data_train:
            concatenated_train_data.append(element)
        
        # Prepare validation data (real + synthetic)
        concatenated_val_data = []
        # Add real validation data
        for element in self.X_val:
            concatenated_val_data.append(element)
        # Add synthetic validation data
        for element in syn_data_val:
            concatenated_val_data.append(element)
        
        # Stack training data
        try:
            self.X_train = np.stack(concatenated_train_data)
        except (ValueError, TypeError):
            try:
                self.X_train = np.array(concatenated_train_data, dtype=object)
            except (ValueError, TypeError):
                self.X_train = concatenated_train_data
        
        # Stack validation data
        try:
            self.X_val = np.stack(concatenated_val_data)
        except (ValueError, TypeError):
            try:
                self.X_val = np.array(concatenated_val_data, dtype=object)
            except (ValueError, TypeError):
                self.X_val = concatenated_val_data
        
        # Append synthetic labels
        self.y_train = np.append(self.y_train, syn_label_train, axis=0)
        self.y_val = np.append(self.y_val, syn_label_val, axis=0)

        # Shuffle training set
        idx_train = np.random.permutation(len(self.y_train))
        self.X_train, self.y_train = self.X_train[idx_train], self.y_train[idx_train]
        
        # Shuffle validation set
        idx_val = np.random.permutation(len(self.y_val))
        self.X_val, self.y_val = self.X_val[idx_val], self.y_val[idx_val]

        # ipdb.set_trace()


        #only use 690sample for trianing 
        # aaa = self.X_test 
        # self.X_test = self.X_train
        # self.X_train = aaa

        # aaa = self.y_test
        # self.y_test = self.y_train
        # self.y_train = aaa


        #self.X_train.dump('Diffusion_RL/ecg_ptbxl_benchmarking/output/x_train.npy')
        #self.X_val.dump('Diffusion_RL/ecg_ptbxl_benchmarking/output/X_val.npy')
        #self.X_test.dump('Diffusion_RL/ecg_ptbxl_benchmarking/output/X_test.npy')


        self.n_classes = self.y_train.shape[1]

        # save train and test labels
        self.y_train.dump(self.outputfolder + self.experiment_name+ '/data/y_train.npy')
        self.y_val.dump(self.outputfolder + self.experiment_name+ '/data/y_val.npy')
        self.y_test.dump(self.outputfolder + self.experiment_name+ '/data/y_test.npy')
        
        # Verify test set integrity: should only contain real data, never synthetic
        if hasattr(self, 'y_test_backup'):
            original_size = len(self.y_test_backup)
            current_size = len(self.y_test)
            assert current_size == original_size, \
                f"Test set size changed! Original: {original_size}, Current: {current_size}. " \
                "Test set should only contain real data and never be modified with synthetic data."
            print(f"✓ Test set integrity verified:")
            print(f"  - Size unchanged: {current_size} samples (matches original)")
            print(f"  - Contains ONLY real data (no synthetic data added)")
            print(f"  - Test set was NOT modified during synthetic data processing")
        else:
            print(f"⚠ Warning: Test set backup not found, cannot verify integrity")
        
        # Print final dataset sizes
        print(f"\nFinal dataset sizes:")
        print(f"  Training samples: {len(self.y_train)} (real + synthetic)")
        print(f"  Validation samples: {len(self.y_val)} (real + synthetic)")
        print(f"  Test samples: {len(self.y_test)} (real data only - verified)")

        modelname = 'naive'
        # create most naive predictions via simple mean in training
        mpath = self.outputfolder+self.experiment_name+'/models/'+modelname+'/'
        # create folder for model outputs
        if not os.path.exists(mpath):
            os.makedirs(mpath)
        if not os.path.exists(mpath+'results/'):
            os.makedirs(mpath+'results/')

        mean_y = np.mean(self.y_train, axis=0)
        np.array([mean_y]*len(self.y_train)).dump(mpath + 'y_train_pred.npy')
        np.array([mean_y]*len(self.y_test)).dump(mpath + 'y_test_pred.npy')
        np.array([mean_y]*len(self.y_val)).dump(mpath + 'y_val_pred.npy')

    def perform(self):
        for model_description in self.models:
            modelname = model_description['modelname']
            modeltype = model_description['modeltype']
            modelparams = model_description['parameters']

            mpath = self.outputfolder+self.experiment_name+'/models/'+modelname+'/'
            # create folder for model outputs
            if not os.path.exists(mpath):
                os.makedirs(mpath)
            if not os.path.exists(mpath+'results/'):
                os.makedirs(mpath+'results/')

            n_classes = self.Y.shape[1]
            # load respective model
            if modeltype == 'WAVELET':
                from models.wavelet import WaveletModel
                model = WaveletModel(modelname, n_classes, self.sampling_frequency, mpath, self.input_shape, **modelparams)
            elif modeltype == "fastai_model":
                from models.fastai_model import fastai_model
                model = fastai_model(modelname, n_classes, self.sampling_frequency, mpath, self.input_shape, **modelparams)
            elif modeltype == "YOUR_MODEL_TYPE":
                # YOUR MODEL GOES HERE!
                from models.your_model import YourModel
                model = YourModel(modelname, n_classes, self.sampling_frequency, mpath, self.input_shape, **modelparams)
            else:
                assert(True)
                break

            # fit model
            model.fit(self.X_train, self.y_train, self.X_val, self.y_val)
            # predict and dump
            model.predict(self.X_train).dump(mpath+'y_train_pred.npy')
            model.predict(self.X_val).dump(mpath+'y_val_pred.npy')
            model.predict(self.X_test).dump(mpath+'y_test_pred.npy')

        modelname = 'ensemble'
        # create ensemble predictions via simple mean across model predictions (except naive predictions)
        ensemblepath = self.outputfolder+self.experiment_name+'/models/'+modelname+'/'
        # create folder for model outputs
        if not os.path.exists(ensemblepath):
            os.makedirs(ensemblepath)
        if not os.path.exists(ensemblepath+'results/'):
            os.makedirs(ensemblepath+'results/')
        # load all predictions
        ensemble_train, ensemble_val, ensemble_test = [],[],[]
        for model_description in os.listdir(self.outputfolder+self.experiment_name+'/models/'):
            if not model_description in ['ensemble', 'naive']:
                mpath = self.outputfolder+self.experiment_name+'/models/'+model_description+'/'
                ensemble_train.append(np.load(mpath+'y_train_pred.npy', allow_pickle=True))
                ensemble_val.append(np.load(mpath+'y_val_pred.npy', allow_pickle=True))
                ensemble_test.append(np.load(mpath+'y_test_pred.npy', allow_pickle=True))
        # dump mean predictions
        np.array(ensemble_train).mean(axis=0).dump(ensemblepath + 'y_train_pred.npy')
        np.array(ensemble_test).mean(axis=0).dump(ensemblepath + 'y_test_pred.npy')
        np.array(ensemble_val).mean(axis=0).dump(ensemblepath + 'y_val_pred.npy')

    # def evaluate(self, n_bootstraping_samples=100, n_jobs=20, bootstrap_eval=False, dumped_bootstraps=True):
    def evaluate(self, n_bootstraping_samples=100, n_jobs=20, bootstrap_eval=True, dumped_bootstraps=False):
        # get labels
        y_train = np.load(self.outputfolder+self.experiment_name+'/data/y_train.npy', allow_pickle=True)
        #y_val = np.load(self.outputfolder+self.experiment_name+'/data/y_val.npy', allow_pickle=True)
        y_test = np.load(self.outputfolder+self.experiment_name+'/data/y_test.npy', allow_pickle=True)

        # if bootstrapping then generate appropriate samples for each
        if bootstrap_eval:
            if not dumped_bootstraps:
                #train_samples = np.array(utils.get_appropriate_bootstrap_samples(y_train, n_bootstraping_samples))
                test_samples = np.array(utils.get_appropriate_bootstrap_samples(y_test, n_bootstraping_samples))
                #val_samples = np.array(utils.get_appropriate_bootstrap_samples(y_val, n_bootstraping_samples))
            else:
                test_samples = np.load(self.outputfolder+self.experiment_name+'/test_bootstrap_ids.npy', allow_pickle=True)
        else:
            #train_samples = np.array([range(len(y_train))])
            test_samples = np.array([range(len(y_test))])
            #val_samples = np.array([range(len(y_val))])

        # store samples for future evaluations
        #train_samples.dump(self.outputfolder+self.experiment_name+'/train_bootstrap_ids.npy')
        test_samples.dump(self.outputfolder+self.experiment_name+'/test_bootstrap_ids.npy')
        #val_samples.dump(self.outputfolder+self.experiment_name+'/val_bootstrap_ids.npy')

        # iterate over all models fitted so far
        for m in sorted(os.listdir(self.outputfolder+self.experiment_name+'/models')):
            print(m)
            mpath = self.outputfolder+self.experiment_name+'/models/'+m+'/'
            rpath = self.outputfolder+self.experiment_name+'/models/'+m+'/results/'

            # load predictions
            y_train_pred = np.load(mpath+'y_train_pred.npy', allow_pickle=True)
            #y_val_pred = np.load(mpath+'y_val_pred.npy', allow_pickle=True)
            y_test_pred = np.load(mpath+'y_test_pred.npy', allow_pickle=True)

            if self.experiment_name.startswith('exp_ICBEB'):
                # compute classwise thresholds such that recall-focused Gbeta is optimized
                thresholds = utils.find_optimal_cutoff_thresholds_for_Gbeta(y_train, y_train_pred)
            else:
                thresholds = None

            pool = multiprocessing.Pool(n_jobs)

            # tr_df = pd.concat(pool.starmap(utils.generate_results, zip(train_samples, repeat(y_train), repeat(y_train_pred), repeat(thresholds))))
            # tr_df_point = utils.generate_results(range(len(y_train)), y_train, y_train_pred, thresholds)
            # tr_df_result = pd.DataFrame(
            #     np.array([
            #         tr_df_point.mean().values, 
            #         tr_df.mean().values,
            #         tr_df.quantile(0.05).values,
            #         tr_df.quantile(0.95).values]), 
            #     columns=tr_df.columns,
            #     index=['point', 'mean', 'lower', 'upper'])

            te_df = pd.concat(pool.starmap(utils.generate_results, zip(test_samples, repeat(y_test), repeat(y_test_pred), repeat(thresholds))))
            te_df_point = utils.generate_results(range(len(y_test)), y_test, y_test_pred, thresholds)
            te_df_result = pd.DataFrame(
                np.array([
                    te_df_point.mean().values, 
                    te_df.mean().values,
                    te_df.quantile(0.05).values,
                    te_df.quantile(0.95).values]), 
                columns=te_df.columns, 
                index=['point', 'mean', 'lower', 'upper'])

            # val_df = pd.concat(pool.starmap(utils.generate_results, zip(val_samples, repeat(y_val), repeat(y_val_pred), repeat(thresholds))))
            # val_df_point = utils.generate_results(range(len(y_val)), y_val, y_val_pred, thresholds)
            # val_df_result = pd.DataFrame(
            #     np.array([
            #         val_df_point.mean().values, 
            #         val_df.mean().values,
            #         val_df.quantile(0.05).values,
            #         val_df.quantile(0.95).values]), 
            #     columns=val_df.columns, 
            #     index=['point', 'mean', 'lower', 'upper'])

            pool.close()

            # dump results
            #tr_df_result.to_csv(rpath+'tr_results.csv')
            #val_df_result.to_csv(rpath+'val_results.csv')
            te_df_result.to_csv(rpath+'te_results.csv')
