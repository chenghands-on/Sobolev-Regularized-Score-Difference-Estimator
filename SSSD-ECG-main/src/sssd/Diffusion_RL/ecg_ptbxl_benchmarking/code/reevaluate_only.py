from experiments.scp_experiment import SCP_Experiment
from util_utility import utils
# model configs
from configs.fastai_configs import *
from configs.wavelet_configs import *

def main():
    datafolder_icbeb = 'Diffusion_RL/ecg_ptbxl_benchmarking/data/ICBEB/'
    outputfolder = 'Diffusion_RL/ecg_ptbxl_benchmarking/output/'
    
    models = [
        conf_fastai_xresnet1d50,
    ]
    
    syn_data_suffix = '_guidance_sobolev'
    experiment_name = f'exp_ICBEB{syn_data_suffix}'
    
    # Create experiment object (need to call prepare to set up experiment_name and paths)
    # But we can skip the data loading since files already exist
    e = SCP_Experiment(experiment_name, 'all', datafolder_icbeb, outputfolder, models,
                       syn_data_suffix=syn_data_suffix,
                       syn_data_subdir='ch256_T200_betaT0.02')
    
    # Only run evaluation (assumes prepare() and perform() were already run)
    # evaluate() loads data from saved files, so we don't need to call prepare()
    print(f"Re-evaluating experiment: {experiment_name}")
    e.evaluate()
    
    # Generate summary table
    print(f"Generating summary table...")
    utils.ICBEBE_table(experiment_name=experiment_name)
    print("Done!")

if __name__ == "__main__":
    main()

