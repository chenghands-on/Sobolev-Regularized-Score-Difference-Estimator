import os
import argparse
import json
import numpy as np
import torch
import time
import random

from utils.util_generation import find_max_epoch, print_size, sampling_label_guided, sampling_label, calc_diffusion_hyperparams, bandit_get_args
from models_new.SSSD_ECG import SSSD_ECG

from density_ratio_guidance import Bandit_Critic_Guide

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


def generate_four_leads(tensor):
    leadI = tensor[:,0,:].unsqueeze(1)
    leadschest = tensor[:,1:7,:]
    leadavf = tensor[:,7,:].unsqueeze(1)

    leadII = (0.5*leadI) + leadavf

    leadIII = -(0.5*leadI) + leadavf
    leadavr = -(0.75*leadI) -(0.5*leadavf)
    leadavl = (0.75*leadI) - (0.5*leadavf)

    leads12 = torch.cat([leadI, leadII, leadschest, leadIII, leadavr, leadavl, leadavf], dim=1)

    return leads12


def generate(output_directory,
             num_samples,
             ckpt_path,
             data_path,
             ckpt_iter,
             use_sobolev=False,
             no_decay=False,
             guidance_iter=1000,
             no_guidance=False,
             experiment_name=None):
    
    
    """
    Generate data based on ground truth 

    Parameters:
    output_directory (str):           save generated speeches to this path
    num_samples (int):                number of samples to generate, default is 4
    ckpt_path (str):                  checkpoint path
    ckpt_iter (int or 'max'):         the pretrained checkpoint to be loaded; 
                                      automitically selects the maximum iteration if 'max' is selected
    data_path (str):                  path to dataset, numpy array.
    """

    # generate experiment (local) path
    local_path = "ch{}_T{}_betaT{}".format(model_config["res_channels"], 
                                           diffusion_config["T"], 
                                           diffusion_config["beta_T"])

    # Add experiment name suffix to base output directory if provided
    # This creates: output/guided_{experiment_name}/ch256_T200_betaT0.02/
    if experiment_name:
        output_directory = f"output/guided_{experiment_name}"
    else:
        output_directory = "output/guided"
    
    # Get shared output_directory ready
    output_directory = os.path.join(output_directory, local_path)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    # Set multiple devices based on visible GPUs (CUDA_VISIBLE_DEVICES)
    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count == 0:
        raise RuntimeError("No CUDA devices available.")
    gpu_ids_env = os.getenv("GPU_IDS", "").strip()
    if gpu_ids_env:
        gpu_ids = [int(x) for x in gpu_ids_env.split(",") if x.strip() != ""]
        if any(i < 0 or i >= visible_gpu_count for i in gpu_ids):
            raise ValueError(
                f"GPU_IDS={gpu_ids_env} out of range for visible devices (0..{visible_gpu_count-1})."
            )
    else:
        gpu_ids = list(range(visible_gpu_count))
    devices = [torch.device(f"cuda:{i}") for i in gpu_ids]
    device = devices[0]  # Main device for guidance_net and other operations
    
    print(f"Using multiple GPUs: {[str(d) for d in devices]} (visible count={visible_gpu_count})")
    
    # map diffusion hyperparameters to all gpus
    # Create copies for each device
    diffusion_hyperparams_per_device = {}
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams_per_device[key] = {}
            for dev in devices:
                diffusion_hyperparams_per_device[key][dev] = diffusion_hyperparams[key].to(dev)
        else:
            diffusion_hyperparams_per_device[key] = diffusion_hyperparams[key]
    
    # Also keep original on main device for compatibility
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].to(device)

    # load checkpoint before wrapping with DataParallel
    # Use unified model weight directory
    # Get script directory to resolve relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_weight_dir = os.path.join(script_dir, 'Diffusion_RL', 'ecg_ptbxl_benchmarking', 'model_weight')
    
    # We'll load the checkpoint first to detect label_embed_classes, then initialize model
    # This allows us to automatically adjust model config based on the saved model
    
    # Track if scratch model is used (for saving with scratch suffix)
    is_scratch = False
    
    # Helper function to find max epoch (defined here so it's available in both branches)
    def find_max_diff_epoch(path, scratch=False, exp_name=None):
        """Find maximum epoch in diff_*.pkl or diff_scratch_*.pkl files, optionally with experiment_name"""
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
    
    if no_guidance:
        # For direct diffusion model generation (no guidance), use diff_*.pkl or diff_scratch_*.pkl files
        # Check if user specified a scratch model by checking if ckpt_iter is a string containing 'scratch'
        if isinstance(ckpt_iter, str) and 'scratch' in ckpt_iter.lower():
            # Extract number from string like "scratch_4000" or "4000" from "diff_scratch_4000"
            try:
                if 'scratch' in ckpt_iter:
                    ckpt_iter = int(ckpt_iter.split('_')[-1]) if '_' in ckpt_iter else int(ckpt_iter.replace('scratch', '').strip('_'))
                is_scratch = True
            except:
                pass
        
        if ckpt_iter == 'max':
            # Try scratch first with experiment_name, then without, then regular with experiment_name, then without
            if experiment_name:
                ckpt_iter = find_max_diff_epoch(model_weight_dir, scratch=True, exp_name=experiment_name)
                if ckpt_iter >= 0:
                    is_scratch = True
                else:
                    ckpt_iter = find_max_diff_epoch(model_weight_dir, scratch=False, exp_name=experiment_name)
                    if ckpt_iter >= 0:
                        is_scratch = False
                    else:
                        # Try without experiment_name
                        ckpt_iter = find_max_diff_epoch(model_weight_dir, scratch=True, exp_name=None)
                        if ckpt_iter >= 0:
                            is_scratch = True
                        else:
                            ckpt_iter = find_max_diff_epoch(model_weight_dir, scratch=False, exp_name=None)
                            if ckpt_iter >= 0:
                                is_scratch = False
            else:
                # No experiment_name specified
                ckpt_iter = find_max_diff_epoch(model_weight_dir, scratch=True, exp_name=None)
                if ckpt_iter >= 0:
                    is_scratch = True
                else:
                    ckpt_iter = find_max_diff_epoch(model_weight_dir, scratch=False, exp_name=None)
                    if ckpt_iter >= 0:
                        is_scratch = False
            
            if ckpt_iter < 0:
                raise Exception('No diff_*.pkl or diff_scratch_*.pkl files found in {}'.format(model_weight_dir))
            
            # Build model_path after finding max iteration
            if is_scratch:
                if experiment_name:
                    model_path = os.path.join(model_weight_dir, 'diff_scratch_{}_{}.pkl'.format(experiment_name, ckpt_iter))
                else:
                    model_path = os.path.join(model_weight_dir, 'diff_scratch_{}.pkl'.format(ckpt_iter))
            else:
                if experiment_name:
                    model_path = os.path.join(model_weight_dir, 'diff_{}_{}.pkl'.format(experiment_name, ckpt_iter))
                else:
                    model_path = os.path.join(model_weight_dir, 'diff_{}.pkl'.format(ckpt_iter))
        else:
            # Try to load the specified checkpoint, check both scratch and regular
            # Priority: with experiment_name first, then without
            model_path = None
            if experiment_name:
                # Try with experiment_name first
                scratch_path = os.path.join(model_weight_dir, 'diff_scratch_{}_{}.pkl'.format(experiment_name, ckpt_iter))
                regular_path = os.path.join(model_weight_dir, 'diff_{}_{}.pkl'.format(experiment_name, ckpt_iter))
                if os.path.exists(scratch_path):
                    is_scratch = True
                    model_path = scratch_path
                elif os.path.exists(regular_path):
                    is_scratch = False
                    model_path = regular_path
            
            # If not found with experiment_name, try without
            if model_path is None or not os.path.exists(model_path):
                scratch_path = os.path.join(model_weight_dir, 'diff_scratch_{}.pkl'.format(ckpt_iter))
                regular_path = os.path.join(model_weight_dir, 'diff_{}.pkl'.format(ckpt_iter))
                if os.path.exists(scratch_path):
                    is_scratch = True
                    model_path = scratch_path
                elif os.path.exists(regular_path):
                    is_scratch = False
                    model_path = regular_path
                else:
                    # Neither model exists, raise error
                    if experiment_name:
                        raise Exception('No model found for iteration {}: checked diff_scratch_{}_{}.pkl, diff_{}_{}.pkl, diff_scratch_{}.pkl, and diff_{}.pkl in {}'.format(
                            ckpt_iter, experiment_name, ckpt_iter, experiment_name, ckpt_iter, ckpt_iter, ckpt_iter, model_weight_dir))
                    else:
                        raise Exception('No model found for iteration {}: neither diff_scratch_{}.pkl nor diff_{}.pkl exists in {}'.format(
                            ckpt_iter, ckpt_iter, ckpt_iter, model_weight_dir))
        
        if not 'model_path' in locals() or model_path is None:
            if is_scratch:
                if experiment_name:
                    model_path = os.path.join(model_weight_dir, 'diff_scratch_{}_{}.pkl'.format(experiment_name, ckpt_iter))
                else:
                    model_path = os.path.join(model_weight_dir, 'diff_scratch_{}.pkl'.format(ckpt_iter))
            else:
                if experiment_name:
                    model_path = os.path.join(model_weight_dir, 'diff_{}_{}.pkl'.format(experiment_name, ckpt_iter))
                else:
                    model_path = os.path.join(model_weight_dir, 'diff_{}.pkl'.format(ckpt_iter))
        
        print('Using direct diffusion model (no guidance){}'.format(' [scratch]' if is_scratch else ''))
    else:
        # For guidance-based generation, use regular *.pkl files (pretrained model)
        # Support experiment_name for base diffusion model
        if ckpt_iter == 'max':
            # Try to find with experiment_name first, then without
            if experiment_name:
                # Try diff_{experiment_name}_*.pkl files
                ckpt_iter = find_max_diff_epoch(model_weight_dir, scratch=False, exp_name=experiment_name)
                if ckpt_iter < 0:
                    # Fall back to regular *.pkl files
                    ckpt_iter = find_max_epoch(model_weight_dir)
            else:
                ckpt_iter = find_max_epoch(model_weight_dir)
        
        # Build model path with experiment_name support
        if experiment_name:
            # Try with experiment_name first
            model_path = os.path.join(model_weight_dir, 'diff_{}_{}.pkl'.format(experiment_name, ckpt_iter))
            if not os.path.exists(model_path):
                # Fall back to regular name
                model_path = os.path.join(model_weight_dir, '{}.pkl'.format(ckpt_iter))
        else:
            model_path = os.path.join(model_weight_dir, '{}.pkl'.format(ckpt_iter))
        print('Using guidance-based generation')
    
    # Load checkpoint to detect label_embed_classes from the saved model
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        # Remove 'module.' prefix if present (in case checkpoint was saved with DataParallel)
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Auto-detect label_embed_classes from the saved model
        if 'embedding.weight' in state_dict:
            detected_label_classes = state_dict['embedding.weight'].shape[0]
            if model_config.get('label_embed_classes') != detected_label_classes:
                print(f"Auto-adjusting label_embed_classes: {model_config.get('label_embed_classes')} -> {detected_label_classes} (from saved model)")
                model_config['label_embed_classes'] = detected_label_classes
        else:
            print("Warning: 'embedding.weight' not found in checkpoint, using config value")
    except Exception as e:
        print(f'Error loading checkpoint for dimension detection: {e}')
        raise Exception('No valid model found at {}'.format(model_path))
    
    # Now initialize model with the (possibly adjusted) config
    net = SSSD_ECG(**model_config).to(device)
    
    # Load the state dict into the model
    try:
        net.load_state_dict(state_dict)
        print('Successfully loaded model at iteration {} from {}'.format(ckpt_iter, model_path))
    except Exception as e:
        print(f'Error loading checkpoint: {e}')
        raise Exception('No valid model found at {}'.format(model_path))
    
    # Wrap model with DataParallel for multi-GPU inference after loading weights
    net = torch.nn.DataParallel(net, device_ids=gpu_ids)
    
    # Initialize guidance network only if not using no_guidance mode
    guidance_net = None
    guidance_suffix = ''
    
    if not no_guidance:
        args = bandit_get_args()
        args.device = device  # Set device to cuda:4 (main device)
        
        # Validate that no_decay is only used with use_sobolev
        if no_decay and not use_sobolev:
            raise ValueError("--no_decay can only be used with --use_sobolev")
        
        # Determine guidance mode suffix (used for model loading and file naming)
        # Naming: clf (naive), clf_sob (Sobolev no decay), clf_sobd (Sobolev decay - default)
        if use_sobolev:
            # Add suffix based on no_decay option (decay is now default)
            decay_to_min = not no_decay  # Default: decay enabled
            if decay_to_min:
                guidance_suffix = 'clf_sobd'
                print("Using Bandit_Critic_Guide mode with Sobolev penalty (decay to 1e-2, default)")
            else:
                guidance_suffix = 'clf_sob'
                print("Using Bandit_Critic_Guide mode with Sobolev penalty (keep at 5.0)")
            # Both sobolev modes use the same Bandit_Critic_Guide architecture
            guidance_net = Bandit_Critic_Guide(3904, 0, args)
        else:
            guidance_suffix = 'clf'
            print("Using Bandit_Critic_Guide mode (default)")
            guidance_net = Bandit_Critic_Guide(3904, 0, args)
        
        guidance_net = guidance_net.to(device)
        
        # Load guidance model weights
        # Support experiment_name in model filename (matching train_density_ratio_net.py)
        if experiment_name:
            guidance_model_path = os.path.join(model_weight_dir, '{}_{}_{}.pkl'.format(guidance_suffix, experiment_name, guidance_iter))
        else:
            guidance_model_path = os.path.join(model_weight_dir, '{}_{}.pkl'.format(guidance_suffix, guidance_iter))
        
        try:
            guidance_checkpoint = torch.load(guidance_model_path, map_location='cpu')
            guidance_net.qt.load_state_dict(guidance_checkpoint['model_state_dict'])
            print(f'Successfully loaded {guidance_suffix} model at iteration {guidance_iter} from {guidance_model_path}')
        except Exception as e:
            print(f'Error loading guidance checkpoint: {e}')
            # Try alternative path if experiment_name was provided but file doesn't exist
            if experiment_name:
                alt_path = os.path.join(model_weight_dir, '{}_{}.pkl'.format(guidance_suffix, guidance_iter))
                print(f'Trying alternative path without experiment_name: {alt_path}')
                try:
                    guidance_checkpoint = torch.load(alt_path, map_location='cpu')
                    guidance_net.qt.load_state_dict(guidance_checkpoint['model_state_dict'])
                    print(f'Successfully loaded {guidance_suffix} model at iteration {guidance_iter} from {alt_path}')
                except:
                    raise Exception(f'No valid {guidance_suffix} model found at {guidance_model_path} or {alt_path}')
            else:
                raise Exception(f'No valid {guidance_suffix} model found at {guidance_model_path}')

    print_size(net)

    # Generate random labels for conditional generation
    # Get label_embed_classes from model_config (already auto-adjusted from saved model)
    label_embed_classes = model_config.get('label_embed_classes', 71)
    
    # ICBEB has 9 classes which are a subset of PTB-XL's 71 classes
    # label_map maps ICBEB's 9 classes to PTB-XL's 71 classes indices
    label_map = [46, 4, 0, 11, 12, 49, 54, 63, 64]  # ICBEB's 9 classes in PTB-XL's 71 classes
    
    print(f'Generating {num_samples} random single-label labels for ICBEB\'s 9 classes.')
    print(f'Model label_embed_classes: {label_embed_classes}')
    
    # Note: Random seed should already be set at the beginning of the script
    if label_embed_classes == 9:
        # For scratch models trained with 9-dimensional labels (ICBEB original)
        print(f'Using 9-dimensional labels (ICBEB original format).')
        labels = np.zeros((num_samples, 9), dtype=np.float32)
        for i in range(num_samples):
            # Each sample randomly selects ONE of the 9 ICBEB classes
            selected_icbeb_class = np.random.randint(0, 9)  # Random index from 0 to 8
            labels[i, selected_icbeb_class] = 1.0
    elif label_embed_classes == 71:
        # For finetune models trained with 71-dimensional labels (PTB-XL format)
        print(f'Using 71-dimensional labels (PTB-XL format with ICBEB mapping).')
        print(f'ICBEB label map (indices in PTB-XL): {label_map}')
        labels = np.zeros((num_samples, 71), dtype=np.float32)
        for i in range(num_samples):
            # Each sample randomly selects ONE of the 9 ICBEB classes
            selected_icbeb_class = np.random.randint(0, 9)  # Random index from 0 to 8
            # Map ICBEB's selected class to PTB-XL's 71 classes
            ptbxl_index = label_map[selected_icbeb_class]
            labels[i, ptbxl_index] = 1.0
    else:
        raise ValueError(f"Unsupported label_embed_classes: {label_embed_classes}. Expected 9 or 71.")
    
    print(f'Generated random labels shape: {labels.shape}')
    
    # Save generated labels for future use
    # Save in the same directory as the generated data (output_directory)
    # Include guidance suffix in filename to distinguish between guidance modes
    # Naming: gen_cond_clf.npy, gen_cond_clf_sob.npy, gen_cond_clf_sobd.npy, gen_cond_diff.npy
    if no_guidance:
        label_suffix = '_diff'
    else:
        label_suffix = f'_{guidance_suffix}'
    label_save_path = os.path.join(output_directory, f'gen_cond{label_suffix}.npy')
    os.makedirs(output_directory, exist_ok=True)
    np.save(label_save_path, labels)
    print(f'Saved generated labels to: {label_save_path}')

    all_generated_data = []
    all_label = []
    batch_for_sampling = int(os.getenv("BATCH_FOR_SAMPLING", "256"))
    if batch_for_sampling <= 0:
        raise ValueError(f"BATCH_FOR_SAMPLING must be positive, got {batch_for_sampling}")
    total_batches = num_samples // batch_for_sampling - 1
    actual_samples = total_batches * batch_for_sampling
    
    print(f'\nStarting generation:')
    print(f'  Total requested samples: {num_samples}')
    print(f'  Batch size: {batch_for_sampling}')
    print(f'  Total batches: {total_batches}')
    print(f'  Actual samples to generate: {actual_samples}')
    print(f'  Progress:')
    
    start_time = time.time()
    batch_times = []
    
    for i in range(total_batches):
        batch_start_time = time.time()
        label = labels[i*batch_for_sampling:(i+1)*batch_for_sampling]
        cond = torch.from_numpy(label).to(device).float()

        # inference
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    
        # Use guided or unguided sampling based on no_guidance flag
        try:
            if no_guidance:
                generated_audio = sampling_label(net, (batch_for_sampling,8,1000), 
                                    diffusion_hyperparams,
                                    cond=cond)
            else:
                generated_audio = sampling_label_guided(net, guidance_net, (batch_for_sampling,8,1000), 
                                    diffusion_hyperparams,
                                    cond=cond)

            # Force sync here to surface any CUDA errors from sampling
            for dev in devices:
                torch.cuda.synchronize(dev)

            generated_audio12 = generate_four_leads(generated_audio)

            # Force sync after post-processing as well
            for dev in devices:
                torch.cuda.synchronize(dev)
        except RuntimeError as exc:
            print(f"\n[CUDA ERROR] batch={i+1}/{total_batches}, samples={batch_for_sampling}, no_guidance={no_guidance}")
            print(f"[CUDA ERROR] Exception: {exc}")
            raise

        end.record()
        # Synchronize all GPUs
        for dev in devices:
            torch.cuda.synchronize(dev)
        batch_time = start.elapsed_time(end) / 1000.0  # Convert to seconds
        batch_times.append(batch_time)
        
        # Progress display
        current_samples = (i + 1) * batch_for_sampling
        progress_pct = (i + 1) / total_batches * 100
        elapsed_time = time.time() - start_time
        avg_batch_time = np.mean(batch_times) if batch_times else batch_time
        eta_seconds = avg_batch_time * (total_batches - i - 1)
        eta_minutes = int(eta_seconds // 60)
        eta_secs = int(eta_seconds % 60)
        
        print(f'  Batch {i+1}/{total_batches} ({progress_pct:.1f}%): '
              f'Generated {current_samples}/{actual_samples} samples | '
              f'Batch time: {batch_time:.1f}s | '
              f'Elapsed: {int(elapsed_time//60)}m {int(elapsed_time%60)}s | '
              f'ETA: {eta_minutes}m {eta_secs}s')

        all_generated_data.append(generated_audio12.detach().cpu().numpy())
        all_label.append(cond.detach().cpu().numpy())

    # Save final results to output_directory (which includes local_path)
    # Include guidance suffix in filename to distinguish between guidance modes
    # Naming: gen_data_clf.npy, gen_data_clf_sob.npy, gen_data_clf_sobd.npy, gen_data_diff.npy, gen_data_diff_scratch.npy
    if no_guidance:
        # Check if scratch model was used
        if is_scratch:
            data_suffix = '_diff_scratch'
        else:
            data_suffix = '_diff'
    else:
        data_suffix = f'_{guidance_suffix}'
    new_data = os.path.join(output_directory, f'gen_data{data_suffix}.npy')
    new_label = os.path.join(output_directory, f'gen_label{data_suffix}.npy')

    np.save(new_data, np.array(all_generated_data))
    np.save(new_label, np.array(all_label))
    
    total_time = time.time() - start_time
    print(f'\nGeneration completed!')
    print(f'  Total samples generated: {actual_samples}')
    print(f'  Total time: {int(total_time//60)}m {int(total_time%60)}s')
    print(f'  Average time per batch: {np.mean(batch_times):.2f}s')
    print(f'  Average time per sample: {total_time/actual_samples:.3f}s')
    print(f'  Data saved to: {new_data}')
    print(f'  Labels saved to: {new_label}')
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/config_SSSD_ECG.json',
                        help='JSON file for configuration')
    parser.add_argument('-ckpt_iter', '--ckpt_iter', default=100000,
                        help='Which checkpoint to use; assign a number or "max"')
    parser.add_argument('-n', '--num_samples', type=int, default=17442,
                        help='Number of utterances to be generated')
    parser.add_argument('--use_sobolev', action='store_true',
                        help='Use Bandit_Critic_Guide with Sobolev penalty (clf_sobd mode by default)')
    parser.add_argument('--no_decay', action='store_true',
                        help='If set with --use_sobolev, use clf_sob model (keep at 5.0). Otherwise, use clf_sobd (decay, default)')
    parser.add_argument('--guidance_iter', type=int, default=1000,
                        help='Guidance model iteration to load (default: 1000)')
    parser.add_argument('--no_guidance', action='store_true',
                        help='Use direct diffusion model generation without guidance (loads diff_*.pkl files)')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name suffix to avoid overwriting results. If None, uses default naming. (default: None)')
    args = parser.parse_args()
    
    # Set random seed first, before any random operations
    set_seed(SEED)

    # Parse configs. Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    print(config)

    gen_config = config['gen_config']

    train_config = config["train_config"]  # training parameters

    global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset

    global diffusion_config
    diffusion_config = config["diffusion_config"]  # basic hyperparameters

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(
        **diffusion_config)  # dictionary of all diffusion hyperparameters

    global model_config
    model_config = config['wavenet_config']

    generate(**gen_config,
             ckpt_iter=args.ckpt_iter,
             num_samples=args.num_samples,
             data_path=trainset_config["data_path"],
             use_sobolev=args.use_sobolev,
             no_decay=args.no_decay,
             guidance_iter=args.guidance_iter,
             no_guidance=args.no_guidance,
             experiment_name=args.experiment_name
             )
