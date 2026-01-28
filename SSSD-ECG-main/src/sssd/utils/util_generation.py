import os
import numpy as np
import torch
import random

import argparse
import ipdb

def flatten(v):
    """
    Flatten a list of lists/tuples
    """

    return [x for y in v for x in y]


def find_max_epoch(path):
    """
    Find maximum epoch/iteration in path, formatted ${n_iter}.pkl
    E.g. 100000.pkl

    Parameters:
    path (str): checkpoint path
    
    Returns:
    maximum iteration, -1 if there is no (valid) checkpoint
    """

    files = os.listdir(path)
    epoch = -1
    for f in files:
        if len(f) <= 4:
            continue
        if f[-4:] == '.pkl':
            try:
                epoch = max(epoch, int(f[:-4]))
            except:
                continue
    return epoch


def print_size(net):
    """
    Print the number of parameters of a network
    """

    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])
        print("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6), flush=True)


# Utilities for diffusion models

def std_normal(size, device=None):
    """
    Generate the standard Gaussian variable of a certain size
    
    Parameters:
    size: size of the tensor
    device: device to place the tensor on. If None, uses cuda:0 if available, else cpu
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.normal(0, 1, size=size).to(device)


def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    """
    Embed a diffusion step $t$ into a higher dimensional space
    E.g. the embedding vector in the 128-dimensional space is
    [sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)), cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]

    Parameters:
    diffusion_steps (torch.long tensor, shape=(batchsize, 1)):     
                                diffusion steps for batch data
    diffusion_step_embed_dim_in (int, default=128):  
                                dimensionality of the embedding space for discrete diffusion steps
    
    Returns:
    the embedding vectors (torch.tensor, shape=(batchsize, diffusion_step_embed_dim_in)):
    """

    assert diffusion_step_embed_dim_in % 2 == 0

    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)
    # Use the same device as diffusion_steps
    device = diffusion_steps.device if isinstance(diffusion_steps, torch.Tensor) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _embed = torch.exp(torch.arange(half_dim, device=device) * -_embed)
    _embed = diffusion_steps * _embed
    diffusion_step_embed = torch.cat((torch.sin(_embed),
                                     torch.cos(_embed)), 1)

    return diffusion_step_embed


def calc_diffusion_hyperparams(T, beta_0, beta_T):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value, 
                                where any beta_t in the middle is linearly interpolated
    
    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
        These cpu tensors are changed to cuda tensors on each individual gpu
    """

    Beta = torch.linspace(beta_0, beta_T, T)  # Linear schedule
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t - 1]  # \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
        Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (
                1 - Alpha_bar[t])  # \tilde{\beta}_t = \beta_t * (1-\bar{\alpha}_{t-1})
        # / (1-\bar{\alpha}_t)
    Sigma = torch.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t

    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = T, Beta, Alpha, Alpha_bar, Sigma
    diffusion_hyperparams = _dh
    return diffusion_hyperparams

  
def sampling_label(net, size, diffusion_hyperparams, cond=None):
    """
    Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

    Parameters:
    net (torch network):            the wavenet model
    size (tuple):                   size of tensor to be generated, 
                                    usually is (number of audios to generate, channels=1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors 
    cond: conditioning as integer tensor
    guidance_weight: weight for classifier-free guidance (if trained with conditioning_dropout>0)
    
    Returns:
    the generated audio(s) in torch.tensor, shape=size
    """

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    assert len(size) == 3
    
    # Get device from the network to ensure all tensors are on the same device
    device = next(net.parameters()).device
    
    # Move Alpha, Alpha_bar, Sigma to the same device as the network
    Alpha = Alpha.to(device)
    Alpha_bar = Alpha_bar.to(device)
    Sigma = Sigma.to(device)
    
    # Move cond to device if provided
    if cond is not None:
        cond = cond.to(device)
    
    # Try to use tqdm for progress bar, fallback to simple print if not available
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
    
    # Create progress bar for diffusion steps
    if use_tqdm:
        pbar = tqdm(range(T-1, -1, -1), desc='Diffusion sampling', unit='step', leave=False, ncols=100)
        step_iter = pbar
    else:
        print('begin sampling, total number of reverse steps = %s' % T)
        step_iter = range(T-1, -1, -1)

    x = std_normal(size, device=device)
    with torch.no_grad():
        for t in step_iter:
            diffusion_steps = (t * torch.ones((size[0], 1), device=device))  # use the corresponding reverse step
            epsilon_theta = net((x, cond, diffusion_steps,))  # predict \epsilon according to \epsilon_\theta
                
            x = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])  # update x_{t-1} to \mu_\theta(x_t)
            if t > 0:
                x = x + Sigma[t] * std_normal(size, device=device)  # add the variance term to x_{t-1}
    return x

def sampling_label_guided(net, guidance_net, size, diffusion_hyperparams, cond=None):
    """
    Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

    Parameters:
    net (torch network):            the wavenet model
    size (tuple):                   size of tensor to be generated, 
                                    usually is (number of audios to generate, channels=1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors 
    cond: conditioning as integer tensor
    guidance_weight: weight for classifier-free guidance (if trained with conditioning_dropout>0)
    
    Returns:
    the generated audio(s) in torch.tensor, shape=size
    """

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    assert len(size) == 3
    
    # Get device from the network
    device = next(net.parameters()).device
    x = std_normal(size, device=device)
    
    # Try to use tqdm for progress bar, fallback to simple print if not available
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
    
    # Create progress bar for diffusion steps
    if use_tqdm:
        pbar = tqdm(range(T-1, -1, -1), desc='Diffusion sampling', unit='step', leave=False, ncols=100)
        step_iter = pbar
    else:
        print(f'begin sampling, total number of reverse steps = {T}')
        step_iter = range(T-1, -1, -1)
    
    with torch.no_grad():
        for t in step_iter:
            diffusion_steps = (t * torch.ones((size[0], 1), device=device))  # use the corresponding reverse step
            epsilon_theta = net((x, cond, diffusion_steps,))  # predict \epsilon according to \epsilon_\theta
                

            # t_pad = torch.ones(x.shape[0], x.shape[1], device=device)
            # additional guidance 
            # ipdb.set_trace()
            epsilon_theta += guidance_net.calculate_guidance(x, diffusion_steps)

            x = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])  # update x_{t-1} to \mu_\theta(x_t)
            if t > 0:
                x = x + Sigma[t] * std_normal(size, device=device)  # add the variance term to x_{t-1}
            
            # Update progress bar (tqdm handles this automatically, but we can add custom info)
            if use_tqdm:
                pbar.set_postfix({'current_step': t})
            elif (T - t) % max(1, T // 20) == 0 or t == 0:  # Print every 5% or at the end
                progress_pct = (T - t) / T * 100
                remaining_steps = t
                print(f'  Diffusion step: {t}/{T-1} ({progress_pct:.1f}%, {remaining_steps} remaining)', end='\r', flush=True)
    
    if not use_tqdm:
        print()  # New line after progress
    elif use_tqdm:
        pbar.close()
    
    return x


def training_loss_label(net, loss_fn, X, diffusion_hyperparams):
    
    """
    Compute the training loss of epsilon and epsilon_theta

    Parameters:
    net (torch network):            the wavenet model
    loss_fn (torch loss function):  the loss function, default is nn.MSELoss()
    X (torch.tensor):               training data, shape=(batchsize, 1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors       
    
    Returns:
    training loss
    """

    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]
    
    audio = X[0]
    label = X[1]
    B, C, L = audio.shape  # B is batchsize, C=1, L is audio length
    
    # Get device from audio tensor to ensure all tensors are on the same device
    device = audio.device
    diffusion_steps = torch.randint(T, size=(B,1,1), device=device)  # randomly sample diffusion steps from 1~T
    z = std_normal(audio.shape, device=device)
    transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * audio + torch.sqrt(1-Alpha_bar[diffusion_steps]) * z
    epsilon_theta = net((transformed_X, label, diffusion_steps.view(B,1),))  
    
    return loss_fn(epsilon_theta, z)

def bandit_get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="8gaussians") # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)             # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--expid", default="default", type=str)    # 
    parser.add_argument("--device", default="cuda", type=str)      #
    parser.add_argument("--save_model", default=1, type=int)       #
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=3.0)        # beta parameter in the paper, use alpha because of legacy
    parser.add_argument('--diffusion_steps', type=int, default=15)
    parser.add_argument('--method', type=str, default="mse")
    print("**************************")
    args = parser.parse_known_args()[0]
    print(args)
    return args
