

import numpy as np
import torch
from models_new.SSSD_ECG import SSSD_ECG

import sys
import os

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

from util_utility.utils import apply_standardizer

from fastai_model import fastai_model
import pickle
import torch.nn.functional as F
import json
import numpy as np
import torch.nn as nn

import ipdb
from utils.util_generation import find_max_epoch, print_size, sampling_label, calc_diffusion_hyperparams, std_normal
from load_icbeb import load_icbeb_data


def load_pretrained_model():
    """
    Load pretrained model from 100000.pkl and fine-tune it.
    This model is used for training (fit), replacing output layer from 71 to 2 classes.
    
    Returns:
        fastai_model: Model loaded from 100000.pkl, output layer replaced from 71 to 2 classes
    """
    # Get script directory to resolve relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_weight_dir = os.path.join(script_dir, 'Diffusion_RL', 'ecg_ptbxl_benchmarking', 'model_weight')
    
    # Essential parameters
    modelname = 'fastai_xresnet1d50'
    pretrained_file = os.path.join(model_weight_dir, '100000.pkl')
    outputfolder = model_weight_dir + '/'  # For saving/loading model
    num_classes = 2
    n_classes_pretrained = 71  # Original model has 71 classes (PTB-XL)
    sampling_frequency = 100
    input_shape = [1000, 12]

    model = fastai_model(
        name=modelname,                    # Model architecture
        n_classes=num_classes,             # Output classes (2 for binary classification)
        freq=sampling_frequency,           # Sampling frequency (100 Hz)
        outputfolder=outputfolder,        # Directory for saving/loading model
        input_shape=input_shape,           # Input shape [1000, 12]
        pretrained=True,                   # Use pretrained weights
        pretrained_file=pretrained_file,   # Direct path to pretrained model (100000.pkl)
        n_classes_pretrained=n_classes_pretrained,  # Pretrained model has 71 classes
        replace_output_layer=True,         # Replace output layer from 71 to 2 classes
        epochs_finetuning=100,             # Number of epochs for fine-tuning
        # pretrainedfolder=None,          # Not needed when pretrained_file is specified
    )
    return model

def load_density_ratio_model():
    """
    Load density ratio model for guidance network.
    This model is used for inference only (predict_guidance), not for training.
    
    Returns:
        fastai_model: Model loaded from fastai_xresnet1d50.pth with 2 classes output
    """
    # Get script directory to resolve relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_weight_dir = os.path.join(script_dir, 'Diffusion_RL', 'ecg_ptbxl_benchmarking', 'model_weight')
    
    # Essential parameters
    modelname = 'fastai_xresnet1d50'
    pretrained_file = os.path.join(model_weight_dir, 'models', 'fastai_xresnet1d50.pth')
    outputfolder = model_weight_dir + '/'  # For saving/loading model
    num_classes = 2
    n_classes_pretrained = 2  # Model already has 2 classes
    sampling_frequency = 100
    input_shape = [1000, 12]

    model = fastai_model(
        name=modelname,                    # Model architecture
        n_classes=num_classes,             # Output classes (2 for binary classification)
        freq=sampling_frequency,           # Sampling frequency (100 Hz)
        outputfolder=outputfolder,        # Directory for saving/loading model
        input_shape=input_shape,           # Input shape [1000, 12]
        pretrained=True,                   # Use pretrained weights
        pretrained_file=pretrained_file,   # Direct path to pretrained model
        n_classes_pretrained=n_classes_pretrained,  # Pretrained model has 2 classes
        replace_output_layer=False,        # Keep original output layer (already 2 classes)
        # pretrainedfolder=None,           # Not needed when pretrained_file is specified
    )
    return model

def load_diffusion_model(device='cuda'):
    model_config = {
        "in_channels": 8,
        "out_channels":8,
        "num_res_layers": 36,
        "res_channels": 256,
        "skip_channels": 256,
        "diffusion_step_embed_dim_in": 128,
        "diffusion_step_embed_dim_mid": 512,
        "diffusion_step_embed_dim_out": 512,
        "s4_lmax": 1000,
        "s4_d_state":64,
        "s4_dropout":0.0,
        "s4_bidirectional":1,
        "s4_layernorm":1,
        "label_embed_dim":128,
        "label_embed_classes":71
    }
    
    net = SSSD_ECG(**model_config).to(device)
    # Get script directory to resolve relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'Diffusion_RL', 'ecg_ptbxl_benchmarking', 'model_weight', '100000.pkl')
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
    except:
        raise Exception('No valid model found')
    return net


def density_ratio_estimation(model, X, Y):

    X = X.cpu().numpy()  # Move to CPU before converting to numpy

    y_pred = model.predict_guidance(X) 
    
    y_pred = torch.tensor(y_pred)
    y_pred = F.softmax(y_pred)
    # ipdb.set_trace()
    density_ratio = y_pred[:,1]/y_pred[:,0]
    return torch.log(torch.tensor(density_ratio))  # Return on CPU, will be moved to device when needed 

def mlp(dims, activation=nn.ReLU, output_activation=None):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'

    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net

class SiLU(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    return x * torch.sigmoid(x)

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[..., None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class GuidanceQt(nn.Module):
    def __init__(self, action_dim, state_dim):
        super().__init__()
        dims = [action_dim+32+state_dim, 512, 512, 512, 512, 1]
        self.qt = mlp(dims, activation=SiLU)
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=32), nn.Linear(32, 32))

        self.conv1 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=10, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=10, stride=2)
        
    def forward(self, action, t, condition=None):

        action = self.conv1(action)
        action = self.conv2(action)

        # Flatten for input to linear layers
        action = action.view(action.size(0), -1)
    
        ''' '''
        # Need to add this when training guidance network 
        # ipdb.set_trace()
        t = torch.squeeze(t) 

        embed = self.embed(t)
        

        '''
        #for sampling
        
        embed = self.embed(t)
        '''

        ats = torch.cat([action, embed, condition], -1) if condition is not None else torch.cat([action, embed], -1)
        return self.qt(ats)


# ============ U-Bottleneck Version with Sobolev Regularization ============
class UGuidanceQt(nn.Module):
    """
    U-Bottleneck version of GuidanceQt with low-dimensional projection.
    Similar to UGuidance in simulation.py, but adapted for ECG guidance network.
    """
    def __init__(self, action_dim, state_dim, k=32, h=128):
        """
        Args:
            action_dim: Dimension of action after conv layers (3904)
            state_dim: Dimension of state/condition (0 in current implementation)
            k: Bottleneck dimension (low-dimensional latent space)
            h: Hidden dimension for g network (reduced from 512 to 128 for efficiency in 2nd-order gradients)
        """
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.k = k
        
        # U matrix for dimensionality reduction: [action_dim, k]
        self.U = nn.Parameter(torch.randn(action_dim, k))
        
        # Time embedding (same as original)
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=32), nn.Linear(32, 32))
        
        # Conv layers (same as original)
        self.conv1 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=10, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=10, stride=2)
        
        # Network g operating in low-dimensional space: [k + 32 + state_dim] -> 1
        # Using smaller h (128 instead of 512) to reduce 2nd-order gradient computation cost
        self.g = nn.Sequential(
            nn.Linear(k + 32 + state_dim, h),
            SiLU(),
            nn.Linear(h, h),
            SiLU(),
            nn.Linear(h, 1)  # Reduced from 3 hidden layers to 2 for efficiency
        )
    
    def forward_with_latent(self, action, t, condition=None, detach_U=False):
        """
        Forward pass that also returns latent representation z.
        
        Returns:
            f: Output logit [B, 1]
            z: Low-dimensional latent [B, k + 32 + state_dim]
            action_flat: Flattened action features [B, action_dim]
            U: Projection matrix (possibly detached)
        """
        # Conv layers
        action = self.conv1(action)
        action = self.conv2(action)
        action_flat = action.view(action.size(0), -1)  # [B, action_dim]
        
        # Time embedding
        t = torch.squeeze(t)
        embed = self.embed(t)  # [B, 32]
        
        # Dimensionality reduction: z_action = action_flat @ U
        U = self.U.detach() if detach_U else self.U
        z_action = action_flat @ U  # [B, k]
        
        # Concatenate with time embedding and condition
        if condition is not None:
            z = torch.cat([z_action, embed, condition], dim=-1)  # [B, k + 32 + state_dim]
        else:
            z = torch.cat([z_action, embed], dim=-1)  # [B, k + 32]
        
        # Network g in low-dimensional space
        f = self.g(z)  # [B, 1]
        
        return f, z, action_flat, U
    
    def forward(self, action, t, condition=None, detach_U=False):
        """Standard forward pass."""
        f, _, _, _ = self.forward_with_latent(action, t, condition, detach_U)
        return f


def sobolev_penalty_guidance(model: UGuidanceQt, action, t, condition=None, detach_U=False):
    """
    Sobolev penalty for guidance network with U-bottleneck: penalizes ||∇_z_action Q_t||^2
    This encourages smoothness of the guidance function in low-dimensional space.
    
    Args:
        model: UGuidanceQt model
        action: Input action [B, 8, 1000]
        t: Diffusion time steps [B, 1, 1]
        condition: Optional condition [B, state_dim]
        detach_U: Whether to detach U from computation graph
    
    Returns:
        penalty: Scalar Sobolev penalty
    """
    # Clone and set requires_grad for action
    action = action.clone().detach().requires_grad_(True)
    
    # Forward pass with latent
    f, z, action_flat, U = model.forward_with_latent(action, t, condition, detach_U=detach_U)
    
    # Compute gradient w.r.t. low-dimensional latent z
    # grad_z: [B, k + 32 + state_dim]
    grad_z = torch.autograd.grad(f.sum(), z, create_graph=True)[0]
    
    # Extract gradient w.r.t. z_action (first k dimensions)
    grad_z_action = grad_z[:, :model.k]  # [B, k]
    
    # Project back to action space: grad_action = grad_z_action @ U.T
    # grad_action = grad_z_action @ U.T  # [B, action_dim]
    
    # Penalty: mean of squared gradient norm
    penalty = (grad_z_action ** 2).sum(dim=1).mean()
    
    return penalty


def sobolev_penalty_guidance_standard(model: GuidanceQt, action, t, condition=None, joint_gradient=False):
    """
    Sobolev penalty for standard guidance network (without U-bottleneck): penalizes ||∇_action Q_t||^2
    This encourages smoothness of the guidance function directly in action space.
    
    Args:
        model: GuidanceQt model
        action: Input action [B, 8, 1000]
        t: Diffusion time steps [B, 1, 1] (integer type)
        condition: Optional condition [B, state_dim]
        joint_gradient: If True, compute gradient w.r.t. both action and t embedding (joint gradient).
                       If False, only compute gradient w.r.t. action (default: False)
    
    Returns:
        penalty: Scalar Sobolev penalty
    """
    # Handle DataParallel wrapper: get underlying model
    actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    # Clone and set requires_grad for action
    action = action.clone().detach().requires_grad_(True)
    
    # Process action through conv layers (same as in model.forward)
    action_conv = actual_model.conv1(action)
    action_conv = actual_model.conv2(action_conv)
    action_flat = action_conv.view(action_conv.size(0), -1)  # [B, action_dim]
    
    # Process t: convert to float and compute embedding
    t_squeezed = torch.squeeze(t.clone().detach())  # [B]
    
    if joint_gradient:
        # Convert t to float and set requires_grad for embedding computation
        t_float = t_squeezed.float().requires_grad_(True)  # [B]
        # Compute embedding with gradient tracking
        embed = actual_model.embed(t_float)  # [B, 32]
    else:
        # Keep t as float but no gradient needed
        t_float = t_squeezed.float()
        embed = actual_model.embed(t_float)  # [B, 32]
    
    # Concatenate action, embed, and condition (same as in model.forward)
    if condition is not None:
        ats = torch.cat([action_flat, embed, condition], -1)
    else:
        ats = torch.cat([action_flat, embed], -1)
    
    # Forward pass through qt network
    f = actual_model.qt(ats)  # [B, 1]
    
    if joint_gradient:
        # Compute joint gradient w.r.t. both action and embed
        grad_action, grad_embed = torch.autograd.grad(
            f.sum(), 
            [action, embed], 
            create_graph=True
        )  # grad_action: [B, 8, 1000], grad_embed: [B, 32]
        
        # Penalty: mean of squared gradient norm for both action and embed
        penalty_action = (grad_action ** 2).sum(dim=(1, 2)).mean()  # [B] -> scalar
        penalty_embed = (grad_embed ** 2).sum(dim=1).mean()  # [B] -> scalar
        penalty = penalty_action + penalty_embed
    else:
        # Compute gradient w.r.t. action only
        grad_action = torch.autograd.grad(f.sum(), action, create_graph=True)[0]  # [B, 8, 1000]
        
        # Penalty: mean of squared gradient norm
        penalty = (grad_action ** 2).sum(dim=(1, 2)).mean()
    
    return penalty



# class Bandit_Critic_Guide_UBottleneck(nn.Module):
#     """
#     U-Bottleneck version of Bandit_Critic_Guide with Sobolev regularization.
#     Uses UGuidanceQt instead of GuidanceQt for low-dimensional projection.
#     """
#     def __init__(self, adim, sdim, args, k=32, h=512) -> None:
#         super().__init__()
#         self.qt = UGuidanceQt(adim, sdim, k=k, h=h).to(args.device)
        
#         # Separate optimizers for main network and U matrix (similar to simulation.py)
#         main_params = [p for n, p in self.qt.named_parameters() if n != 'U']
#         self.qt_optimizer_main = torch.optim.Adam(main_params, lr=3e-4)
#         self.qt_optimizer_U = torch.optim.Adam([self.qt.U], lr=2e-3)
        
#         self.guidance_net = load_density_ratio_model()
        
#         self.args = args
#         self.guidance_scale = 1.0
#         self.alpha = 1.0
        
#         with open('config/config_SSSD_ECG.json') as f:
#             data = f.read()
        
#         config = json.loads(data)
#         diffusion_config = config["diffusion_config"]
#         diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config) 
#         _dh = diffusion_hyperparams
#         self.T, self.Alpha_bar = _dh["T"], _dh["Alpha_bar"].to(args.device)
        
#         self.model = load_diffusion_model(args.device)
#         X_train_ICBEB, X_val_ICBEB, X_test_ICBEB, y_train_ICBEB, y_val_ICBEB, y_test_ICBEB = load_icbeb_data()
#         X_train_ICBEB = X_test_ICBEB
#         y_train_ICBEB = y_test_ICBEB
#         indices = [i for i, d in enumerate(X_train_ICBEB) if d.shape[0] >= 1000]
#         X_train_ICBEB = [d[:1000,:] for d in X_train_ICBEB if d.shape[0] >= 1000]
#         y_train_ICBEB = y_train_ICBEB[indices]
        
#         new_labels = np.zeros((y_train_ICBEB.shape[0], 71))
#         label_map = [46, 4, 0, 11, 12, 49, 54, 63, 64]
#         for i in range(y_train_ICBEB.shape[0]):
#             for j in range(y_train_ICBEB.shape[1]):
#                 new_labels[i, label_map[j]] = y_train_ICBEB[i, j]
        
#         self.target_labels = torch.tensor(new_labels).float().to(args.device)
#         # Stack list elements before converting to tensor (don't use array before cropping)
#         self.target_sample = torch.tensor(np.stack(X_train_ICBEB)).to(args.device)
#         self.target_sample = torch.permute(self.target_sample, (0,2,1))
#         index_8 = torch.tensor([0,2,3,4,5,6,7,11]).to(args.device)
#         index_4 = torch.tensor([1,8,9,10]).to(args.device)
#         self.target_sample = torch.index_select(self.target_sample, 1, index_8).float()
    
#     def forward(self, a, condition=None):
#         return self.qt(a, condition)
    
#     def calculate_guidance(self, a, t, condition=None):
#         with torch.enable_grad():
#             a.requires_grad_(True)
#             Q_t = self.qt(a, t, condition)
#             guidance = self.guidance_scale * torch.autograd.grad(torch.sum(Q_t), a)[0]
#         return guidance.detach()
    
#     def calculated_consistence_regularization(self):
#         random_indices = torch.randint(0, self.target_sample.size(0), (8,))
#         sample = self.target_sample[random_indices]
#         label = self.target_labels[random_indices]
        
#         B, C, L = sample.shape
#         diffusion_steps = torch.randint(self.T, size=(B,1,1)).to(self.args.device)
#         z = std_normal(sample.shape).to(self.args.device)
#         transformed_X = torch.sqrt(self.Alpha_bar[diffusion_steps]) * sample + torch.sqrt(1-self.Alpha_bar[diffusion_steps]) * z
#         score = self.model((transformed_X, label, diffusion_steps.view(B,1),))  
        
#         transformed_X.requires_grad = True
#         # For U-bottleneck version, detach U when computing consistency loss
#         Q_t = self.qt(transformed_X, diffusion_steps, detach_U=True)
#         guidance = self.guidance_scale * torch.autograd.grad(torch.sum(Q_t), transformed_X, retain_graph=True)[0]
#         loss = torch.mean(torch.sum((score + guidance - z)**2, dim=(1,)))
#         return loss
    
#     def update_qt_with_ubottleneck(self, audio, label, lambda_grad=1e-4, u_update_every=100, qr_every=300, beta_orth=1e-4, step=0):
#         """
#         Update qt network with U-bottleneck and Sobolev regularization.
#         Similar to train_U_sobolev in simulation.py.
        
#         Args:
#             audio: Input audio [B, 12, 1000]
#             label: Labels (not used in current implementation)
#             lambda_grad: Weight for Sobolev penalty
#             u_update_every: Update U every N steps
#             qr_every: Orthonormalize U every N steps
#             beta_orth: Weight for orthogonal regularization
#             step: Current training step (for annealing)
        
#         Returns:
#             loss: Training loss
#         """
#         index_8 = torch.tensor([0,2,3,4,5,6,7,11]).to(self.args.device)
#         index_4 = torch.tensor([1,8,9,10]).to(self.args.device) 
#         reward = density_ratio_estimation(self.guidance_net, audio, label).to(self.args.device)
#         audio = torch.index_select(audio, 1, index_8).float().to(self.args.device)
        
#         update_U = (step % u_update_every == 0)
        
#         if self.args.method == "mse":
#             B, C, L = audio.shape
#             diffusion_steps = torch.randint(self.T, size=(B,1,1)).to(self.args.device)
#             z = std_normal(audio.shape).to(self.args.device)
#             transformed_X = torch.sqrt(self.Alpha_bar[diffusion_steps]) * audio + torch.sqrt(1-self.Alpha_bar[diffusion_steps]) * z
            
#             # Forward pass (detach_U when not updating U)
#             logits = self.qt.forward(transformed_X, diffusion_steps, None, detach_U=not update_U)
#             loss_ce = torch.mean((logits - reward * self.alpha)**2)
            
#             # Sobolev penalty
#             sob = sobolev_penalty_guidance(self.qt, transformed_X, diffusion_steps, None, detach_U=not update_U)
            
#             # Orthogonal regularization (only when updating U)
#             if update_U:
#                 I = torch.eye(self.qt.U.shape[1], device=self.args.device)
#                 # orth_pen = ((self.qt.U.T @ self.qt.U - I)**2).sum()
#             else:
#                 # orth_pen = torch.tensor(0.0, device=self.args.device)
#                 pass
#             # Sobolev annealing (first 100 steps from 0 to lambda_grad)
#             lam0, lam1, T_anneal = 0.0, lambda_grad, 100
#             lam_t = lam0 + (lam1 - lam0) * min(step / T_anneal, 1.0)
            
#             # Consistency regularization
#             loss_consistence = self.calculated_consistence_regularization()
            
#             # Total loss
#             # loss = loss_ce + lam_t * sob + beta_orth * orth_pen + loss_consistence
#             loss = loss_ce + lam_t * sob + loss_consistence

#             if step % 100 == 0:
#                 print(f"Step {step}, Loss: {loss.item():.6f}, Loss_ce: {loss_ce.item():.6f}, Loss_sob: {sob.item():.6f}, Loss_consistence: {loss_consistence.item():.6f}")
#             # print(f"Loss: {loss.item():.6f}, Loss_ce: {loss_ce.item():.6f}, Loss_sob: {sob.item():.6f}, Loss_orth: {orth_pen.item():.6f}, Loss_consistence: {loss_consistence.item():.6f}")
            
#             # Optimize
#             self.qt_optimizer_main.zero_grad(set_to_none=True)
#             if update_U:
#                 self.qt_optimizer_U.zero_grad(set_to_none=True)
#             loss.backward()
#             self.qt_optimizer_main.step()
#             if update_U:
#                 self.qt_optimizer_U.step()
#                 # Orthonormalize U periodically
#                 # Column normalization (mild) - use no_grad to avoid gradient computation issues
#                 with torch.no_grad():
#                     self.qt.U /= (self.qt.U.norm(dim=0, keepdim=True) + 1e-12)
#                 # Do QR projection every 100 steps (hard projection)
#                 if step % qr_every == 0:
#                     with torch.no_grad():
#                         q, _ = torch.linalg.qr(self.qt.U.data)
#                         self.qt.U.copy_(q[:, :self.qt.k])
            
#             return loss.detach().cpu().numpy()
#         else:
#             raise NotImplementedError(f"Method {self.args.method} not implemented for U-bottleneck version")


class Bandit_Critic_Guide(nn.Module):
    def __init__(self, adim, sdim, args, weight_decay=0.0) -> None:
        super().__init__()
        self.qt = GuidanceQt(adim, sdim).to(args.device)
        self.qt_optimizer = torch.optim.Adam(self.qt.parameters(), lr=3e-4, weight_decay=weight_decay)
        self.guidance_net = load_density_ratio_model()  # Load density ratio estimator
        
        self.args = args
        self.guidance_scale = 1.0
        self.alpha = 1.0


        with open('config/config_SSSD_ECG.json') as f:
            data = f.read()

        config = json.loads(data)

        diffusion_config = config["diffusion_config"]
        diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config) 
        _dh = diffusion_hyperparams
        self.T, self.Alpha_bar = _dh["T"], _dh["Alpha_bar"].to(args.device)

        self.model = load_diffusion_model(args.device)
        X_train_ICBEB, X_val_ICBEB, X_test_ICBEB, y_train_ICBEB, y_val_ICBEB, y_test_ICBEB = load_icbeb_data()
        X_train_ICBEB = X_test_ICBEB
        y_train_ICBEB = y_test_ICBEB
        indices = [i for i, d in enumerate(X_train_ICBEB) if d.shape[0] >= 1000]
        X_train_ICBEB = [d[:1000,:] for d in X_train_ICBEB if d.shape[0] >= 1000]
        y_train_ICBEB = y_train_ICBEB[indices]

        new_labels = np.zeros((y_train_ICBEB.shape[0], 71))

        label_map = [46, 4, 0, 11, 12, 49, 54, 63, 64]
        # Iterate through each example and update new_labels
        for i in range(y_train_ICBEB.shape[0]):
            for j in range(y_train_ICBEB.shape[1]):
                new_labels[i, label_map[j]] = y_train_ICBEB[i, j]

        self.target_labels = torch.tensor(new_labels).float().to(args.device)
        # Stack list elements before converting to tensor (don't use array before cropping)
        self.target_sample = torch.tensor(np.stack(X_train_ICBEB)).to(args.device)
        self.target_sample = torch.permute(self.target_sample, (0,2,1))
        index_8 = torch.tensor([0,2,3,4,5,6,7,11]).to(args.device)
        index_4 = torch.tensor([1,8,9,10]).to(args.device)
        self.target_sample = torch.index_select(self.target_sample, 1, index_8).float()

    def forward(self, a, condition=None):
        return self.qt(a, condition)

    def calculate_guidance(self, a, t, condition=None):
        with torch.enable_grad():
            a.requires_grad_(True)
            Q_t = self.qt(a, t, condition)
            # Q_t = torch.log(Q_t) # This is the unnormalized version
            guidance =  self.guidance_scale * torch.autograd.grad(torch.sum(Q_t), a)[0]
        return guidance.detach()

    def calculated_consistence_regularization(self):
        random_indices = torch.randint(0, self.target_sample.size(0), (8,))
        sample = self.target_sample[random_indices]
        label = self.target_labels[random_indices]

        B, C, L = sample.shape  # B is batchsize, C=1, L is audio length
        diffusion_steps = torch.randint(self.T, size=(B,1,1)).to(self.args.device)  # randomly sample diffusion steps from 1~T
        z = std_normal(sample.shape).to(self.args.device)
        transformed_X = torch.sqrt(self.Alpha_bar[diffusion_steps]) * sample + torch.sqrt(1-self.Alpha_bar[diffusion_steps]) * z
        score = self.model((transformed_X, label, diffusion_steps.view(B,1),))  

        # Ensure transformed_X has requires_grad for gradient computation
        # Clone to avoid any potential in-place modification issues
        transformed_X = transformed_X.clone().detach().requires_grad_(True)
        Q_t = self.qt(transformed_X, diffusion_steps)
        # guidance = self.guidance_scale * Q_t
        # Use create_graph=True to ensure gradients flow back to self.qt parameters
        guidance =  self.guidance_scale * torch.autograd.grad(torch.sum(Q_t), transformed_X, create_graph=True, retain_graph=True)[0]
        # ipdb.set_trace()
        loss = torch.mean(torch.sum((score  +  guidance - z)**2, dim=(1,)))
        return loss

    def update_qt(self, audio, label, lambda_grad=0.0, step=0, use_sobolev=False, decay_to_min=False, verbose=False, joint_gradient=False, improved_schedule=False, total_iterations=None, consistency_weight=0.0, data_noise_scale=0.0):
        """
        Update qt network with optional Sobolev regularization.
        
        Args:
            audio: Input audio [B, 12, 1000]
            label: Labels
            lambda_grad: Maximum weight for Sobolev penalty (default: 0.0, disabled)
            step: Current training step (for scheduling, default: 0)
            use_sobolev: Whether to use Sobolev penalty (default: False)
            decay_to_min: If True, decay from lambda_grad (5.0) to 1e-2 (500-1500 steps). If False, keep at lambda_grad (default: False)
            verbose: Whether to print lambda scheduling info (default: False)
            joint_gradient: If True, compute Sobolev penalty w.r.t. both x and t (joint gradient). 
                           If False, only compute w.r.t. x (default: False)
            improved_schedule: If True, use T_decay1=1500 instead of 800 (default: False)
            total_iterations: Total training iterations. If provided, T_decay1 will be capped at this value (default: None)
            consistency_weight: Weight for consistency regularization loss (default: 0.0, disabled). 
                               If > 0, ensures guidance is consistent with score function.
            data_noise_scale: Scale of Gaussian noise to add to training data (default: 0.0, disabled).
                            If > 0, adds noise to audio data. Higher values (0.02-0.05) help Sobolev regularization
                            by making gradients unstable if model overfits.
        
        Returns:
            loss: Training loss
        """
        index_8 = torch.tensor([0,2,3,4,5,6,7,11]).to(self.args.device)
        index_4 = torch.tensor([1,8,9,10]).to(self.args.device) 
        reward = density_ratio_estimation(self.guidance_net,audio, label).to(self.args.device)
        audio = torch.index_select(audio, 1, index_8).float().to(self.args.device)
        
        
        if self.args.method == "mse":

            
            B, C, L = audio.shape  # B is batchsize, C=1, L is audio length
            # diffusion_steps = torch.randint(T, size=(B,1,1)).to(self.args.device)  # randomly sample diffusion steps from 1~T
            diffusion_steps = torch.randint(self.T, size=(B,1,1)).to(self.args.device)  # randomly sample diffusion steps from 1~T
            # ！！！ it is very important to restrict the T near 0
            # ipdb.set_trace()
            z = std_normal(audio.shape).to(self.args.device)
            transformed_X = torch.sqrt(self.Alpha_bar[diffusion_steps]) * audio + torch.sqrt(1-self.Alpha_bar[diffusion_steps]) * z
            
            # Add noise to transformed_X if enabled
            # This makes gradients unstable if model overfits, helping Sobolev regularization
            # If model overfits, it becomes sensitive to small noise perturbations, causing gradient instability
            # Sobolev penalty will then penalize these unstable gradients, preventing overfitting
            if data_noise_scale > 0:
                # Generate random noise for each batch (different noise each iteration)
                # Noise is added to transformed_X (diffusion-processed data) where model actually operates
                noise = torch.randn_like(transformed_X) * data_noise_scale
                transformed_X = transformed_X + noise
            loss_ce = torch.mean((self.qt(transformed_X, diffusion_steps, None)- reward * self.alpha)**2)
            # loss = torch.mean((self.qt(transformed_X, diffusion_steps, label)- reward * self.alpha)**2)

            # Sobolev penalty (optional)
            if use_sobolev and lambda_grad > 0:
                # sobolev_penalty_guidance_standard will clone and detach internally, so it won't affect transformed_X
                sob = sobolev_penalty_guidance_standard(self.qt, transformed_X, diffusion_steps, None, joint_gradient=joint_gradient)
                # Sobolev scheduling strategy:
                # 1. 0-100 steps: linear increase from 0 to lambda_grad (max value, e.g., 5.0)
                # 2. 100-500 steps: keep at lambda_grad (5.0)
                # 3. 500-T_decay1 steps: if decay_to_min=True, linear decay from lambda_grad (5.0) to 1e-2; else keep at lambda_grad
                # 4. T_decay1+ steps: keep at final value (1e-2 if decay_to_min=True, or lambda_grad if False)
                T_warmup = 100      # Warmup period: 0 -> lambda_grad
                T_plateau = 500    # Plateau period: keep at lambda_grad
                T_decay1 = 1500 if improved_schedule else 800  # Decay period: lambda_grad -> 1e-2 (only if decay_to_min=True)
                # Cap T_decay1 at total_iterations if provided, to ensure decay completes before training ends
                if total_iterations is not None:
                    T_decay1 = min(T_decay1, total_iterations)
                lambda_min = 1e-2   # Minimum value (used if decay_to_min=True)
                
                if step <= T_warmup:
                    # Phase 1: Warmup (0 -> lambda_grad)
                    lam_t = lambda_grad * (step / T_warmup)
                elif step <= T_plateau:
                    # Phase 2: Plateau (keep at lambda_grad)
                    lam_t = lambda_grad
                elif step <= T_decay1:
                    # Phase 3: Decay period (500-1500 steps)
                    if decay_to_min:
                        # Decay from lambda_grad to 1e-2
                        progress = (step - T_plateau) / (T_decay1 - T_plateau)
                        lam_t = lambda_grad + (lambda_min - lambda_grad) * progress
                    else:
                        # Keep at lambda_grad (no decay)
                        lam_t = lambda_grad
                else:
                    # Phase 4: After 1500 steps
                    if decay_to_min:
                        # Keep at minimum value (1e-2)
                        lam_t = lambda_min
                    else:
                        # Keep at maximum value (lambda_grad)
                        lam_t = lambda_grad
                
                # Debug output for lambda scheduling (every 100 steps)
                if verbose and step % 100 == 0:
                    if step <= T_warmup:
                        phase_name = "Warmup"
                    elif step <= T_plateau:
                        phase_name = "Plateau"
                    elif step <= T_decay1:
                        phase_name = "Decay" if decay_to_min else "KeepMax"
                    else:
                        phase_name = "KeepMin" if decay_to_min else "KeepMax"
                    print(f"Step {step}: Sobolev lambda = {lam_t:.6f} (Phase: {phase_name})")
            else:
                sob = torch.tensor(0.0, device=self.args.device)
                lam_t = 0.0

            # Consistency regularization (optional)
            if consistency_weight > 0:
                loss_consistence = self.calculated_consistence_regularization()
                loss = loss_ce + lam_t * sob + consistency_weight * loss_consistence
            else:
                loss = loss_ce + lam_t * sob
            
            # Debug: Check if gradients exist before backward
            if step % 100 == 0:
                # Check if loss components require grad
                loss_ce_requires_grad = loss_ce.requires_grad if isinstance(loss_ce, torch.Tensor) else False
                # loss_cons_requires_grad = loss_consistence_regularization.requires_grad if isinstance(loss_consistence_regularization, torch.Tensor) else False
                # print(f"Step {step}, Loss: {loss.item():.6f}, Loss_ce: {loss_ce.item():.6f}, Loss_sob: {sob.item():.6f}, Loss_consistence: {loss_consistence_regularization.item():.6f}")
                # print(f"  Grad check - loss_ce.requires_grad: {loss_ce_requires_grad}, loss_cons.requires_grad: {loss_cons_requires_grad}")

        # elif self.args.method == "emse":
        #     pass

        # elif self.args.method == "CEP":
        #     pass

        else:
            raise NotImplementedError

        self.qt_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Debug: Check gradients after backward
        if step % 100 == 0:
            grad_norm = 0.0
            param_count = 0
            for name, param in self.qt.named_parameters():
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item() ** 2
                    param_count += 1
            grad_norm = grad_norm ** 0.5
            print(f"  After backward - param_count with grad: {param_count}, grad_norm: {grad_norm:.6f}")
        
        self.qt_optimizer.step()

        return loss.detach().cpu().numpy()