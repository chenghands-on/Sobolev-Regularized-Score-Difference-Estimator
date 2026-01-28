# %%
import torch
import torch.nn.functional as F
from numpy import *

from core.gradest import version, infer_KL, infer_ULSIF
from core.util import comp_dist, comp_median

import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
def infer_Chi2_cpp(Xp, Xq, X, sigma, lmbd):
    grad, sigma = infer_ULSIF(Xp.detach().cpu().numpy(), Xq.detach().cpu().numpy(), X.detach().cpu().numpy(), sigma_chosen=sigma, lambda_chosen=lmbd, maxiter = 250)
    return torch.from_numpy(grad).to(device), sigma

def infer_KL_cpp(Xp, Xq, X, sigma, lmbd):
    grad, sigma = infer_KL(Xp.detach().cpu().numpy(), Xq.detach().cpu().numpy(), X.detach().cpu().numpy(), sigma_chosen=sigma, lambda_chosen=lmbd, maxiter = 250)
    return torch.from_numpy(grad).to(device), sigma
# %%
def subsample(xp, xq, n = 2000):
    xps = xp[random.choice(xp.shape[0], n, replace=False),:]
    xqs = xq[random.choice(xq.shape[0], n, replace=False),:]
    return xps, xqs
# %%
seed = 1
torch.manual_seed(seed)

VGD_batchsize = 4000
forwardKLflow = True
DEBUG = False

print("\n------------------------\n")
print("Random Seed: ", seed)
print("VGD Batch Size: ", VGD_batchsize)
print("Flow type:", "Forward KL" if forwardKLflow else "Reverse KL\n")

print("Debugging Type:", "ON" if DEBUG else "OFF" + "\n")
print("Please find generated figures in ./figs/mnist/\n")
print("During the generation process, you can cancel at any time by pressing Ctrl+C\n")
print("\n------------------------\n")
print("To Continue, press enter. ")
input()

# load mnist dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms

# rescaling the MNIST images to 32x32
image_size = 32

transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor()
                             ])

dataroot = "./data"
trainset = dset.MNIST(root=dataroot, train=True, download=True, transform=transform)


# %%
from torch.distributions.multivariate_normal import MultivariateNormal as MVN

trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=True)
XData, _ = next(iter(trainloader)); XData = XData.view(XData.shape[0], -1).to(device)

# save XData into MATLAB format
import scipy.io as sio
sio.savemat('XData.mat', {'XData': XData.cpu().numpy()})
# %%
import matplotlib.pyplot as plt

img = F.avg_pool2d(XData.reshape(-1,1,32,32), kernel_size=1, stride=1, padding=0)
img = F.interpolate(img, size=32, mode='bilinear').reshape(-1,32,32)
img = img.reshape(-1,32*32) + torch.randn_like(img.reshape(-1,32*32))*0.001

plt.figure(figsize=(6,6))
for i in range(49):
    plt.subplot(7,7,i+1)
    plt.imshow(torch.clamp(img[i, :].reshape(32,32),0,1).cpu(), cmap='gray')
    plt.axis('off')
# %%

muq = torch.mean(img, dim=0)
covarq = torch.cov(img.T)

from torch.distributions.multivariate_normal import MultivariateNormal as MVN

Xq_N = MVN(muq, covarq).sample((20000,)).to(device)
Xq_N = torch.clamp(Xq_N, 0, 1)
plt.imshow(covarq.cpu().numpy(), cmap='gray')

plt.figure(figsize=(6,6))
for i in range(49):
    plt.subplot(7,7,i+1)
    plt.imshow(Xq_N[i,:].reshape(32,32).cpu(), cmap='gray')
    plt.axis('off')

base = "./"
flowtype = "forwardKL" if forwardKLflow else "reverseKL"

# Define the directory path
directory_path = base + f"figs/mnist/{flowtype}"

# Check if the directory exists, and if not, create it
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

plt.savefig(base + f"figs/mnist/{flowtype}/gradest_0.png")

# %%
seq = 1
for i in range(10000):

    print("iteration:", i, "...")
    Xps, Xqs_N = subsample(XData, Xq_N, VGD_batchsize)
    Xps_N = F.avg_pool2d(Xps.reshape(-1,1,32,32), kernel_size=1, stride=1, padding=0)
    Xps_N = F.interpolate(Xps_N, size=32, mode='bilinear').reshape(-1,32*32)
    
    # # clip Xps_N and Xqs_N
    Xps_N = torch.clamp(Xps_N, 0, 1)
    Xqs_N = torch.clamp(Xqs_N, 0, 1)

    if DEBUG:
        print("mean of Xp", torch.mean(Xps_N, dim=0)[:5])
        print("mean of Xq", torch.mean(Xqs_N, dim=0)[:5])
        print("std of Xp", torch.std(Xps_N, dim=0)[:5])
        print("std of Xq", torch.std(Xqs_N, dim=0)[:5])
    
    lmbd = 0.0000
    med = comp_median(Xqs_N)/5
    
    if forwardKLflow:
        # Forward KL flow
        grad, sigma = infer_Chi2_cpp(Xps_N, Xqs_N, Xq_N, med, lmbd)
    else:
        # Reserse KL flow
        grad, sigma = infer_KL_cpp(Xps_N, Xqs_N, Xq_N, med, lmbd)
    
    #get rid of the bias term
    grad = grad[:, :32*32]
    
    # gradient variational descent
    Xq_N = Xq_N + .1*grad
    # clipping to [0,1]
    Xq_N = torch.clamp(Xq_N, 0, 1)
    
    if DEBUG:
        # print some samples of the gradient
        print(grad[:5, :])
    
    plt.figure(figsize=(6,6), num=999, clear=True)
    #plot the first 10 samples from xq
    for ii in range(49):
        plt.subplot(7,7,ii+1)
        plt.imshow(Xq_N[ii,:].detach().cpu().reshape(32,32), cmap='gray')
        plt.axis('off')
    # plt.show()
    plt.savefig(base + f"./figs/mnist/{flowtype}/gradest_{seq}.png")
    
    seq = seq + 1
    # Traj.append(Xq_N[0:10, :])

    print("done!\n")

torch.save(Xq_N, "Xq_N.pt")
# %%
Xq_N = torch.load("Xq_N.pt")
plt.figure(figsize=(6,6), num=999, clear=True)
#plot the first 10 samples from xq
for ii in range(49):
    plt.subplot(7,7,ii+1)
    plt.imshow(Xq_N[ii,:].cpu().reshape(32,32), cmap='gray')
    plt.axis('off')

# plot the k nearest neighbour of the first 10 samples from Xq_N
dist = comp_dist(Xq_N[:10, ], XData)
nearest_i = torch.argsort(dist, dim=1)[:, :7]

plt.figure(figsize=(6,6), num=999, clear=True)
for i in range(10):
    plt.subplot(10,8,i*8+1)
    plt.imshow(Xq_N[i,:].cpu().reshape(32,32), cmap='gray')
    for j in range(7):
        plt.subplot(10,8,i*8+j+2)
        plt.imshow(XData[nearest_i[i,j],:].cpu().reshape(32,32), cmap='gray')
        plt.axis('off')
    
# %%
