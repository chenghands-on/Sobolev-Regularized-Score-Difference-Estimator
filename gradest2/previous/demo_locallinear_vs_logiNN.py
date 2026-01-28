# %%
from core.gradest import version, infer_KL

import torch
from core.nn import LogiNet
from core.torchGF import gradest_logidre
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
torch.set_default_device(device)

from core.util import kernel_comp

# %%

def infer_KL_cpp(Xp, Xq, X, sigma, lmbd):
    grad, sigma = infer_KL(Xp.detach().cpu().numpy(), Xq.detach().cpu().numpy(), X.detach().cpu().numpy(), sigma_chosen=sigma, lambda_chosen=lmbd, maxiter=2000)
    return torch.from_numpy(grad).to(device), sigma

# %%
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
import numpy as np
import matplotlib.pyplot as plt

d = 1

def sample_p(n):
    x1 = MVN(torch.zeros(d) - 5, torch.eye(d)*.5).sample((int(n/3),))
    x2 = MVN(torch.zeros(d) + 0, torch.eye(d)*.5).sample((int(n/3),))
    x3 = MVN(torch.zeros(d) + 5, torch.eye(d)*.5).sample((int(n/3),))
    x = torch.cat([x1, x2, x3], 0)
    # shuffle rows of x
    return x[torch.randperm(x.size()[0])]
    
    
def sample_q(n):
    x1 = MVN(torch.zeros(d) - 5, torch.eye(d)).sample((int(n/3),))
    x2 = MVN(torch.zeros(d) + 0, torch.eye(d)).sample((int(n/3),))
    x3 = MVN(torch.zeros(d) + 5, torch.eye(d)).sample((int(n/3),))
    x = torch.cat([x1, x2, x3], 0)
    # shuffle x
    return x[torch.randperm(x.size()[0])]

def density_p(x):
    return MVN(torch.zeros(d) - 5, torch.eye(d)*.5).log_prob(x).exp() / 3 \
         + MVN(torch.zeros(d) + 0, torch.eye(d)*.5).log_prob(x).exp() / 3 \
         + MVN(torch.zeros(d) + 5, torch.eye(d)*.5).log_prob(x).exp() / 3
def density_q(x):
    return MVN(torch.zeros(d) - 5, torch.eye(d)).log_prob(x).exp() / 3 \
         + MVN(torch.zeros(d) + 0, torch.eye(d)).log_prob(x).exp() / 3 \
         + MVN(torch.zeros(d) + 5, torch.eye(d)).log_prob(x).exp() / 3

# plot density p and q
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
bound = 8
ax.set_xlim(-bound, bound)
xmarks = torch.reshape(torch.tensor(np.linspace(-bound, bound, 100)).to(device), (100, 1))
plt.plot(np.linspace(-bound, bound, 100), torch.log(density_p(xmarks)/density_q(xmarks)).cpu(), label='log r')

#plot xp and xq, log ratio
Xp = sample_p(1000)
Xq = sample_q(1000)
plt.scatter(Xq.cpu().numpy(), torch.log(density_p(Xq)/density_q(Xq)).cpu(), label='Xq')
plt.scatter(Xp.cpu().numpy(), torch.log(density_p(Xp)/density_q(Xp)).cpu(), label='Xp')
plt.legend()

# title
plt.title(f"log r")
plt.savefig('logr.png')

# %% 
n = 5000
seed = 1
torch.manual_seed(seed)

print("\n------------------------\n")
print("Random Seed: ", seed)
print("Sample Size: ", n)

print("Please find generated figures (*.png) in the current folder.\n")
print("\n------------------------\n")
print("To Continue, press enter. ")
input()

Xp = sample_p(n)
Xq = sample_q(n)

xmarks = torch.linspace(-bound, bound, 100).reshape(100, 1).to(device)
xmarks.requires_grad = True
    
print("Estimating using local linear estimator...")
grad_loglinear, sigma = infer_KL_cpp(Xp, Xq, xmarks, -1, lmbd = .0)
grad_loglinear = grad_loglinear[:, :d]
print("done!"); print()

print("Estimating using logistic regression with NN...")
net2 = LogiNet(d)
grad_logi = gradest_logidre(xmarks, Xp, Xq, net2, 0.0)

print("done!"); print()

plt.figure()            
plt.plot(xmarks.detach().cpu(), grad_loglinear.detach().cpu(), label = 'local linear')
plt.plot(xmarks.detach().cpu(), grad_logi.detach().cpu(), label = 'logistic with NN')
grad_true = torch.autograd.grad(torch.log(density_p(xmarks)/density_q(xmarks)).sum(), xmarks)[0]
plt.plot(np.linspace(-bound, bound, 100), grad_true.cpu(), label='ground truth')
plt.legend()

# title
plt.title(f"gradient estimation comparison")
plt.savefig('Local_Linear_vs_LogiNN.png')

plt.show()        
# %%