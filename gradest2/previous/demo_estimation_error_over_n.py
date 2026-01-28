# %%

from core.gradest import infer_sm, infer_KL

import torch
device = torch.device("cuda:0" if torch.device else "cpu")
# device = torch.device("cpu")
torch.set_default_device("cuda:0")

from IPython import display
from core.util import comp_median
from core.torchGF import gradest_nw

# %%
def infer_KL_cpp(Xp, Xq, X, sigma, lmbd):
    grad, sigma = infer_KL(Xp.detach().cpu().numpy(), Xq.detach().cpu().numpy(), X.detach().cpu().numpy(), sigma_chosen=sigma, lambda_chosen=lmbd)
    return torch.from_numpy(grad).to(device), sigma
def infer_cpp_sm(Xp, Xq, X, sigma, lmbd):
    grad, sigma = infer_sm(Xp.detach().cpu().numpy(), Xq.detach().cpu().numpy(), X.detach().cpu().numpy(), sigma_chosen=sigma, lambda_chosen=lmbd)
    return torch.from_numpy(grad).to(device), sigma
# %%
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
import numpy as np

err = []
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

# %%
ntrial = 30 # number of trials, the paper uses 30, but I recommend try 2 if you are in a hurry.
error_over_n = []
error2_over_n = []
error3_over_n = []
nsamples = [100, 500, 1000, 1500]

print("\n------------------------\n")
print("Number of trials: ", ntrial)
print("Number of samples: ", nsamples)
print("Please find generated figures (*.png) in the current folder.")
print("\n------------------------\n")
print("To Continue, press enter. ")
input()

# %%
for n in nsamples:
    err = []
    err2 = []
    err3 = []
    for seed in range(ntrial): 
        print("\n ------- n:", n, "seed:", seed, "-------\n")
        torch.manual_seed(seed)
        # generate training samples
        Xp = sample_p(n)
        Xq = sample_q(n)

        # generate benchmarking samples
        Xqt = sample_q(1000)

        print("local linear...")
        grad, sigma = infer_KL_cpp(Xp, Xq, Xqt, sigma=-1, lmbd = .0)
        grad = grad[:, :d]
        print("done!\n")
        
        print("kernel density gradient estimator...")
        med = comp_median(Xq)
        grad2 = gradest_nw(Xqt, Xp, Xq)
        print("done!\n")
        
        print("score matching...")
        grad3 = infer_cpp_sm(Xp, Xq, Xqt, sigma=med, lmbd=1e-7)[0]
        print("done!\n")
        
        Xqt.requires_grad = True
        tt = torch.log(density_p(Xqt) / density_q(Xqt))
        grad_Xqt = torch.autograd.grad(tt.sum(), Xqt)[0]
        
        # mean squared error
        e = torch.mean(torch.sum((grad - grad_Xqt)**2, 1))
        e2 = torch.mean(torch.sum((grad2 - grad_Xqt)**2, 1))
        e3 = torch.mean(torch.sum((grad3 - grad_Xqt)**2, 1))
        print("error local linear:", e.item())
        print("error KDE:", e2.item())
        print("error SM:", e3.item())
        
        err.append(e.item())
        err2.append(e2.item())
        err3.append(e3.item())
        print("\n ----------------- \n")

    error_over_n.append(err)
    error2_over_n.append(err2)
    error3_over_n.append(err3)

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 5))
# plot errorbar for each n
plt.errorbar(nsamples, np.mean(error_over_n, 1), yerr=np.std(error_over_n, 1)/np.sqrt(ntrial), fmt='-o')
plt.errorbar(nsamples, np.mean(error2_over_n, 1), yerr=np.std(error2_over_n, 1)/np.sqrt(ntrial), fmt='-o')
plt.errorbar(nsamples, np.mean(error3_over_n, 1), yerr=np.std(error3_over_n, 1)/np.sqrt(ntrial), fmt='-o')
plt.legend(['local linear', 'kernel density gradient estimator', 'score matching'], fontsize = 14)
# plt.yscale('log')
plt.xlabel('n', fontsize = 14)
plt.ylabel('MSE', fontsize = 14)
plt.gca().tick_params(axis='both', labelsize=14)
plt.savefig('error_over_n.png')

 # %%
