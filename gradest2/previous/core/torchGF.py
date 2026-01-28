# %%
import torch
from core.loss import logidreloss
from torch import optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np
from core.util import comp_median, kernel_comp

# %%
# estimate gradient of log r(x) w.r.t. x using logistic regression
def gradest_logidre(x0, xp, xq, net, lam):
    x0.requires_grad = True
    optimizer = optim.SGD(net.parameters(), lr=1e-2)
    
    old_para = net.forward(x0).clone().detach()
    n = xp.shape[0]
    batchsize = round(n/10)
    for i in range(10000):
        # select a random batch from xp and xq
        xp_i = xp[torch.randperm(xp.shape[0])[:batchsize], ]
        xq_i = xq[torch.randperm(xq.shape[0])[:batchsize], ]
        
        optimizer.zero_grad()
        loss = logidreloss(net.forward(xp_i), net.forward(xq_i))
        loss = loss + lam*torch.mean(net.forward(x0)**2)
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            newpara = net.forward(x0).clone().detach()        
            # compare with old para
            paradiff = (newpara- old_para).norm()
            if paradiff < 1e-5:
                print("break at", i)
                break
            else:
                print("paradiff", paradiff.item(), end = "\r")
        
            old_para = newpara
    
    gradx0 = torch.autograd.grad(net.forward(x0).sum(), x0)[0]
    return gradx0

# nadaraya-watson gradient estimator
def gradest_nw(X0, Xp, Xq):
    nwgradest = []

    medp = comp_median(Xp)
    medq = comp_median(Xq)

    Xp.requires_grad = True
    Xq.requires_grad = True

    for i in range(X0.shape[0]):
        Kpt = kernel_comp(Xp, X0[i:i+1, :], medp)
        nablaxKpt = -torch.autograd.grad(Kpt.sum(dim=0), Xp)[0]
        
        Kqt = kernel_comp(Xq, X0[i:i+1, :], medq)
        nablaxKqt = -torch.autograd.grad(Kqt.sum(dim=0), Xq)[0]
        
        t = torch.mean(nablaxKpt,0)/torch.mean(Kpt,0) - torch.mean(nablaxKqt,0)/torch.mean(Kqt,0)
        nwgradest.append(t)
        # print(i)
        
    return torch.stack(nwgradest)