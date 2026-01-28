import torch

def logidreloss(fp, fq):
       
    return torch.mean(torch.mean(torch.log(1+torch.exp( - fp )), 0)) \
         + torch.mean(torch.mean(torch.log(1+torch.exp(   fq )), 0))
