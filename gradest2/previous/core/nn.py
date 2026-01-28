# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
    
class LogiNet(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.fc1 = nn.Linear(m, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return self.fc3(x)
    
    