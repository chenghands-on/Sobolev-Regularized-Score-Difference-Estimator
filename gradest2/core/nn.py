
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

def gradx(Wx, S, x):
    gradx = torch.zeros(0).to('cuda')
    # construct dataload from x
    dataloader = DataLoader( torch.utils.data.TensorDataset(x), 
                                                batch_size=10000, shuffle=False)
    for i, data in enumerate(dataloader):
        # compute the gradient of resnet50
        xi = data[0].to('cuda')
        xi.requires_grad = True
        
        sxi = S(xi)
        m = sxi.shape[1]
        Wxi = Wx(sxi)[:, :m].detach()
        
        f = torch.sum(torch.sum(sxi * Wxi, 1),0)
        f.backward()
        
        gradx = torch.cat((gradx, xi.grad.detach()), 0)
        
        if i % 100 == 0:
            print("i = ", i)
        
    return gradx

# a d-in, d+1 out neural network with one hidden layer of size h

class NN(nn.Module):
    def __init__(self, n, m):
        super().__init__()
        self.fc1 = nn.Linear(m, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, m+1)
        # self.Wb = nn.Parameter(torch.zeros(n, m+1))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)
        # return self.Wb
        

# create a toy version of the above network
class NN_toy(nn.Module):
    def __init__(self, n, m):
        super().__init__()
        self.fc1 = nn.Linear(m, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, m+1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
# create a toy version of the above network
class JackNet(nn.Module):
    def __init__(self, n, m):
        super().__init__()
        self.fc1 = nn.Linear(m, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, m+1)

    def forward(self, x):
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        x = F.silu(self.fc3(x))
        x = F.silu(self.fc4(x))
        x = F.silu(self.fc5(x))
        return self.fc6(x)
    
# Classifier network for density ratio estimation (outputs logit = log(p/q))
class DensityRatioClassifier(nn.Module):
    def __init__(self, d, h=64, activation='relu'):
        """
        Args:
            d: input dimension
            h: hidden dimension
            activation: 'relu' or 'silu' (SiLU/Swish activation)
        """
        super().__init__()
        if activation == 'silu':
            act = nn.SiLU()
        else:
            act = nn.ReLU()
        
        self.net = nn.Sequential(
            nn.Linear(d, h), act,
            nn.Linear(h, h), act,
            nn.Linear(h, 1)  # Output: logit = log(p/q)
        )
    
    def forward(self, x):
        return self.net(x)
    

class NPnet(nn.Module):
    def __init__(self, n, m):
        super().__init__()
        self.Wb = nn.Parameter(torch.zeros(n, m+1))

    def forward(self, x):
        return self.Wb
        
# a d-in, d+1 out neural network with one hidden layer of size h

class timeNN(nn.Module):
    def __init__(self, n, m):
        super().__init__()
        self.fc1 = nn.Linear(m+1, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, m+1)
        # self.Wb = nn.Parameter(torch.randn(n, m+1))

    def forward(self, x, t):
        x = torch.cat((x, t * torch.ones(x.shape[0],1,device='cuda')), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)
        # return self.Wb
        
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward pass of LSTM layer
        out, _ = self.lstm(x, (h0, c0))

        # Decode hidden state of the last time step
        out = self.fc(out[:, -1, :])

        return out
    
class SegmentationCNN(nn.Module):
    def __init__(self):
        super(SegmentationCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Output layer - 2 filters for 2 output channels
        self.out = nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        #reshape into one channel 32 by 32
        x = x.view(x.shape[0], 1, 32, 32)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Output layer
        x = self.out(x)  # No activation function here, can be added based on the use-case
        # reshape into 1024 dimensional vector
        b = torch.mean(x[:, 1, :, :].reshape(x.shape[0], 1024), 1, keepdim=True)
        w = x[:, 0, :, :].reshape(x.shape[0], -1)
        
        return torch.cat((w, b), 1)
    

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc = nn.Linear(in_features=64 * 8 * 8, out_features=1024)

    def forward(self, x):
        x = x.view(x.shape[0], 1, 32, 32)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)  # Flatten the feature maps
        x = self.fc(x)
        return x
    
# GAN generator for grayscale images

# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.fc1 = nn.Linear(100, 128)
#         self.fc2 = nn.Linear(128, 256)
#         self.fc3 = nn.Linear(256, 512)
#         self.fc4 = nn.Linear(512, 1024)
#         self.fc5 = nn.Linear(1024, 32*32)
        
#     def forward(self, x):
#         x = F.leaky_relu(self.fc1(x), 0.2)
#         x = F.leaky_relu(self.fc2(x), 0.2)
#         x = F.leaky_relu(self.fc3(x), 0.2)
#         x = F.leaky_relu(self.fc4(x), 0.2)
#         x = torch.tanh(self.fc5(x))
#         return x

# Generator Code

# Generator Code

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.ngpu = 1
        nz = 100
        ngf = 64
        nc = 1
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Flatten()
        )

    def forward(self, input):
        return self.main(input)
        