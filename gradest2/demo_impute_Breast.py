# %%
import matplotlib.pyplot as plt
import torch
from numpy import *
import numpy as np
from core.nn import SegmentationCNN, NN, NN_toy, NPnet, CustomCNN, JackNet
import torch.optim as optim
from core.util import comp_dist, comp_median, load_mnist, plot_images, kernel_comp
from wgf import gradest, MMD_flow
import pandas as pd
from IPython import display

import pylab as pl
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.io import loadmat, savemat

# %%
seed = 213
torch.manual_seed(seed)
random.seed(seed)

DEBUG = False

def kernel(x,y,sigma):
    # # the first hald, use the sigma1
    x1 = x[:, :x.shape[1]//2]
    y1 = y[:, :y.shape[1]//2]
    k1 = kernel_comp(x1, y1, sigma)
    
    # the second half, use the sigma2
    x2 = x[:, x.shape[1]//2:]
    y2 = y[:, y.shape[1]//2:]
    # linear kernel
    k2 = torch.matmul(x2, y2.T)
    
    return k1*k2

def kernel2(x,y,sigma):
    # # the first till the second last, use gaussian kernel
    x1 = x[:, :-1]
    y1 = y[:, :-1]

    dist2 = torch.cdist(x1, y1, p=2)
    
    k1 = -torch.sqrt(dist2)
    
    # the second half, use the sigma2
    x2 = x[:, x.shape[1]//2:]
    y2 = y[:, y.shape[1]//2:]
    # linear kernel
    k2 = torch.matmul(x2, y2.T)
    
    return k1*k2
    # return kernel_comp(x, y, sigma)

def calAUC(X_filled, labels):
    X_train, X_test, y_train, y_test = train_test_split(X_filled, labels, test_size=.3, random_state=5)
    # fit svm
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)
    # compute the testing AUC
    y_pred = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    return auc
# %%
for missingrate in [.5, .6, .7, .8]:
    gainauc_list = []
    sinkhornauc_list = []
    hyperauc_list = []
    miracleauc_list = []
    miwaeauc_list = []
    wgfauc_list = []
    baseauc_list = []
    
    for seed in range(10):
        # load the data
        data = loadmat('missingdata/{:d}_{:.2f}.mat'.format(1+seed, missingrate))
        X = torch.tensor(data['x0'])
        y = torch.tensor(data['y'])
        labels = torch.tensor(data['labels']).squeeze()
        X0 = X.clone()
        mask = 1 - y
        X_miss = X
        X_miss[mask == 1] = np.nan
        
        # X_basefill = X0.clone().to(device)
        # auc = calAUC(X_basefill.cpu().numpy(), labels)
        # print('seed: {:d} wgf AUC: {:.3f}'.format(seed, auc))
        # baseauc_list.append(auc)
        
        X_fill = X0.clone().to(dtype = torch.float32, device = device)
        y = y.clone().to(dtype = torch.float32, device = device)
        zp = torch.cat([X_fill, y], 1)
        y_perm = torch.bernoulli(torch.ones_like(X_fill) * (1-missingrate)).to(device = device, dtype = torch.float32)
        zq = torch.cat([X_fill, y_perm], 1)
        
        train_loader = torch.utils.data.DataLoader( 
            torch.utils.data.TensorDataset(torch.tensor(range(zp.shape[0])), 
            torch.tensor(range(zq.shape[0]))), batch_size=zp.shape[0], shuffle=True)
        
        for epoch in range(500):
            
            for i, data in enumerate(train_loader):
                
                idxp, idxq = data
                zpi = zp[idxp, :]; zqi = zq[idxq, :]
                
                sigma = (.5*comp_dist(X_fill, X_fill).flatten().median()).sqrt() 
                
                net = NPnet(X_fill.shape[0], X_fill.shape[1]*2).to(device)
                # create the optimizer
                optimizer = optim.RMSprop(net.parameters(), lr = 1e-3, alpha = .9, eps = 1e-6)

                # Reserse KL flow
                gradnet = gradest(net, zpi, zqi, zq, sigma, optimizer, kernel = kernel, batch_size=zqi.shape[0], nepochs = 101)
                grad = gradnet(zq)[:, :X_fill.shape[1]].detach()
                
                # grad = MMD_flow(zp, zq, kernel = kernel2, sigma = None)[:, :X_fill.shape[1]].detach() * .0001
                
                X_fill[y == 0] = X_fill[y == 0] + 1*grad[y == 0]
                
                zp = torch.cat([X_fill, y], 1)
                y_perm = torch.bernoulli(torch.ones_like(X_fill) * (1-missingrate)).to(device = device, dtype = torch.float32)
                zq = torch.cat([X_fill, y_perm], 1)
                
                auc = calAUC(X_fill.cpu().numpy(), labels)
                # print('seed: {:d} wgf AUC: {:.3f}'.format(seed, auc))
                
        auc = calAUC(X_fill.cpu().numpy(), labels)
        print('seed: {:d} wgf AUC: {:.3f}'.format(seed, auc))
        wgfauc_list.append(auc)
        
        from hyperimpute.plugins.imputers import Imputers
        imputers = Imputers()

        # method = 'gain'
        # plugin = Imputers().get(method)
        # X = pd.DataFrame(X_miss.numpy())
        # X_filledG = plugin.fit_transform(X.copy())

        # auc = calAUC(X_filledG, labels)
        # print('seed: {:d} gain AUC: {:.3f}'.format(seed, auc))
        # gainauc_list.append(auc)
        # X_filledG = np.array(X_filledG)
        # savemat('res/irir_imputed_{:d}_{:.2f}_gain.mat'.format(1+seed, missingrate), {'x': X_filledG})
        
        # method = 'sinkhorn'
        # plugin = Imputers().get(method)
        # X = pd.DataFrame(X_miss.numpy())
        # X_filledS = plugin.fit_transform(X.copy())
        # X_filledS = np.array(X_filledS)
        # # save the imputed data
        # savemat('res/irir_imputed_{:d}_{:.2f}_sinkhorn.mat'.format(1+seed, missingrate), {'x': X_filledS})

        # auc = calAUC(X_filledS, labels)
        # print('seed: {:d} sinkhorn AUC: {:.3f}'.format(seed, auc))
        # sinkhornauc_list.append(auc)
        
        # method = 'hyperimpute'
        # plugin = Imputers().get(method)
        # X = pd.DataFrame(X_miss.numpy())
        # X_filledH = plugin.fit_transform(X.copy())
        
        # auc = calAUC(X_filledH, labels)
        # print('seed: {:d} hyper AUC: {:.3f}'.format(seed, auc))
        # hyperauc_list.append(auc)
        # X_filledH = np.array(X_filledH)
        # savemat('res/irir_imputed_{:d}_{:.2f}_hyper.mat'.format(1+seed, missingrate), {'x': X_filledH})
        
        # method = 'miracle'
        # plugin = Imputers().get(method)
        # X = pd.DataFrame(X_miss.numpy())
        # X_filledM = plugin.fit_transform(X.copy())
        
        # auc = calAUC(X_filledM, labels)
        # print('seed: {:d} miracle AUC: {:.3f}'.format(seed, auc))
        # miracleauc_list.append(auc)
        # X_filledM = np.array(X_filledM)
        # savemat('res/irir_imputed_{:d}_{:.2f}_miracle.mat'.format(1+seed, missingrate), {'x': X_filledM})
        
        # method = 'miwae'
        # plugin = Imputers().get(method)
        # X = pd.DataFrame(X_miss.numpy())
        # X_filledMI = plugin.fit_transform(X.copy())
        
        # auc = calAUC(X_filledMI, labels)
        # print('seed: {:d} miwae AUC: {:.3f}'.format(seed, auc))
        # miwaeauc_list.append(auc)
        # X_filledMI = np.array(X_filledMI)
        # savemat('res/irir_imputed_{:d}_{:.2f}_miwaeauc.mat'.format(1+seed, missingrate), {'x': X_filledMI})

        # X_filled2 = torch.tensor(data['x'])
        # auc = calAUC(X_filled2, labels)
        # print('seed: {:d} wgf AUC: {:.3f}'.format(seed, auc))
        # wgfauc_list.append(auc)

    print('missing rate: {:.2f}'.format(missingrate))
    # print('gain auc: {:.3f} +- {:.3f}'.format(np.mean(gainauc_list), np.std(gainauc_list)))
    # print('sinkhorn auc: {:.3f} +- {:.3f}'.format(np.mean(sinkhornauc_list), np.std(sinkhornauc_list)))
    # print('hyper auc: {:.3f} +- {:.3f}'.format(np.mean(hyperauc_list), np.std(hyperauc_list)))
    # print('miracle auc: {:.3f} +- {:.3f}'.format(np.mean(miracleauc_list), np.std(miracleauc_list)))
    # print('miwae auc: {:.3f} +- {:.3f}'.format(np.mean(miwaeauc_list), np.std(miwaeauc_list)))
    print('wgf auc: {:.3f} +- {:.3f}'.format(np.mean(wgfauc_list), np.std(wgfauc_list)))
    # print('base auc: {:.3f} +- {:.3f}'.format(np.mean(baseauc_list), np.std(baseauc_list)))
# %%

