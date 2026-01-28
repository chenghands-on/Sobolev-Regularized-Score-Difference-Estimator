from IPython import display

import pylab as pl
import matplotlib.pyplot as plt
import torch
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import load_iris
import numpy as np
from matplotlib.colors import ListedColormap

def comp_dist(x,y):
    # in case x and y are not vectors
    x = x.view(x.shape[0], -1)
    y = y.view(y.shape[0], -1)
    t1 = torch.tile(torch.sum(x**2, dim=1, keepdim=True), (1, y.shape[0]))
    t2 = -2*torch.matmul(x, y.T)
    t3 = torch.tile(torch.sum(y**2, dim=1, keepdim=True).T, (x.shape[0], 1))
    return t1 + t2 + t3

def comp_median(x):
    return torch.sqrt(.5*comp_dist(x, x).flatten().median())

def kernel_comp(x, y, sigma):
    x = x.view(x.shape[0], -1)
    y = y.view(y.shape[0], -1)
    # compute a kernel matrix
    t1 = torch.tile(torch.sum(x**2, dim=1, keepdim=True), (1, y.shape[0]))
    t2 = -2*torch.matmul(x, y.T)
    t3 = torch.tile(torch.sum(y**2, dim=1, keepdim=True).T, (x.shape[0], 1))
    return torch.exp(- (t1 + t2 + t3)/2/(sigma**2))

def dKernel_comp(k, x, y, sigma, dim):
    # compute the derivative of a kernel matrix with respect to the input x at dimension dim
    d = x.shape[1]
    if dim < d:
        return -k/sigma**2 * (torch.tile(x[:,dim:dim+1], (1, y.shape[0])) - torch.tile(y[:,dim:dim+1].T, (x.shape[0], 1)))
    
    

# load mnist dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms


def load_mnist(device):
    # rescaling the MNIST images to 32x32
    image_size = 32
    # grayscale, resize, center, and convert to tensor
    transform=transforms.Compose([
                                transforms.Grayscale(num_output_channels=1),
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor()
                                ])

    dataroot = "./data"
    trainset = dset.MNIST(root=dataroot, train=True, download=True, transform=transform)
    # trainset = dset.FashionMNIST(root=dataroot, train=True, download=True, transform=transform)
    # trainset = dset.CIFAR10(root=dataroot, train=True, download=True, transform=transform)
    # trainset = dset.CIFAR100(root=dataroot, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=True)
    XData, yData = next(iter(trainloader)); XData = XData.view(XData.shape[0], -1).to(device)
    return XData, yData

def plot_images(xq, filename=None):
    pl.clf()
    #plot the first 10 samples from xq
    for ii in range(49):
        plt.subplot(7,7,ii+1)
        plt.imshow(xq[ii,:].detach().cpu().reshape(32,32), cmap='gray')
        plt.axis('off')
    
    display.clear_output(wait=True)
    display.display(pl.gcf())
    
    if filename is not None:
        pl.savefig(filename)
    
    
def svm(x,y, xt, yt, gamma = 1):

    # Split the data into training and test sets (optional)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Define parameter grid
    param_grid = {
        'C': np.logspace(-3, 3, 5),
        'gamma': np.linspace(.2, 5, 5) * gamma,
        'kernel': ['rbf']
        # 'C': [1],
        # 'kernel': ['linear']
    }

    # Create a SVC classifier with fixed random_state for reproducibility
    svc = SVC(random_state=42)

    # Initialize GridSearchCV with fixed random_state for cross-validation
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(svc, param_grid, refit=True, verbose=2, cv=cv)

    # Fit the model
    grid_search.fit(X_train, y_train)

    # # Best parameters and best score
    # print("Best Parameters:", grid_search.best_params_)
    # print("Best Score:", grid_search.best_score_)

    # Evaluate on test set (optional)
    test_accuracy = grid_search.score(xt, yt)
    print("Test Set Accuracy:", test_accuracy)
    
    return grid_search.predict(xt), test_accuracy
    
    # # Train the best model on the full training set
    # best_svc = grid_search.best_estimator_
    # best_svc.fit(X_train, y_train)

    # # Create a mesh grid for plotting
    # x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    # y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.arange(-2, 2, 0.02),
    #                     np.arange(-2, 2, 0.02))

    # # Predict decision function for each point in the mesh grid
    # Z = best_svc.decision_function(np.c_[xx.ravel(), yy.ravel()])
    # Z = Z.reshape(xx.shape)

    # # Plot decision boundary and margins
    # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    # plt.scatter(xt[:,0], xt[:,1], s=1, c='g')
    # plt.scatter(x[y==1,0], x[y==1,1], s=1, c='r')
    # plt.scatter(x[y==0,0], x[y==0,1], s=1, c='b')
    # plt.xlim(-2, 2); plt.ylim(-2, 2)
    
def svmplot(x, y, xt, yt, gamma=1, plotname="svmplot.png"):

    # Split the data into training and test sets (optional)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Define parameter grid
    param_grid = {
        'C': np.logspace(-3, 3, 5),
        'gamma': np.linspace(.2, 5, 5) * gamma,
        'kernel': ['linear']
    }

    # Create a SVC classifier
    svc = SVC()

    # Initialize GridSearchCV
    grid_search = GridSearchCV(svc, param_grid, refit=True, verbose=2, cv=5)

    # Fit the model
    grid_search.fit(X_train, y_train)

    # # Best parameters and best score
    # print("Best Parameters:", grid_search.best_params_)
    # print("Best Score:", grid_search.best_score_)

    # Evaluate on test set (optional)
    test_accuracy = grid_search.score(xt, yt)
    print("Test Set Accuracy:", test_accuracy)
        
    # Train the best model on the full training set
    best_svc = grid_search.best_estimator_
    best_svc.fit(X_train, y_train)

    # Create a mesh grid for plotting
    # x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    # y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(-4, 4, 0.02),
                        np.arange(-4, 4, 0.02))

    # Predict decision function for each point in the mesh grid
    Z = best_svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Define color maps for the decision boundary and scatter plot
    cmap_light = ListedColormap(['r', 'g', 'b']) # Light colors for decision boundaries
    cmap_bold = ['red', 'green', 'blue'] # Bold colors for scatter plot points

    plt.figure(figsize=(5,5))
    # Plot the decision boundaries
    # Plot the decision boundaries
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.35)
        # Plot the scatter plot
    for i, color in enumerate(cmap_bold):
        idx = np.where(yt == i+1)
        plt.scatter(xt[idx, 0], xt[idx, 1], c=color, edgecolors=color)
    # plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='g')
    plt.title('test accuracy: ' + str(test_accuracy))
    plt.savefig(plotname)
    
    return grid_search.predict(xt), test_accuracy
    
    
def svm_simple(x,y, xt, gamma, C):
    # train a svm with a given gamma and C
    clf = SVC(gamma=gamma, C=C)
    clf.fit(x, y)
    return clf.predict(xt)
    