# %%

import time
import ctypes
from numpy import *
import platform

import torch
if platform.system() == 'Linux':
    print("Linux detected, loading linux library")
    if torch.cuda.is_available():
        libc = ctypes.cdll.LoadLibrary("core/PythonWrapper.so")
    else: 
        print("CUDA not available. Please run this on a machine with CUDA enabled.")
elif platform.system() == 'Windows':
    print("Windows detected, loading windows library")
    #check if cuda is available
    if torch.cuda.is_available():
        libc = ctypes.CDLL("core/PythonWrapperGradEst.dll", winmode=0)
    else: 
        print("CUDA not available. Please run this on a machine with CUDA enabled.")
elif platform.system() == 'Darwin':
    libc = ctypes.cdll.LoadLibrary("core/PythonGradEst.dylib")
    
def version():
    libc.info()

def infer_ULSIF(xp, xq, x, sigma_chosen = -1, lambda_chosen = 0, maxiter = 2000):    
    
    d = xp.shape[1]
    np = xp.shape[0]
    nq = xq.shape[0]
    n = x.shape[0]

    grad = zeros(n*(d+1), dtype=float32)
    sigma = zeros(1, dtype=float32)

    libc.GF_ULSIF.argtypes = [ctypeslib.ndpointer(dtype=float32), 
                                ctypeslib.ndpointer(dtype=float32), 
                                ctypeslib.ndpointer(dtype=float32), 
                                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, 
                                ctypes.c_float,
                                ctypes.c_float,
                                ctypes.c_int,
                                ctypeslib.ndpointer(dtype=float32),
                                ctypeslib.ndpointer(dtype=float32)]
    libc.GF_ULSIF.restype = None

    start = time.perf_counter()
    libc.GF_ULSIF(xp.ravel('F'), xq.ravel('F'), x.ravel('F'), np, nq, n, d, sigma_chosen, lambda_chosen, maxiter, grad, sigma)
    end = time.perf_counter()
    # print(f"execution time: {end-start: 0.4f} seconds")

    grad = grad.reshape((n,d+1), order='F')
    sigma = sigma[0]
    
    return grad, sigma

def infer_KL(xp, xq, x, sigma_chosen = -1, lambda_chosen = 0, maxiter = 2000):    
    
    d = xp.shape[1]
    np = xp.shape[0]
    nq = xq.shape[0]
    n = x.shape[0]

    grad = zeros(n*(d+1), dtype=float32)
    sigma = zeros(1, dtype=float32)

    libc.GF_KL.argtypes = [ctypeslib.ndpointer(dtype=float32), 
                                ctypeslib.ndpointer(dtype=float32), 
                                ctypeslib.ndpointer(dtype=float32), 
                                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, 
                                ctypes.c_float,
                                ctypes.c_float,
                                ctypes.c_int, 
                                ctypeslib.ndpointer(dtype=float32),
                                ctypeslib.ndpointer(dtype=float32)]
    libc.GF_KL.restype = None

    start = time.perf_counter()
    libc.GF_KL(xp.ravel('F'), xq.ravel('F'), x.ravel('F'), np, nq, n, d, sigma_chosen, lambda_chosen, maxiter, grad, sigma)
    end = time.perf_counter()
    # print(f"execution time: {end-start: 0.4f} seconds")

    grad = grad.reshape((n,d+1), order='F')
    sigma = sigma[0]
    
    return grad, sigma

def infer_sm(xp, xq, x, sigma_chosen, lambda_chosen):    
    
    d = xp.shape[1]
    np = xp.shape[0]
    nq = xq.shape[0]
    n = x.shape[0]

    grad = zeros(n*d, dtype=float32)
    sigma = zeros(1, dtype=float32)

    libc.SMGF.argtypes = [ctypeslib.ndpointer(dtype=float32), 
                                ctypeslib.ndpointer(dtype=float32), 
                                ctypeslib.ndpointer(dtype=float32), 
                                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, 
                                ctypes.c_float,
                                ctypes.c_float,
                                ctypeslib.ndpointer(dtype=float32),
                                ctypeslib.ndpointer(dtype=float32)]
    libc.SMGF.restype = None

    start = time.perf_counter()
    libc.SMGF(xp.ravel('F'), xq.ravel('F'), x.ravel('F'), np, nq, n, d, sigma_chosen, lambda_chosen, grad, sigma)
    end = time.perf_counter()
    # print(f"execution time: {end-start: 0.4f} seconds")

    grad = grad.reshape((n,d), order='F')
    sigma = sigma[0]
    
    return grad, sigma

