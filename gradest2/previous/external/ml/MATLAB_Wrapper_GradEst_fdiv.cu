/*
 * Example of how to use the mxGPUArray API in a MEX file.  This example shows
 * how to write a MEX function that takes a gpuArray input and returns a
 * gpuArray output, e.g. B=mexFunction(A).
 *
 * Copyright 2012 The MathWorks, Inc.
 */

#include "mex.h"
#include "../cpp/juzhen.hpp" 
#include "../ml/gradest_fdiv.cuh"

template <class T>
inline Matrix<T> KLFcon(Matrix<T>&& m) {
    return std::move(exp(m - 1));
}

template <class T>
inline Matrix<T> dKLFcon(Matrix<T>&& m) {
    return std::move(exp(m - 1));
}

template <class T>
inline Matrix<T> Chi2Fcon(Matrix<T>&& m) {
    return std::move(square(m)/2 + m);
}

template <class T>
inline Matrix<T> dChi2Fcon(Matrix<T>&& m) {
    return std::move(m+1);
}

/* Input Arguments */

#define XP_IN prhs[0] 
#define XQ_IN prhs[1]
#define XT_IN prhs[2]
#define SIGMA_CHOSEN prhs[3]
#define LAMBDA_CHOSEN prhs[4]
#define MAX_ITER prhs[5]

/* Output Arguments */
#define GRAD_OUT plhs[0]
#define SIGMA_OUT plhs[1]

/*
 * Host code
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{   
    std::cout.precision(5);
#ifdef DEBUG
    std::cout << "num of input: " << nrhs << std::endl;
    std::cout << "num of output: " << nlhs << std::endl;
#endif

    size_t d, np, nq, nt, maxiter;
    np = mxGetM(XP_IN);
    d = mxGetN(XP_IN);    
    nq = mxGetM(XQ_IN);
    nt = mxGetM(XT_IN);
    maxiter = mxGetScalar(MAX_ITER);

    float sigma_chosen = mxGetScalar(SIGMA_CHOSEN);
    float lambda_chosen = mxGetScalar(LAMBDA_CHOSEN);
    
#ifdef DEBUG
    std::cout << "d: " << d << std::endl;
    std::cout << "np: " << np << std::endl;
    std::cout << "nq: " << nq << " nt: " << nt << std::endl;
    std::cout << "sigma: " << sigma_chosen << " lambda: " << lambda_chosen << std::endl;
    std::cout << "maxiter: " << maxiter << std::endl;
#endif

    {
        Memory<int> mdi;
        Memory<float> md;
#ifndef CPU_ONLY
        Memory<CUDAfloat> gpumd;
        CuBLASErrorCheck(cublasCreate(&Matrix<CUDAfloat>::global_handle));
    
        CM Xp("Xp", np, d);
        cudaMemcpy((float *) Xp.data(), mxGetPr(XP_IN), sizeof(float)*np*d, cudaMemcpyHostToDevice);
        CM Xq("Xq", nq, d);
        cudaMemcpy((float *) Xq.data(), mxGetPr(XQ_IN), sizeof(float)*nq*d, cudaMemcpyHostToDevice);
        CM X("Xt", nt, d);
        cudaMemcpy((float *) X.data(), mxGetPr(XT_IN), sizeof(float)*nt*d, cudaMemcpyHostToDevice);
#else
        M Xp("Xp", np, d);
        memcpy(Xp.data(), mxGetPr(XP_IN), sizeof(float)*np*d);
        M Xq("Xq", nq, d);
        memcpy(Xq.data(), mxGetPr(XQ_IN), sizeof(float)*nq*d);
        M X("Xt", nt, d);
        memcpy(X.data(), mxGetPr(XT_IN), sizeof(float)*nt*d);
#endif

        auto ret = infer_fdiv(Xp, Xq, X, KLFcon, dKLFcon, sigma_chosen, lambda_chosen, maxiter);
    
        GRAD_OUT = mxCreateNumericMatrix(nt, d+1, mxSINGLE_CLASS, mxREAL);
        
#ifndef CPU_ONLY
        cudaMemcpy(mxGetPr(GRAD_OUT), (float *) ret.grad.data(), sizeof(float)*nt*(d+1), cudaMemcpyDeviceToHost);
#else
        memcpy(mxGetPr(GRAD_OUT), ret.grad.data(), sizeof(float)*nt*(d+1));
#endif

        std::cout << ret.sigma << std::endl;
        SIGMA_OUT = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
        float *psigmaOut = (float *) mxGetPr(SIGMA_OUT);
        *psigmaOut = ret.sigma;
        
#ifndef CPU_ONLY
        CuBLASErrorCheck(cublasDestroy(Matrix<CUDAfloat>::global_handle));
#endif
    }
}
