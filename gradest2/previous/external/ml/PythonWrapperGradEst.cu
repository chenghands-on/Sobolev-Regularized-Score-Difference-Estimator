/*
 * Example of how to use the mxGPUArray API in a MEX file.  This example shows
 * how to write a MEX function that takes a gpuArray input and returns a
 * gpuArray output, e.g. B=mexFunction(A).
 *
 * Copyright 2012 The MathWorks, Inc.
 */
#include "../cpp/juzhen.hpp"
#include "../ml/gradest_sm.cuh"
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

extern "C" {
#ifdef _WIN64
__declspec(dllexport) void __cdecl info() {
#else
void info(){
#endif
        {
            std::cout << "Gradient Flow: Version 1.0" << std::endl;
            
            int a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
            int len = 10;
            for (int i = 0; i < len; i++) {
                std::cout << a[i] << " ";
            }
            std::cout << std::endl;
        }
    }
}

extern "C" {
#ifdef _WIN64
__declspec(dllexport) void __cdecl GF_ULSIF(float *xp, float *xq, float *x, int np, int nq, int n,
                 int d, float sigma_chosen, float lambda_chosen, int maxiter, float *grad_out, float *sigma_out) {
#else
void GF_ULSIF(float *xp, float *xq, float *x, int np, int nq, int n,
                 int d, float sigma_chosen, float lambda_chosen, int maxiter, float *grad_out, float *sigma_out) {
#endif
    std::cout.precision(5);

#ifdef DEBUG
    std::cout << "d: " << d << std::endl;
    std::cout << "np: " << np << std::endl;
    std::cout << "nq: " << nq << std::endl;
    std::cout << "n: " << n << std::endl;
#endif

    {
#ifndef CPU_ONLY
        CuBLASErrorCheck(cublasCreate(&Matrix<CUDAfloat>::global_handle));
        Memory<CUDAfloat> gpumd;
#endif
        Memory<int> mdi;
        Memory<float> md;

#ifndef CPU_ONLY
        CM Xp("Xp", np, d);
        cudaMemcpy((float *)Xp.data(), xp, sizeof(float) * np * d,
                   cudaMemcpyHostToDevice);
        CM Xq("Xq", nq, d);
        cudaMemcpy((float *)Xq.data(), xq, sizeof(float) * nq * d,
                   cudaMemcpyHostToDevice);
        CM X("X", n, d);
        cudaMemcpy((float *)X.data(), x, sizeof(float) * n * d,
                   cudaMemcpyHostToDevice);
#else
        M Xp("Xp", np, d);
        memcpy((float *)Xp.data(), xp, sizeof(float) * np * d);
        M Xq("Xq", nq, d);
        memcpy((float *)Xq.data(), xq, sizeof(float) * nq * d);
        M X("X", n, d);
        memcpy((float *)X.data(), x, sizeof(float) * n * d);
#endif

        auto [grad, sigma] = infer_fdiv(Xp, Xq, X, Chi2Fcon, dChi2Fcon, sigma_chosen, lambda_chosen, maxiter);

#ifndef CPU_ONLY
        CuBLASErrorCheck(cublasDestroy(Matrix<CUDAfloat>::global_handle));

        cudaMemcpy(grad_out, (float *)grad.data(),
                   sizeof(float) * n * (d+1), cudaMemcpyDeviceToHost);
#else
        memcpy(grad_out, (float *)grad.data(), sizeof(float) * n * d);
#endif
        *sigma_out = sigma;
        
    }
}
}

extern "C" {
#ifdef _WIN64
__declspec(dllexport) void __cdecl GF_KL(float *xp, float *xq, float *x, int np, int nq, int n,
                 int d, float sigma_chosen, float lambda_chosen, int maxiter, float *grad_out, float *sigma_out) {
#else
void GF_KL(float *xp, float *xq, float *x, int np, int nq, int n,
                 int d, float sigma_chosen, float lambda_chosen, int maxiter, float *grad_out, float *sigma_out) {
#endif
    std::cout.precision(5);

#ifdef DEBUG
    std::cout << "d: " << d << std::endl;
    std::cout << "np: " << np << std::endl;
    std::cout << "nq: " << nq << std::endl;
    std::cout << "n: " << n << std::endl;
#endif

    {
#ifndef CPU_ONLY
        CuBLASErrorCheck(cublasCreate(&Matrix<CUDAfloat>::global_handle));
        Memory<CUDAfloat> gpumd;
#endif
        Memory<int> mdi;
        Memory<float> md;

#ifndef CPU_ONLY
        CM Xp("Xp", np, d);
        cudaMemcpy((float *)Xp.data(), xp, sizeof(float) * np * d,
                   cudaMemcpyHostToDevice);
        CM Xq("Xq", nq, d);
        cudaMemcpy((float *)Xq.data(), xq, sizeof(float) * nq * d,
                   cudaMemcpyHostToDevice);
        CM X("X", n, d);
        cudaMemcpy((float *)X.data(), x, sizeof(float) * n * d,
                   cudaMemcpyHostToDevice);
#else
        M Xp("Xp", np, d);
        memcpy((float *)Xp.data(), xp, sizeof(float) * np * d);
        M Xq("Xq", nq, d);
        memcpy((float *)Xq.data(), xq, sizeof(float) * nq * d);
        M X("X", n, d);
        memcpy((float *)X.data(), x, sizeof(float) * n * d);
#endif

        auto [grad, sigma] = infer_fdiv(Xp, Xq, X, KLFcon, dKLFcon, sigma_chosen, lambda_chosen, maxiter);

#ifndef CPU_ONLY
        CuBLASErrorCheck(cublasDestroy(Matrix<CUDAfloat>::global_handle));

        cudaMemcpy(grad_out, (float *)grad.data(),
                   sizeof(float) * n * (d+1), cudaMemcpyDeviceToHost);
#else
        memcpy(grad_out, (float *)grad.data(), sizeof(float) * n * d);
#endif
        *sigma_out = sigma;
        
    }
}
}

extern "C" {
#ifdef _WIN64
__declspec(dllexport) void __cdecl SMGF(float* xp, float* xq, float* x, int np, int nq, int n,
                 int d, float sigma_chosen, float lambda_chosen, float* grad_out, float* sigma_out) {
#else
void SMGF(float *xp, float *xq, float *x, int np, int nq, int n,
                 int d, float sigma_chosen, float lambda_chosen, float *grad_out, float *sigma_out) {
#endif
    std::cout.precision(5);

#ifdef DEBUG
    std::cout << "d: " << d << std::endl;
    std::cout << "np: " << np << std::endl;
    std::cout << "nq: " << nq << std::endl;
    std::cout << "n: " << n << std::endl;
#endif

    {
#ifndef CPU_ONLY
        CuBLASErrorCheck(cublasCreate(&Matrix<CUDAfloat>::global_handle));
        Memory<CUDAfloat> gpumd;
#endif
        Memory<int> mdi;
        Memory<float> md;

#ifndef CPU_ONLY
        CM Xp("Xp", np, d);
        cudaMemcpy((float *)Xp.data(), xp, sizeof(float) * np * d,
                   cudaMemcpyHostToDevice);
        CM Xq("Xq", nq, d);
        cudaMemcpy((float *)Xq.data(), xq, sizeof(float) * nq * d,
                   cudaMemcpyHostToDevice);
        CM X("X", n, d);
        cudaMemcpy((float *)X.data(), x, sizeof(float) * n * d,
                   cudaMemcpyHostToDevice);
#else
        M Xp("Xp", np, d);
        memcpy((float *)Xp.data(), xp, sizeof(float) * np * d);
        M Xq("Xq", nq, d);
        memcpy((float *)Xq.data(), xq, sizeof(float) * nq * d);
        M X("X", n, d);
        memcpy((float *)X.data(), x, sizeof(float) * n * d);
#endif

        auto gradp = SMupdate(Xp, X, sigma_chosen, lambda_chosen);
        auto gradq = SMupdate(Xq, X, sigma_chosen, lambda_chosen);


#ifndef CPU_ONLY
        CuBLASErrorCheck(cublasDestroy(Matrix<CUDAfloat>::global_handle));

        cudaMemcpy(grad_out, (float *)(gradp - gradq).data(),
                   sizeof(float) * n * d, cudaMemcpyDeviceToHost);
#else
        memcpy(grad_out, (float *)grad.data(), sizeof(float) * n * d);
#endif
        *sigma_out = sigma_chosen;
        
    }

}
}