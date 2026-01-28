/*
 * some testing code
 */
#define DEBUG

#include "../cpp/juzhen.hpp" 
#include "../ml/gradest.cuh"

#ifdef CPU_ONLY
#define FLOAT float
#else
#define FLOAT CUDAfloat
#endif

template <class T>
inline void dump(std::string fname, const Matrix<T> &M){
#ifdef CPU_ONLY
        write(fname, M);
#else
        write(fname, M.to_host());
#endif
}

int compute() {
    // spdlog::set_level(spdlog::level::debug);
    std::string base = PROJECT_DIR + std::string("/data/");
#ifndef CPU_ONLY
    GPUSampler s(1234);
#else
    global_rand_gen.seed(0);
#endif
    int np = 5001;
    int nq = 5002;
    int nt = 5000;
    int d = 2;

    auto Xp = Matrix<FLOAT>::randn(np, d)*1 + 1.0;
    auto Xt = Matrix<FLOAT>::randn(nt, d)*1 + 1.0;

    auto Xq = Matrix<FLOAT>::randn(nq, d)*1;
    
    TIC(gf);
    auto [grad, sigma] = infer(Xp, Xq, Xq, -1, 0);
    TOC(gf); 

    std::cout << "grad: " << grad.rows(0,d) << std::endl;
    std::cout << "sigma_chosen: " << sigma << std::endl;
    
    return 0;
}
