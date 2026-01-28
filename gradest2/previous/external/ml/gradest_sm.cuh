
#include "../cpp/juzhen.hpp" 
#include "../ml/util.cuh" 

#ifdef CPU_ONLY
#define vstack vstack<T>
#else
#define vstack vstack
#endif

template <class T>
Matrix<T> SMupdate(const Matrix<T> &Xq, const Matrix<T> &X, float sigma = 1, float lambda = 0.0f) {
    int nq = Xq.num_row();
    int n = X.num_row();
    int d = X.num_col();

    int b = 501;
    if (nq < b){
        b = nq;
    }
    auto Xb = Xq.rows(0, b);

    auto Kbq = kernel_gau(comp_dist(Xb, Xq), sigma);
    
    auto enb1 = Matrix<T>::ones(b, 1);
    auto e1nq = Matrix<T>::ones(1, nq);
    auto dKbq = Matrix<T>::zeros(d, b);
    for (int i = 0; i < d; i++){
        auto Xbi = Xb.columns(i, i + 1);
        auto Xqi = Xq.columns(i, i + 1);
        auto dKi = sum(hadmd( (Xbi * e1nq - enb1 * Xqi.T())/sigma/sigma, Kbq), 1)/nq;
        dKbq.rows(i, i+1, dKi.T());
    }
    dKbq = hstack({dKbq, Matrix<T>::zeros(d, 1)});

    auto theta = Matrix<T>::zeros(d, b+1);
    
    const Matrix<T> Kbqone = vstack({Kbq, Matrix<T>::ones(1, nq)});
    auto Cq = Kbqone * Kbqone.T() / nq;
    
    float pre_gfn = 0.0f;
    float gfn = 0.0f;
    int i;
    // gradient descent for maximum 2000 iterations
    for (i = 0; i < 100000; i ++){

        auto gradTheta = theta * Cq + dKbq + 
             2 * lambda * hstack({theta.columns(0, b), Matrix<T>::zeros(d, 1)});
        float gfn = gradTheta.norm()/sqrt(n*d);
        if(std::isnan(gfn) || gfn < 1e-6){
            break;
        }
        else{ 
            theta -= 0.01*gradTheta;
            // std::cout << "theta norm: " << theta.norm() << std::endl;
            pre_gfn = gfn;
        }
    }

#ifdef DEBUG
    std::cout << "break at " << i <<", gfn: " << gfn << ", pre_gfn: " << pre_gfn << std::endl;
    std::cout << "sigma: " << sigma << ", lambda: " << lambda << std::endl;
#endif

    auto Kbx = kernel_gau(comp_dist(X, Xb), sigma);
    Kbx = hstack({Kbx, Matrix<T>::ones(n, 1)});
    return Kbx*theta.T();
}
