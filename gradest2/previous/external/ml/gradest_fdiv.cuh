#include "../cpp/juzhen.hpp" 
#include "../ml/util.cuh" 

#ifdef CPU_ONLY
#define vstack vstack<T>
#else
#define vstack vstack
#endif

template <class T>
struct result {Matrix<T> grad; float sigma;};

template <class T>
Matrix<T> update_fdiv(const Matrix<T> &Xp, const Matrix<T> &Xq, const Matrix<T> &X, 
                    Matrix<T> (*dfun_con)(Matrix<T> &&), float sigma = 1, float lambda = 0.0f, int maxiter = 2000) {

    int np = Xp.num_row();
    int nq = Xq.num_row();
    int n = X.num_row();
    int d = Xp.num_col();

    auto KX0q = kernel_gau(comp_dist(X, Xq), sigma);
    auto KX0p = kernel_gau(comp_dist(X, Xp), sigma);

    auto theta = Matrix<T>::zeros(n, d+1);
    auto fXp = hstack({Xp, Matrix<T>::ones(np, 1)});
    auto fXq = hstack({Xq, Matrix<T>::ones(nq, 1)});

    auto t1 = KX0p * fXp / np;
    
    float pre_gfn = 0.0f;
    float gfn = 0.0f;
    int i;
    
    adam_state<T> state(theta);
    // gradient descent for maximum 2000 iterations
    for (i = 0; i < maxiter; i ++){
        auto t2 = hadmd(dfun_con(theta * fXq.T()), KX0q) * fXq / nq;

        auto g = - t1 + t2;
        auto gradTheta = adam_update(std::move(g), state);
        // + 2*lambda*hstack({theta.columns(0, d), zn1});
        gfn = gradTheta.norm()/sqrt(n*d);
        if(std::isnan(gfn) || gfn < 1e-6){
            break;
        }
        else{
            theta -= gradTheta;
            pre_gfn = gfn;
        }
    }

#ifdef DEBUG
    std::cout << "break at " << i <<", gfn: " << gfn << ", pre_gfn: " << pre_gfn << std::endl;
    std::cout << "sigma: " << sigma << ", lambda: " << lambda << std::endl;
#endif

    return theta;
}

template <class T>
struct result<T> infer_fdiv(const Matrix<T> &Xp, const Matrix<T> &Xq, const Matrix<T> &X, 
                           Matrix<T> (*Fcon)(Matrix<T> &&), Matrix<T> (*DFcon)(Matrix<T> &&),
                          float sigma_chosen, float lambda_chosen = 0.0f, int maxiter = 2000){


    int np = Xp.num_row();
    int nq = Xq.num_row();
    int d = Xp.num_col();

    if(sigma_chosen < 0){

#ifdef DEBUG
        std::cout << "sigma not given, choosing sigma using cross validation..." << std::endl;
#else
        std::cout << "Running cross validation and please wait.\n To show debugging info, compile use DEBUG flag..." << std::endl;
#endif

        auto sigma = comp_med(Xq);
        std::vector<float> candidates = {.1, .2, .4, .8, 1.2, 1.5, 2};
        std::vector<float> testerrors;

        std::for_each(candidates.begin(), candidates.end(), [&](float sigma) {

            auto score = 0.0f;
            // do k-fold cross validation
            for (int k = 0; k < 5; k ++){
                // split Xp into Xp_train and Xp_test
                auto Xpk = vstack({Xp.rows(0, k*np/5), Xp.rows((k+1)*np/5, np)});
                auto Xpt_k = vstack({Xp.rows(k*np/5, (k+1)*np/5)});
                // split Xq into Xq_train and Xq_test
                auto Xqk = vstack({Xq.rows(0, k*nq/5), Xq.rows((k+1)*nq/5, nq)});
                auto Xqt_k = vstack({Xq.rows(k*nq/5, (k+1)*nq/5)});
                
                int nqk = Xqk.num_row();
                int npk = Xpk.num_row();
                int nqt_k = Xqt_k.num_row();
                int npt_k = Xpt_k.num_row();

				auto res = update_fdiv(Xpk, Xqk, vstack({ Xpt_k, Xqt_k }), DFcon, sigma, lambda_chosen, maxiter);

                auto fXpt_k = hstack({Xpt_k, Matrix<T>::ones(npt_k, 1)});
                auto fXqt_k = hstack({Xqt_k, Matrix<T>::ones(nqt_k, 1)});
                auto logrpt = sum(hadmd(res.rows(0,npt_k), fXpt_k), 1);
                auto logrqt = sum(hadmd(res.rows(npt_k,npt_k+nqt_k), fXqt_k), 1);

                auto t = - sum(logrpt, 0)/npt_k + sum(Fcon(std::move(logrqt)), 0)/nqt_k;

                score += item(t)/5;
            }

#ifdef DEBUG
            std::cout << "sigm:" << sigma << " " << "e: " <<  score << std::endl;
#endif
            testerrors.push_back(score);

        });

        int min_idx = argmin(testerrors);
        sigma_chosen = candidates[min_idx];
        
#ifdef DEBUG
        std::cout << "min_idx: " << min_idx << std::endl;
        std::cout << "sigma chosen: " << sigma_chosen << std::endl;
#else 
        std::cout << "sigma chosen: " << sigma_chosen << std::endl;
#endif

    }

    auto gradlogr  = update_fdiv(Xp, Xq, X, DFcon, sigma_chosen, lambda_chosen, maxiter);
// #ifdef DEBUG
//     std::cout << "gradients: " << gradlogr.slice(0, 5, 0, 5) << std::endl;
// #endif

    return result<T> {std::move(gradlogr), sigma_chosen};

}