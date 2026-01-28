function  [theta, traj, kl] = svgd(theta0, dlog_p, max_iter, master_stepsize, ada, xp)

%%%%%%%%
% Bayesian Inference via Stein Variational Gradient Descent

% input:
%   -- theta0: initialization of particles, m * d matrix (m is the number of particles, d is the dimension)
%   -- dlog_p: function handle of first order derivative of log p(x)
%   -- max_iter: maximum iterations
%   -- master_stepsize: the general learning rate for adagrad
%   -- h/bandwidth: bandwidth for rbf kernel. Using median trick as default
%   -- auto_corr: momentum term
%   -- method: use adagrad to select the best \epsilon

% output:
%   -- theta: a set of particles that approximates p(x)
%%%%%%%%

if nargin < 4; master_stepsize = 0.1; end;
if nargin < 5; ada = false; end;

% for the following parameters, we always use the default settings
h = -1; 
auto_corr = 0.9;
method = 'adagrad'; 

switch lower(method)
    
    case 'adagrad'
        %% AdaGrad with momentum
        theta = theta0;
        
        fudge_factor = 1e-6;
        historial_grad = 0;
        
        traj = [];
        kl = []; 
        for iter = 1:max_iter
            traj = [traj, theta];
            kl = [kl, compute_KL(theta', xp')];

            grad = KSD_KL_gradxy(theta, dlog_p, h);   %\Phi(theta)
            % grad = mysvgd(theta', dlog_p, true)';
            if historial_grad == 0
                historial_grad = historial_grad + grad.^2;
            else
                historial_grad = auto_corr * historial_grad + (1 - auto_corr) * grad.^2;
            end
            adj_grad = grad ./ (fudge_factor + sqrt(historial_grad));
            if ada
                theta = theta + master_stepsize * adj_grad; % update
            else
                theta = theta + master_stepsize * grad; % update
            end
        end
        
    otherwise
        error('wrong method');
end
end

% function [grad] = mysvgd(X, gradlogp, normalize)
% 
% % transpose, so that rows are obs and cols are features
% X = X';
% 
% % median trick. You can play with this, median /2, median *2 etc. 
% sigma = comp_med(X');
% 
% % compute kernel matrix
% Xb = X; % pick your basis. The classic SVGD uses all particles as basis
% D = comp_dist(X', Xb');
% n = size(X,1);
% nb = size(Xb,1);
% K = kernel_gau(D, sigma); % K is n \times nb
% 
% % evaluate gradient for each Xb
% glogp = [];
% for i = 1:size(Xb,1)
%     glogp(:, i) = gradlogp(Xb(i,:)');
% end
% 
% grad = zeros(n, size(X,2));
% for i = 1:size(X,2) 
%     glogp_K = repmat(glogp(i,:), n, 1) .* K;
% 
%     diff = (repmat(X(:,i),1,nb) - repmat(Xb(:,i)',n,1))/sigma^2;
% 
%     % SVGD = E_xb(nabla_xb logp K + nabla_xb K)
%     grad(:, i) = mean(glogp_K + diff.*K, 2);
% 
%     % normalize SVGD or not? 
%     if normalize
%         grad(:, i) = grad(:, i)./(sum(K,2)/n);
%     end
% end
% 
% % transpose back to row feature, col observation
% grad = grad';
% end


% %% KL compute
% function [kl]  = compute_KL(Xp, Xq)
% 
%     % halfidx = round(size(Xp, 2)/2);
%     % Xpt = Xp(:, 1:halfidx);
%     % Xp = Xp(:, halfidx+1:end);
%     % Xqt = Xq(:, 1:halfidx);
%     % Xq = Xq(:, halfidx+1:end);
% 
%     med = comp_med(Xp);
%     % med = .5;
%     b = 50;
%     Xb = Xp(:, 1:b);
%     dist2pp = comp_dist(Xb, Xp);
%     dist2pq = comp_dist(Xb, Xq);
% 
%     fp = kernel_gau(dist2pp, med);
%     fq = kernel_gau(dist2pq, med);
% 
%     % obj_loglinearKLIEP = @(theta) - mean(theta'*fp, 2) + log(mean(exp(theta'*fq), 2));
%     obj_loglinearKLIEP = @(theta) LLKLIEP(theta, fp, fq);
%     opts = optimset('fminunc');
%     opts.GradObj = 'on';
%     opts.MaxFunEvals = 100000;
%     % opts.DerivativeCheck = 'on';
% 
%     theta_hat = fminunc(obj_loglinearKLIEP, zeros(b, 1), opts);
%     logrmarks = theta_hat'*fp - log(mean(exp(theta_hat'*fq), 2));
% 
%     kl = mean(logrmarks);
% end