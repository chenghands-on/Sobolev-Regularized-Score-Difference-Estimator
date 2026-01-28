function [W,b] = GradEst_fdiv(Xp, Xq, X, psicon,  sigma, iter, K, master_stepsize, W, b)
[n, d] = size(X);
np = size(Xp, 1); nq = size(Xq, 1);

syms x real;
dpsicon = matlabFunction(diff(psicon(x)), "Vars", x);

if nargin < 7
    K = @(x,y, sigma) kernel_gau(comp_dist(x', y'), sigma);
end

Kp = K(X, Xp, sigma);
Kq = K(X, Xq, sigma);

if nargin < 8
    master_stepsize = .001; 
end

if nargin < 9
    W = zeros(n, d); b = zeros(n, 1);
end

% eta = .1;
% for i = 1:iter
% 
%     dwb_q = W*Xq' + b;
% 
%     gW = Kp * Xp/np - Kq .* dpsicon(dwb_q)*Xq/nq;
%     gb = Kp * ones(np, 1)/np - Kq .* dpsicon(dwb_q)*ones(nq, 1)/nq;
% 
% 
%     W = W + eta*gW;
%     b = b + eta*gb;
% 
%     norm(gW)
% end

%% AdaGrad with momentum
auto_corr = 0.9;
fudge_factor = 1e-6;
historial_grad = 0;

for iter = 1:iter
    theta = [W, b];

    dwb_q = W*Xq' + b;
    gW = Kp * Xp/np - Kq .* dpsicon(dwb_q)*Xq/nq;
    gb = Kp * ones(np, 1)/np - Kq .* dpsicon(dwb_q)*ones(nq, 1)/nq;

    grad = [gW, gb];
    if historial_grad == 0
        historial_grad = historial_grad + grad.^2;
    else
        historial_grad = auto_corr * historial_grad + (1 - auto_corr) * grad.^2;
    end
    adj_grad = grad ./ (fudge_factor + sqrt(historial_grad));
    theta = theta + master_stepsize * adj_grad; % update

    % norm(gW)

    W = theta(:, 1:d);
    b = theta(:, d+1);
end

end