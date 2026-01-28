%% KL compute
function [kl]  = compute_KL(Xp, Xq)
    
    % halfidx = round(size(Xp, 2)/2);
    % Xpt = Xp(:, 1:halfidx);
    % Xp = Xp(:, halfidx+1:end);
    % Xqt = Xq(:, 1:halfidx);
    % Xq = Xq(:, halfidx+1:end);
    
    med = comp_med(Xp);
    % med = .5;
    b = 100;
    Xb = Xp(:, 1:b);
    dist2pp = comp_dist(Xb, Xp);
    dist2pq = comp_dist(Xb, Xq);
    
    fp = kernel_gau(dist2pp, med);
    fq = kernel_gau(dist2pq, med);
    
    % obj_loglinearKLIEP = @(theta) - mean(theta'*fp, 2) + log(mean(exp(theta'*fq), 2));
    obj_loglinearKLIEP = @(theta) LLKLIEP(theta, fp, fq, .0001);
    opts = optimset('fminunc');
    opts.GradObj = 'on';
    opts.MaxFunEvals = 100000;
    opts.MaxIter = 1000000;
    % opts.DerivativeCheck = 'on';
    
    theta_hat = fminunc(obj_loglinearKLIEP, zeros(b, 1), opts);
    logrmarks = theta_hat'*fp - log(mean(exp(theta_hat'*fq), 2));

    kl = mean(logrmarks);
end

% %% KL compute
% function [kl]  = compute_KL(Xp, Xq)
% 
%     halfidx = round(size(Xp, 2)/2);
%     Xpt = Xp(:, 1:halfidx);
%     Xp = Xp(:, halfidx+1:end);
%     Xqt = Xq(:, 1:halfidx);
%     Xq = Xq(:, halfidx+1:end);
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
%     dist2pt = comp_dist(Xb, Xpt);
%     ft = kernel_gau(dist2pt, med);
%     dist2pqt = comp_dist(Xb, Xqt);
%     fqt = kernel_gau(dist2pqt, med);
%     logrmarks = theta_hat'*ft - log(mean(exp(theta_hat'*fqt), 2));
% 
%     kl = mean(logrmarks);
% end