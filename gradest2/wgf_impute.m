
function [x, xtraj, AUC] = wgf_impute(x, y, labels, missingrate, maxWGFiter, maxiter)
%%

x0 = x;

n = size(x,1);
d = size(x,2);

y2 = rand(n, d) > missingrate; % zero is missing
zp = [x, y];
zq = [x, y2];

figure;
hold on;
genError = [];
%%
% LL
xtraj = {};

cv = cvpartition(labels, 'HoldOut', 0.3);
idx = cv.test;

% W = zeros(n, d*2); b = zeros(n, 1);

for iter = 1:maxWGFiter
    xtraj{end+1} = x;

    Xtr = x(~idx,:);
    Ytr = labels(~idx);
    Xte = x(idx,:);
    Yte = labels(idx);

    svmModel = fitcsvm(Xtr, Ytr,"KernelFunction","linear");
    [~, score] = predict(svmModel, Xte);

    % compute AUC of the ROC
    [X,Y,T,AUC] = perfcurve(Yte, score(:, 2), 1);
    genError = [genError, AUC]

    d2 = sqrt(comp_dist(x', x')/2); d2 = d2(:);
    sigma(1) = median(d2);

    gzp = gpuArray(single(zp));
    gzq = gpuArray(single(zq));

    % sigma_list = logspace(-1,1.5,10)
    % val_error = [];
    % for i = 1:length(sigma_list)
    %     val_error(i) = cv(sigma_list(i), zp, zq)
    % end
    % [~, idx] = min(val_error);
    % fprintf("idx %d, sigma_chosen: %.3f\n", idx, sigma_list(idx))

    [W,b] = GradEst_fdiv(gzp, gzq, gzq, @(d) d.^2/2 + d, sigma, maxiter, ...
        @(x,y,sigma) kernel(x,y,sigma), 0.0001);

    grad = W(:, 1:d);

    fprintf("norm: %.2f\n", norm(x - x0));
    x(y==0) = x(y==0) + 1*double(gather(grad(y==0)));

    zp = [x, y];
    y2 = rand(n, d) > missingrate; % zero is missing
    zq = [x, y2];

    clf
    plot(genError)
    drawnow

end

% save('imputed_x.mat', 'x', 'x0', 'y', 'labels');

%%
% figure;
% hold on;
% % scatter(x0(sum(y,2)<d,1), x0(sum(y,2)<d,2), 'b.');
% % hold on;
% % scatter(x(sum(y,2)<d,1), x(sum(y,2)<d,2), 'ro');
% 
% for i = 1:n
%     xi_traj = [];
% 
%     for j = 1:iter
% 
%         xi_traj = [xi_traj; xtraj{j}(i, :)];
%     end
%     h = plot(xi_traj(:,1), xi_traj(:,2), 'r-');
%     h.Color(4) = 0.25;
% end

end

%%

function [K] = kernel(x, y, sigma)
d = size(x,2);
x1 = x(:, 1:(d/2));
y1 = y(:, 1:(d/2));
K1 = kernel_gau(comp_dist(x1', y1'), sigma(1));

x2 = x(:, (d/2)+1:end);
y2 = y(:, (d/2)+1:end);

K2 = x2*y2';

K = K1 .* K2;
end