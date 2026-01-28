clear
addpath("external/matlab");

% anonymized code for ATSTATS2024

%%
% Problem set up. 
% Random seeds.
seed = 1
rng(seed)
n = 50; % number of particles
d= 1;
%%

% grad log p, p = N(0, .25^2)
logp = @(x) log(mvnpdf(x', zeros(1,d)+2, eye(d)/16))';
x = sym('x', [d,1], 'real'); 
nabla_logp = matlabFunction(gradient(logp(x)),'var',{x});

% initial particles, from N(0,1)
xq = [randn(d,n)];

% plot trajactory
xq_traj1 = [];
for iter = 1:50

    xq_traj1 = [xq_traj1; xq];
    [estgrad] = svgd(xq, nabla_logp, false); %SVGD
    xq = xq + .01*estgrad;
end

xq_traj2 = [];
xq = [randn(d,n)];
for iter = 1:50

    xq_traj2 = [xq_traj2; xq];
    [estgrad] = svgd(xq, nabla_logp, true); %NSVGD
    xq = xq + .01*estgrad;

end

h =  figure('Position', [100, 100, 800, 300]);
subplot(1,2,1)
hold on;
for i = 1:n
    h = plot(xq_traj1(:, i), 'r', 'LineWidth', 2);
    h.Color = [1, 0, 0, .2];
end
ymarks = linspace(-3,3, 1000);
pdfy = normpdf(ymarks, 0, 1) * 20;
h1 = line(pdfy, ymarks, 'Color', 'k', 'LineWidth', 2, 'LineStyle', '--');
pdfy = 50 - normpdf(ymarks, 2, .25) * 5;
h2 = line(pdfy, ymarks, 'Color', 'k', 'LineWidth', 2);
grid on;
xlabel('t')
title("SVGD")

subplot(1,2,2)
hold on;
for i = 1:n
    h = plot(xq_traj2(:, i), 'r', 'LineWidth', 2);
    h.Color = [1, 0, 0, .2];
end
ymarks = linspace(-3,3, 1000);
pdfy = normpdf(ymarks, 0, 1) * 20;
h1 = line(pdfy, ymarks, 'Color', 'k', 'LineWidth', 2, 'LineStyle', '--');
pdfy = 50 - normpdf(ymarks, 2, .25) * 5;
h2 = line(pdfy, ymarks, 'Color', 'k', 'LineWidth', 2);
grid on;
xlabel('t')
title("NSVGD")

hleg = legend([h1, h2], '$q_0$', '$p$');
saveas(h, 'demo_SVGD_NSVGD.png');

%% SVGD update

function [grad] = svgd(X, gradlogp, normalize)

% transpose, so that rows are obs and cols are features
X = X';

% median trick. You can play with this, median /2, median *2 etc. 
sigma = comp_med(X');

% compute kernel matrix
Xb = X; % pick your basis. The classic SVGD uses all particles as basis
D = comp_dist(X', Xb');
n = size(X,1);
nb = size(Xb,1);
K = kernel_gau(D, sigma); % K is n \times nb

% evaluate gradient for each Xb
glogp = [];
for i = 1:size(Xb,1)
    glogp(:, i) = gradlogp(Xb(i,:)');
end

grad = zeros(n, size(X,2));
for i = 1:size(X,2) 
    logp_K = repmat(glogp(i,:), n, 1) .* K;

    diff = (repmat(X(:,i),1,nb) - repmat(Xb(:,i)',n,1))/sigma^2;

    % SVGD = E_xb(nabla_xb logp K + nabla_xb K)
    grad(:, i) = mean(logp_K + diff.*K, 2);

    % normalize SVGD or not? 
    if normalize
        grad(:, i) = grad(:, i)./(sum(K,2)/n);
    end
end

% transpose back to row feature, col observation
grad = grad';
end
