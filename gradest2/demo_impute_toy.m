% anonymized code for ICML2026

clear
addpath("utils");

%%
% Problem set up.
% Random seeds.
seed = 342342
rng(seed)
% n = 1000; % number of particles
% d= 4;
%%
D = load("s_curve.mat");
x = D.X;
xtrue = x;

n = size(x,1);
d = size(x,2);
% x0 = x;
% 
y = rand(n, d) > .3; % zero is missing 
x(y == 0) = nan;
mu = nanmean(x,1);
sd = nanstd(x,1);
x = x - mu;
x = x ./ sd;

noise = mvnrnd(zeros(1, d), eye(d), n);
x(y == 0) = noise(y==0);

x0 = x;

%%
% load S.mat
n = size(x,1);
d = size(x,2);

rng(4321)
y2 = rand(n, d) > .3; % zero is missing
zp = [x, y];
zq = [x, y2];


figure; 
hold on;
h = scatter(x(sum(y,2) < 2,1), x(sum(y,2) < 2,2), 'r.'); 
% scatter(zp(sum(y,2)==1,1), zp(sum(y,2)==1,2), 'r.'); 
scatter(x(sum(y,2)==2,1), x(sum(y,2)==2,2), 'b.'); 

%%
% LL
xq_traj4 = {};

for iter = 1:100
    xq_traj4{end+1} = x;

    d2 = sqrt(comp_dist(x', x')/2); d2 = d2(:);
    sigma(1) = median(d2)/3;

    gzp = gpuArray(single(zp));
    gzq = gpuArray(single(zq));

    sigma_list = logspace(-1,1.5,10)
    val_error = [];
    parfor i = 1:length(sigma_list)
        val_error(i) = cv(sigma_list(i), zp, zq)
    end
    [~, idx] = min(val_error);
    fprintf("idx %d, sigma_chosen: %.3f\n", idx, sigma_list(idx))

    [W,b] = GradEst_fdiv(gzp, gzq, gzq, @(d) d.^2/2 + d, sigma_list(idx), 2000, ...
                            @(x,y,sigma) kernel(x,y,sigma));
    
    grad = W(:, 1:d);
    x(y==0) = x(y==0) + .25*double(gather(grad(y==0)));
    zp = [x, y];
    y2 = rand(n, d) > .3; % zero is missing
    zq = [x, y2];

    clf
    scatter(zp(sum(y,2)==d,1), zp(sum(y,2)==d,2), 'r.'); 
    hold on;
    scatter(zp(sum(y,2)<d,1), zp(sum(y,2)<d,2), 'b.'); 
    % axis([-2,2,-3,3])
    drawnow

end

%% 
figure; 
hold on;
% scatter(x0(sum(y,2)<d,1), x0(sum(y,2)<d,2), 'b.');
% hold on;
% scatter(x(sum(y,2)<d,1), x(sum(y,2)<d,2), 'ro');



for i = 1:n
    xi_traj = [];

    for j = 1:iter

        xi_traj = [xi_traj; xq_traj4{j}(i, :)];
    end
    h = plot(xi_traj(:,1), xi_traj(:,2), 'r-');
    h.Color(4) = 0.25;
end

%%
xq = xq_traj4{1};
figure; 
hold on;
h = scatter(xq(sum(y,2) < 2,1), xq(sum(y,2) < 2,2), 'r.'); 
scatter(xq(sum(y,2)==2,1), xq(sum(y,2)==2,2), 'b.'); 
axis([-3, 3, -3, 3])

xq = xq_traj4{end};
figure; 
hold on;
h = scatter(xq(sum(y,2) < 2,1), xq(sum(y,2) < 2,2), 'r.'); 
scatter(xq(sum(y,2)==2,1), xq(sum(y,2)==2,2), 'b.'); 
axis([-3, 3, -3, 3])
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