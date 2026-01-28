% anonymized code for ICML2026

clear
addpath("utils");
addpath("Stein-Variational-Gradient-Descent\matlab");

%%
res = {}
for missingrate = [.5, .6, .7, .8]
    parfor seed = 1:10

        % Random seeds.
        rng(seed)
        % n = 1000; % number of particles
        % d= 4;
        %%
        % D = load("iris.mat")
        D = load('breast_cancer.mat');
        % D = load('credit.mat')
        % random sample 1500 data points
        % idx = randperm(size(D.X,1));
        % idx = idx(1:1500);
        % D.X = D.X(idx,:);
        % D.y = D.y(idx);
        
        x = D.X;
        xtrue = x;
        labels = D.y;
        % labels = [ones(50, 1); zeros(100, 1)];

        n = size(x,1);
        d = size(x,2);

        y = rand(n, d) > missingrate; % zero is missing
        x(y == 0) = nan;
        mu = nanmean(x,1);
        sd = nanstd(x,1);
        x = x - mu;
        x = x ./ sd;

        noise = mvnrnd(zeros(1, d), eye(d), n);
        x(y == 0) = noise(y==0);

        x0 = x;
        [x, xqtraj] = wgf_impute(x,y,labels, missingrate, 1000, 100);

        res{seed} = x;

        parsave(sprintf('res/irir_imputed_%d_%.2f.mat', seed, missingrate), x, x0, y, labels);
    end
end


function parsave(fname, x, x0, y, labels)
  save(fname, 'x', 'x0', 'y', 'labels');
end