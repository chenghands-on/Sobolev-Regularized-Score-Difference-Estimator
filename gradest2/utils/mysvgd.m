%% SVGD update

function [grad] = mysvgd(X, gradlogp, normalize, gX, dgX)
if nargin < 4
    gX = ones(1, size(X, 2));
    dgX = zeros(size(X));
end

% transpose, so that rows are obs and cols are features
X = X';
gX = gX'; 
dgX = dgX'; 

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
    glogp_K = repmat(glogp(i,:), n, 1) .* K;

    diff = (repmat(X(:,i),1,nb) - repmat(Xb(:,i)',n,1))/sigma^2;

    % SVGD = E_xb(nabla_xb logp K + nabla_xb K)
    grad(:, i) = mean(gX'.*glogp_K + gX'.*diff.*K + dgX(:, i)'.*K, 2);

    % normalize SVGD or not? 
    if normalize
        grad(:, i) = grad(:, i)./mean(gX'.*K,2);
    end
end

% transpose back to row feature, col observation
grad = grad';
end
