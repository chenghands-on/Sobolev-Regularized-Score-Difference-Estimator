%% make sure you compile the CUDA C++ files using external/make.m

clear
rng default
addpath("external/matlab");

nq = 5000;

% perform subspace WGF?
findsubspace = true

if findsubspace
    U = orth(randn(5,2));
else
    U = eye(5);
end

B = orth(eye(5));
Xp = mvnrnd(zeros(1,5)+[0,0,0,0,0], ([1, .0, 0, 0, 0; ...
    .0, 1, 0, 0, 0; ...
    0, 0, 1, 0, 0; ...
    0, 0, 0, 1, .0; ...
    0, 0, 0, .0, 1]), nq);
Xp(:,1) = cos(Xp(:,2)*2) + randn(nq, 1) + 0;

Xp = [Xp, randn(nq, 0)];
Xp = Xp*B;

Xq = mvnrnd(zeros(1,5), ([1, .0, 0, 0, 0; ...
    .0, 1, 0, 0, 0; ...
    0, 0, 1, 0, 0; ...
    0, 0, 0, 1, 0; ...
    0, 0, 0, 0, 1]), nq);
Xq = [Xq, randn(nq, 0)];
Xq = Xq*B;



%%
figure
Xq_traj = {};

for Iter = 1:15
    Xpi = Xp;
    Xqi = Xq; 
    for i = 0:10
        f = @(x) x*U;

        fXpi = single(f(Xpi));  
        fXqi = single(f(Xqi));

        clf
        hold on
        scatter(fXqi(:, 1), fXqi(:, 2),'b.')
        scatter(fXpi(:, 1), fXpi(:, 2),'r.')
        axis([-6,6,-6,6])
        drawnow

        [grad,sigma] = MATLAB_Wrapper_GradEst_fdiv(fXpi, fXqi, [fXpi; fXqi], -1, 0, 250);

        if findsubspace
            fea = @(x) [f(x), ones(size(x,1),1)];
            gradp = grad(1:nq, :);
            gradq = grad(nq+1:end, :);
            wq = exp(sum(fea(Xqi).*gradq,2) - 1);
            gU = Xpi'*gradp(:, 1:2) / nq - Xqi'*(diag(wq))*gradq(:, 1:2)/nq;
    
            U = U + 1*gU;
    
            obj = @(V) norm(V - double(U), 'fro').^2;
            con = @(V) deal([], V'*V - eye(2));
    
            U = fmincon(obj, double(U), [], [], [], [], [], [], con);
            U(1:5, :), gU(1:5, :)
        else
            break;
        end
    end

    f = @(x) x*U;
    df = @(x) U';
    grad = MATLAB_Wrapper_GradEst_fdiv(f(single(Xpi)), f(single(Xqi)), f(single(Xq)), sigma, 0, 250);
    dimf = size(f(Xq),2);
    grad = grad(:, 1:dimf)*df(Xq);

    Xq = Xq + .1*grad;

    Xq_traj{end+1} = Xq;
end
%%
save(sprintf('subspace_%d', findsubspace));
%% visualizations
% make sure you run both findsubspace = true and findsubspace=false before 
% continue to the following section!

load subspace_0.mat

figure('Position', [100, 100, 1000, 300])
hold on
iter = 1;
for i = 1:3:15
    subplot(2,6,iter)
    h = scatter(Xq_traj{i}(:, 2), Xq_traj{i}(:, 1), 2, 'bo');
    h.MarkerFaceAlpha = .1;
    h.MarkerEdgeAlpha = .1;
    axis([-4,4,-4,4])
    axis tight;
    axis off
    iter = iter + 1;
end

subplot(2,6,iter)
h = scatter(Xp(:, 2), Xp(:, 1), 2, 'ko');
h.MarkerFaceAlpha = .1;
h.MarkerEdgeAlpha = .1;
axis tight;
axis off
iter = iter + 1;

load subspace_1.mat

for i = 1:3:15
    subplot(2,6,iter)
    h = scatter(Xq_traj{i}(:, 2), Xq_traj{i}(:, 1), 2, 'ro');
    h.MarkerFaceAlpha = .1;
    h.MarkerEdgeAlpha = .1;
    axis([-4,4,-4,4])
    axis tight;
    axis off
    iter = iter + 1;
end

subplot(2,6,iter)
h = scatter(Xp(:, 2), Xp(:, 1), 2, 'ko');
h.MarkerFaceAlpha = .1;
h.MarkerEdgeAlpha = .1;
axis tight;
axis off
