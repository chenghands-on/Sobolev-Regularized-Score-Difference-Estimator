% anonymized code for ICML2026

clear
addpath("utils");
addpath("svgd\matlab");

%%
% Problem set up. 
% Random seeds.
seed = 123
rng(seed)
n = 500; % number of particles
d= 1;
%%

% grad log p, p = N(0, 1)
logp = @(x) log(mvnpdf(x', zeros(1,d), eye(d)))';
xp = 0 + randn(1,n);
x = sym('x', [d,1], 'real'); 
nabla_logp = matlabFunction(gradient(logp(x)),'var',{x});

% initial particles, from N(0,1)
xq = -1 + [randn(d,n)*.25];
x0 = xq;

xpbench = 0 + randn(1,500);

% SVGD
[xq, xq_traj1, kl1] = svgd(xq', nabla_logp, 50, .1, false, xpbench');
xq_traj1 = xq_traj1';

xq = x0;

% SVGD with AdaGrad.
[xq, xq_traj2, kl2] = svgd(xq', nabla_logp, 50, .1, true, xpbench');
xq_traj2 = xq_traj2';

xq = x0;

% NW
kl3 = [];
xq_traj3 = [];
for iter = 1:50
    kl3 = [kl3, compute_KL(xq, xpbench)];
    xq_traj3 = [xq_traj3; xq];
    [estgrad] = mysvgd(xq, nabla_logp, true); 
    % [estgrad] = mykde(xp, xq) - mykde(xq);
    xq = xq + .1*estgrad;

end

xq = x0;

% LL
kl4 = [];
xq_traj4 = [];
for iter = 1:50
    iter 

    kl4 = [kl4, compute_KL(xq, xpbench)];
    xq_traj4 = [xq_traj4; xq];
    med = comp_med(xq);
    [W,b] = GradEst_fdiv(xp', xq', xq', @(d) exp(d - 1), med, 2000);
    xq = xq + .1*W';

    % [estgrad,sigma] = MATLAB_Wrapper_GradEst_fdiv(single(xp'), single(xq'), single(xq'), med, .0, 2000);
    % xq = xq + .1*double(estgrad(:, 1)');
end

%%

h =  figure('Position', [100, 100, 800, 300]);
subplot(1,5,1)
hold on;
for i = 1:n
    h = plot(xq_traj1(:, i), 'r', 'LineWidth', 1);
    h.Color = [1, 0, 0, .1];
end
ymarks = linspace(-3,3, 1000);
pdfy = normpdf(ymarks, -1, .25) * 7;
h1 = line(pdfy, ymarks, 'Color', 'k', 'LineWidth', 2, 'LineStyle', '--');
pdfy = 50 - normpdf(ymarks, 0, 1) * 30;
h2 = line(pdfy, ymarks, 'Color', 'k', 'LineWidth', 2);
grid on;
xlabel('t')
htitle = title("SVGD");
htitle.FontSize = 12;
axis([0,50,-3,3])

subplot(1,5,2)
hold on;
for i = 1:n
    h = plot(xq_traj2(:, i), 'r', 'LineWidth', 1);
    h.Color = [1, 0, 0, .1];
end
ymarks = linspace(-3,3, 1000);
pdfy = normpdf(ymarks, -1, .25) * 7;
h1 = line(pdfy, ymarks, 'Color', 'k', 'LineWidth', 2, 'LineStyle', '--');
pdfy = 50 - normpdf(ymarks, 0, 1) * 30;
h2 = line(pdfy, ymarks, 'Color', 'k', 'LineWidth', 2);
grid on;
xlabel('t')
htitle = title("SVGD with Ada.");
htitle.FontSize = 12;
axis([0,50,-3,3])

subplot(1,5,3)
hold on;
for i = 1:n
    h = plot(xq_traj3(:, i), 'r', 'LineWidth', 1);
    h.Color = [1, 0, 0, .1];
end
ymarks = linspace(-3,3, 1000);
pdfy = normpdf(ymarks, -1, .25) * 7;
h1 = line(pdfy, ymarks, 'Color', 'k', 'LineWidth', 2, 'LineStyle', '--');
pdfy = 50 - normpdf(ymarks, 0, 1) * 30;
h2 = line(pdfy, ymarks, 'Color', 'k', 'LineWidth', 2);
grid on;
xlabel('t')
htitle = title("NW");
htitle.FontSize = 12;
axis([0,50,-3,3])


subplot(1,5,4)
hold on;
for i = 1:n
    h = plot(xq_traj4(:, i), 'r', 'LineWidth', 1);
    h.Color = [1, 0, 0, .1];
end
ymarks = linspace(-3,3, 1000);
pdfy = normpdf(ymarks, -1, .25) * 7;
h1 = line(pdfy, ymarks, 'Color', 'k', 'LineWidth', 2, 'LineStyle', '--');
pdfy = 50 - normpdf(ymarks, 0, 1) * 30;
h2 = line(pdfy, ymarks, 'Color', 'k', 'LineWidth', 2);
grid on;
xlabel('t')
htitle = title("LL");
htitle.FontSize = 12;
axis([0,50,-3,3])

hleg = legend([h1, h2], '$q_0$', '$p$');
hleg.Interpreter = "latex";
hleg.FontSize = 12;

subplot(1,5,5)
plot(kl1, 'g', 'LineWidth', 2, DisplayName="SVGD")
hold on;
plot(kl2, 'b', 'LineWidth', 2, DisplayName="SVGD with Ada.")
plot(max(kl3, 0), 'r-', 'LineWidth', 2, DisplayName="NW")
plot(max(kl4, 0), 'm-', 'LineWidth', 2, DisplayName="LL")
hleg = legend("toggle");
hleg.FontSize = 12;
axis([0,50,0,1.5])
xlabel("t");
htitle = title("$\mathrm{KL}[q_t, p]$");
htitle.Interpreter = "latex";
htitle.FontSize = 12;

saveas(h, 'demo_SVGD_NW.png');

