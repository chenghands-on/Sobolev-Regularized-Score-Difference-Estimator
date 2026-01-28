function [l,g] = LLKLIEP(theta,kP,kQ, lambda)
    l = - mean(theta'*kP, 2) + log(mean(exp(theta'*kQ), 2)) + lambda*theta'*theta;
    N_q = sum(exp(theta'*kQ),2);
    g_q = exp(theta'*kQ)./ N_q;
    g = -mean(kP,2) + sum(kQ .* g_q, 2) + 2*lambda*theta;
end
% l = -mean(tensorprod(theta, kP, 1),2) + log(mean(exp(tensorprod(theta, kQ, 1)),2));
% 
% N_q = sum(exp(tensorprod(theta, kQ, 1)),2);
% g_q = exp(tensorprod(theta, kQ, 1))./ N_q;
% g = -mean(kP,2) + sum(kQ .* g_q, 2);
% 
% l = mean(l, 3);
% g = mean(g, 3);
% end
% 
% function [M] = tensorprod(M1, M2, dim)
%     M = sum(M1 .* M2, dim);
% end