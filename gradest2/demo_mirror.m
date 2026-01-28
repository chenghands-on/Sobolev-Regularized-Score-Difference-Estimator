clear; 

syms r real; 
assume(r > 0);

phi = @(r) r*log(r)
% phi = @(r) -log(r)
% phi = @(r) 1/2/r - 1/2
% phi = @(r) 1/2*(r-1).^2
% phi = @(r) 1/2 * abs(r - 1)
% phi = @(r) (sqrt(r) - 1).^2/2
% phi = @(r) sqrt(r.^2 + 1) - sqrt(2)
% phi = @(r) r*log(r) - log(r)
% phi = @(r) 1/2*(r*log(r) - (r + 1)*log((r+1)/2));
psi_prime = r * diff(phi(r)) - phi(r);

psi = int(psi_prime);
psi = simplify(psi - subs(psi, r, 1));

pretty(psi)

figure; fplot(psi)
hold on; fplot(phi(r))