function [k] = solve_k(S,lambda)

if lambda~=0
sigma1 = reshape(diag(S),1,[]);
sigma2 = sigma1.^2;
sigma3 = sigma2(end:-1:1);
sigma4 = cumsum(sigma3);
sigma5 = sigma4(end:-1:1);

r_max = length(sigma1);
candidate_r = 1:r_max;


[v k]= min(candidate_r(2:end) + lambda * sigma5(2:end));
k = min(k + 1, r_max);
fprintf(' | lambda = %e, the optimal k = %d. \n', lambda, k);
else
   k = rank(S); 
end
