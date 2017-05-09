function [wtOuter_chg, wtHidd_chg] = FFBPsigmoid(x_i, d, wtHidd, wtOuter)
global sigm eta

x_j = sigm(wtHidd * x_i);
y = sigm(wtOuter * x_j);

delta_k = (d-y) * (1-y) * y;
wtOuter_chg = eta .* delta_k .* x_j';

delta_j = (1-x_j) .* x_j .* (delta_k .* wtOuter');
wtHidd_chg = (eta .* delta_j) * x_i';