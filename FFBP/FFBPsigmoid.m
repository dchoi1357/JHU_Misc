function [wtOuter_chg, wtHidd_chg] = FFBPsigmoid(x_i, d, wtHidd, wtOuter)
global sigm eta

x_j = sigm(wtHidd * [x_i; 1]); % concatenate bias input (always=1)
y = sigm(wtOuter * [x_j; 1]); % concatenate bias input (always=1)

delta_k = (d-y) * (1-y) * y;
wtOuter_chg = eta .* delta_k .* [x_j' 1];

delta_j = (1-x_j) .* x_j .* (delta_k .* wtOuter(1:5)');
wtHidd_chg = (eta .* delta_j) * [x_i' 1];