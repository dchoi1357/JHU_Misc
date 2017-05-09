function [wtOuter_chg, wtHidd_chg] = FFBPramp(x_i, d, wtHidd, wtOuter)
global ramp eta

x_j = ramp(wtHidd * x_i);
y = ramp(wtOuter * x_j);

gamma_k = (d-y) * (1 - exp(-y));
wtOuter_chg = eta .* gamma_k .* x_j';

gamma_j = (1 - exp(-x_j)) .* (gamma_k .* wtOuter');
wtHidd_chg = (eta .* gamma_j) * x_i';