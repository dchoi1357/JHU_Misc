function [wtOutput_chg, wtHidd_chg] = FFBPramp(x_i, d, wtHidd, wtOut)
global ramp eta % activation function and step size

x_j = ramp(wtHidd * [x_i; 1]); % concatenate bias input (always=1)
y = ramp(wtOut * [x_j; 1]); % concatenate bias input (always=1)

gamma_k = (d-y) * (1 - exp(-y));
% e = d-y; % for debugging
wtOutput_chg = eta .* gamma_k .* [x_j' 1]; 

gamma_j = (1 - exp(-x_j)) .* (gamma_k .* wtOut(1:end-1)'); % exclude bias wt
wtHidd_chg = (eta .* gamma_j) * [x_i' 1];