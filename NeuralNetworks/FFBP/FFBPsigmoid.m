function [wtOuput_chg, wtHidd_chg] = FFBPsigmoid(x_i, d, wtHidd, wtOut)
% x_i = inputs; wtHidd = hidden layer weights; wtOut = output layer weights
% wtOutput_chg = chg in out layer weights; wtHidd_chg = chg in hidden layer wts
global sigm eta % activation function and step size

x_j = sigm(wtHidd * [x_i; 1]); % concatenate bias input (always=1)
y = sigm(wtOut * [x_j; 1]); % concatenate bias input (always=1)

delta_k = (d-y) * (1-y) * y; % gradient
wtOuput_chg = eta .* delta_k .* [x_j' 1];

delta_j = (1-x_j) .* x_j .* (delta_k .* wtOut(1:end-1)'); % exclude bias wt
wtHidd_chg = (eta .* delta_j) * [x_i' 1];