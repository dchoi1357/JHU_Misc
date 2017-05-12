function [t, RMSE] = EvalSOW(t, wtHidd, wtOut)
% t = test data; wtHidd = hidden layer weights; wtOut = output layer weights
global ramp % global vars
oneVec = ones(height(t),1); % vector of ones for biases

xi = ramp(wtHidd * [t.GI_n t.LAC_n oneVec]'); % hidden layer outputs 
t.SOW_y = ramp( wtOut * [xi; oneVec'] )'; % activation value of perceptrons
t.SOW_ev = exp( t.SOW_y*4 ); % on/off post threshold

RMSE = sqrt(  sum((t.SOW_ev-t.SOW).^2) / height(t)  ); % root mean squared error