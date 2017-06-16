function [t, roc] = EvalTACA(t, wtHidd, wtOut)
% t = test data; wtHidd = hidden layer weights; wtOut = output layer weights
global sigm binThresh % activation function
oneVec = ones(height(t),1); % vector of ones for biases

xi = sigm(wtHidd * [t.GI_n t.LAC_n oneVec]'); % hidden layer outputs 
t.TACA_y = sigm( wtOut * [xi; oneVec'] )'; % activation value of perceptrons
t.TACA_ev = ceil(t.TACA_y - binThresh); % on/off post threshold

% Calculate cells of confusion matrix
TP = t.TACA' * t.TACA_ev;
FN = t.TACA' * ~t.TACA_ev;
FP = ~t.TACA' * t.TACA_ev;
TN = sum(~t.TACA & ~t.TACA_ev); % STUPID MATLAB CANNOT MULTIPLY 2 LOGICAL ARRAYS

% Calculate various ROC metrics
roc.Sens = TP ./ (TP + FN);
roc.Spec = TN ./ (TN + FP);
roc.PPV = TP ./ (TP + FP);
roc.NPV = TN ./ (FN + TN);