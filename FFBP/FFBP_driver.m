clc; clear;

%% read data
cdir = 'F:\Profiles\johnwu\Documents\JHU\NeuralNetwork\JHU_Misc\FFBP';
fl = 'final_data.csv';
cd(cdir);

raw = readtable( fullfile(cdir,fl) ); % read data
GI_range = [50 150]; %assume range of GI = 50K-150K
LAC_range = [1 3]; % range of LAC = 1-3

% Numerical inputs and outputs are rescaled to fit better in ANN frameowrk
raw.GI_n = (raw.GI - min(GI_range)) ./ 100;
raw.LAC_n = (raw.LAC - mean(LAC_range)) .* 2;
raw.SOW_n = log(raw.SOW)/4;

est = raw(1:2:end, :); % estimation set
est = repmat(est, 10, 1); % repeat data 10 times to train in batch of 10
test = raw(2:2:end, :); % testing set

%% Setup
global ramp sigm eta binThresh; % Make function handle global variables
ramp = @(x) log( 1 + exp(x) ); % SoftPlus ramp activation function
sigm = @(x) 1 ./ (1 + exp(-x)); % Sigmoid activation function
rand1 = @(x,y) rand(x,y) .* 2 -1; % func to generate random uniform in (-1, 1)
binThresh = 0.5; % threshold of "on" for predictive nodes

% Calculate the range so that the results stay reasonable under Sigmoid
n_hid = 5; % number of hidden layer nodes
rn = 8; % maximum weights to ensure sigmoid results are reasonable
GI_wt_range = rn / ( (max(GI_range)-min(GI_range))/10 * n_hid); % range of GI
LAC_wt_range = rn / ( (max(LAC_range)-min(LAC_range)) * n_hid); % range of LAC

%% Train ANN to predit TACA and evaluate performance
eta = 2; % step size
% randomly generate starting weights based on previously established range
wtHidd1 = [rand1(n_hid,1).*GI_wt_range, rand1(n_hid,2).*LAC_wt_range];
wtOut1 = rand1(1,n_hid+1) * rn/n_hid;

ROChist = Inf(2,10); % record ROC metrics history, both sens and spec
for z = 1:30 % train at most 30x10 rounds
	for n = 1: height(est)
		d = est.TACA(n);
		x_i = [est.GI_n(n); est.LAC_n(n)];
		[chgOut, chgHidd] = FFBPsigmoid(x_i, d, wtHidd1, wtOut1);
		
		wtHidd1 = wtHidd1 + chgHidd;
		wtOut1 = wtOut1 + chgOut;
	end
	
	[test, roc] = EvalTACA(test, wtHidd1, wtOut1); %% Evaluate TACA
	ROChist = [[roc.Sens; roc.Spec] ROChist(:,1:end-1)]; % add new sens and spec
	if sum( diff(ROChist,1,2) > 0) > 2
		break % stop if ROC stats decreased more than twice in last 5 trainings
	end
end

%% Train ANN to predit SOW and evaluate performance
eta = 0.1; % step size
% randomly generate starting weights based on previously established range
wtHidd2 = [rand1(n_hid,1).*GI_wt_range, rand1(n_hid,2).*LAC_wt_range];
wtOut2 = rand1(1,n_hid+1) * rn/n_hid;

RMSEhist = Inf(1,5); % record RMSE history
for z = 1:100 % train maximum of 100x10 rounds
	for n = 1: height(est)
		d = est.SOW_n(n);
		x_i = [est.GI_n(n); est.LAC_n(n)]; % inputs
		[chgOut, chgHidd] = FFBPramp(x_i, d, wtHidd2, wtOut2);
		
		wtHidd2 = wtHidd2 + chgHidd;
		wtOut2 = wtOut2 + chgOut;
	end
	
	[test, RMSE] = EvalSOW(test, wtHidd2, wtOut2); % Evaluate TACA
	RMSEhist = [RMSE RMSEhist(1:end-1)]; % add new RMSE to history
	if sum( diff(RMSEhist) < 0) > 1 
		break % stop if RMSE increased more than twice in last 5 trainings
	end
end
