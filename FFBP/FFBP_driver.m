clc; clear;

%% read data
cdir = 'F:\Profiles\johnwu\Documents\JHU\NeuralNetwork\JHU_Misc\FFBP';
% cdir = 'C:\Users\E3JXW01\Documents\MATLAB\FFBP';
fl = 'final_data.csv';
cd(cdir);

raw = readtable( fullfile(cdir,fl) );
GI_range = [50 150]; %assume range of GI = 50K-150K
LAC_range = [1 3]; % range of LAC = 1-3

raw.GI_n = (raw.GI - min(GI_range)) ./ 10; % normalize input
raw.LAC_n = raw.LAC - mean(LAC_range); % "normalize" the input
% raw.LAC_n = raw.LAC ; % "normalize" the input
est = raw(1:2:end, :); % estimation set
test = raw(2:2:end, :); % testing set

%% Setup
global ramp sigm eta % Make function handle global variables
ramp = @(x) log( 1 + exp(x) ); % SoftPlus ramp activation function
sigm = @(x) 1 ./ (1 + exp(-x)); % Sigmoid activation function
eta = 1; % step size

% Calculate the range so that the results stay reasonable under Sigmoid
n_hid = 5; % number of hidden layer nodes
z = 8; % maximum weights to ensure sigmoid results are reasonable
GI_wt_range = z / ( (max(GI_range)-min(GI_range))/10 * n_hid); % range
LAC_wt_range = z / ( (max(LAC_range)-min(LAC_range)) * n_hid); % range

%% Random weights for TACA neural networks
wtHidd1 = [(rand(n_hid,1)*2-1).*GI_wt_range  (rand(n_hid,1)*2-1).*LAC_wt_range];
wtOuter1 = (rand(1,n_hid)*2-1) * z/n_hid;

%% Train TACA using Sigmoid
for z = 1:30
	for n = 1: height(est)
		d = est.TACA(n);
		x_i = [est.GI_n(n); est.LAC_n(n)];
		[chgOut, chgHidd] = FFBPsigmoid(x_i, d, wtHidd1, wtOuter1);
		
		wtHidd1 = wtHidd1 + chgHidd;
		wtOuter1 = wtOuter1 + chgOut;
	end
end

%% Evaluate TACA
test.TACA_y = nan(size(test.TACA)); % variable for evaluation results
test.TACA_ev = nan(size(test.TACA));
for n = 1: height(test)
	d = test.TACA(n);
	x_i = [test.GI_n(n); test.LAC_n(n)];
	
	test.TACA_y(n) = sigm(wtOuter1 * sigm(wtHidd1 * x_i));
	test.TACA_ev(n) = test.TACA_y(n) > 0.4;
end
% disp(test)