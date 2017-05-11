clc; clear;

%% read data
cdir = 'F:\Profiles\johnwu\Documents\JHU\NeuralNetwork\JHU_Misc\FFBP';
% cdir = 'C:\Users\E3JXW01\Documenots\MATLAB\FFBP';
% cdir = 'C:\Users\E3JXW01\Desktop\FFBP';
fl = 'final_data.csv';
cd(cdir);

raw = readtable( fullfile(cdir,fl) );
GI_range = [50 150]; %assume range of GI = 50K-150K
LAC_range = [1 3]; % range of LAC = 1-3

raw.GI_n = (raw.GI - min(GI_range)) ./ 100; % normalize input
% raw.GI_n = log(raw.GI)/10; % normalize input

raw.LAC_n = (raw.LAC - mean(LAC_range)) .* 2; % "normalize" the input
% raw.LAC_n = raw.LAC ; % "normalize" the input
raw.SOW_n = log(raw.SOW)/4;

est = raw(1:2:end, :); % estimation set
est = repmat(est, 10, 1); % repeat data 10 times to train in batch of 10

test = raw(2:2:end, :); % testing set

%% Setup
global ramp sigm eta % Make function handle global variables
ramp = @(x) log( 1 + exp(x) ); % SoftPlus ramp activation function
sigm = @(x) 1 ./ (1 + exp(-x)); % Sigmoid activation function
rand1 = @(x,y) rand(x,y) .* 2 -1; 

% Calculate the range so that the results stay reasonable under Sigmoid
n_hid = 5; % number of hidden layer nodes
rn = 8; % maximum weights to ensure sigmoid results are reasonable
GI_wt_range = rn / ( (max(GI_range)-min(GI_range))/10 * n_hid); % range
LAC_wt_range = rn / ( (max(LAC_range)-min(LAC_range)) * n_hid); % range

%% Train TACA network using Sigmoid
eta = 2; % step size
wtHidd1 = [rand1(n_hid,1).*GI_wt_range, rand1(n_hid,2).*LAC_wt_range];
wtOuter1 = rand1(1,n_hid+1) * rn/n_hid;

for z = 1:30 % train at most 30x10 rounds
	for n = 1: height(est)
		d = est.TACA(n);
		x_i = [est.GI_n(n); est.LAC_n(n)];
		[chgOut, chgHidd] = FFBPsigmoid(x_i, d, wtHidd1, wtOuter1);
		
		wtHidd1 = wtHidd1 + chgHidd;
		wtOuter1 = wtOuter1 + chgOut;
	end
end

%% Evaluate TACA
test.TACA_y = nan(size(test.TACA)); % variable for sigmoid probabilities

for n = 1: height(test)
	x_i = [test.GI_n(n); test.LAC_n(n)];
	test.TACA_y(n) = sigm(wtOuter1 * [sigm(wtHidd1 * [x_i; 1]); 1]);
end
test.TACA_ev = ceil(test.TACA_y - 0.4); % transformed binary results
% disp(test)

%% Train SOW using SoftPlus ramp
eta = 0.1;
wtHidd2 = [rand(n_hid,1)/5 rand(n_hid,1)/5];
wtOuter2 = rand(1,n_hid)/5;

for z = 1:100
	for n = 1: height(est)
		d = est.SOW_n(n);
		x_i = [est.GI_n(n); est.LAC_n(n)];
		[chgOut, chgHidd] = FFBPramp(x_i, d, wtHidd2, wtOuter2);
		
		wtHidd2 = wtHidd2 + chgHidd;
		wtOuter2 = wtOuter2 + chgOut;
	end
end

%% Evaluate TACA
test.SOW_y = nan(size(test.SOW)); % variable for transformed test result

for n = 1: height(test)
	x_i = [test.GI_n(n); test.LAC_n(n)];
	test.SOW_y(n) = ramp(wtOuter2 * ramp(wtHidd2 * x_i));
end
test.SOW_ev = exp( test.SOW_y*4 ); % raw results
% disp(test);
