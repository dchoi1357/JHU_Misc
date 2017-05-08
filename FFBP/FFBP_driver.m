%% read data
cdir = 'F:\Profiles\johnwu\Documents\JHU\NeuralNetwork\JHU_Misc\FFBP';
fl = 'final_data.csv';
cd(cdir);

raw = readtable( fullfile(cdir,fl) );
GI_range = [50 150]; %assume range of GI = 50K-150K 
LAC_range = [1 3]; % range of LAC = 1-3

raw.GI_n = (raw.GI - mean(GI_range)) ./ 10; % normalize input
raw.LAC_n = raw.LAC - mean(LAC_range); % "normalize" the input
est = raw(1:2:end, :); % estimation set
test = raw(2:2:end, :); % testing set

%% Setup
ramp = @(x) log( 1 + exp(x) ); % SoftPlus ramp activation function
sigm = @(x) sigmf(x, [1,0]); % Sigmoid activation function

eta = 0.5; % step size
n_hid = 5; % number of hidden layer nodes

GI_wt_range = 4 / ( (max(GI_range)-mean(GI_range))/10 * n_hid); 
LAC_wt_range = 4 / ( (max(LAC_range)-mean(LAC_range)) * n_hid);

%%
wt_hid = [ (rand(5,1)*2-1).*GI_wt_range  (rand(5,1)*2-1).*LAC_wt_range];
wt_out = ;

%% Train TACA


for n = 1: height(est)
	x = 
	
end