%% read data
% cdir = 'F:\Profiles\johnwu\Documents\JHU\NeuralNetwork\JHU_Misc\FFBP';
cdir = 'C:\Users\E3JXW01\Documents\MATLAB\FFBP';
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
sigm = @(x) 1 ./ (1 + exp(-x)); % Sigmoid activation function

eta = 0.5; % step size
n_hid = 5; % number of hidden layer nodes

GI_wt_range = 4 / ( (max(GI_range)-mean(GI_range))/10 * n_hid); 
LAC_wt_range = 4 / ( (max(LAC_range)-mean(LAC_range)) * n_hid);

%%
wtHidd = [ (rand(5,1)*2-1).*GI_wt_range  (rand(5,1)*2-1).*LAC_wt_range];
wtOuter = (rand(1,5)*2-1) * 4/5; % range = (-2,2) since there are two nodes

%%
nn = 1;
d = est.TACA(nn);
x_i = [est.GI_n(nn); est.LAC_n(nn)];

x_j = sigm(wtHidd * x_i);
y = sigm(wtOuter * x_j);

dlt_k = (d-y) * (1-y) * y;
wtOuter_chg = eta .* dlt_k .* x_j;

dlt_j = (1-x_j) .* x_j .* (dlt_k .* wtOuter');
wtHidd_chg = (eta .* dlt_j) * x_i';

%% Train TACA
for n = 1: height(est)
  

end
