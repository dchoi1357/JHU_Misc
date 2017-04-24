%% Configuration
% cdir = 'C:\Users\johnw\Documents\JHU\JHU_Misc';
cdir = 'F:\Profiles\johnwu\Documents\JHU\NeuralNetwork\TSP_Boltzmann';
cd(cdir);


%% Initialize parameters
cities = ['A', 'B', 'C', 'D', 'E']; % text for print-visualizaion
pen = 100; % penalty distance
% distance: diagonal is 0 because penalty imposed through other ways
D = [0 10 20 5 18; 10 0 15 32 10; 20 15 0 25 16;
	5 32 25 0 35; 18 10 16 35 0]; 
% n = numel(cities);

pm = [1, 0]; % parameter of sigmoidal membership function
T = 101; % starting temperature
maxIter = 1000; % maximum 1000 iterations
dT = (T-1)/maxIter; % change in temperature every iteration

nRej = 0; % number of consecutive rejects in a row;
maxRej = 10; % max number of rejects in a row before algorithm stops

%%
x = false(5,6); % pre-allocate matrix representing route-state
x(:,2:5) = rand(5,4) < 0.5; % randomly generate initial state
x(1,[1,6]) = true; % force first and last epoch to city A
xOrig = x;

%%
for it = 1 : maxIter % iterate set number of times
	[i,j] = deal(randi(5), 1+randi(4)); % select coordinate of x to change
	
	% marginal chg in energy of turning on/off this perceptron
	E = (-x(i,j)*2 + 1) * -EnergyAtPt(D, x, i, j, pen); % negate if currently on
	
	if E < 0 % accept if chg in energy less than 0
		x(i,j) = ~x(i,j);
	else
		x(i,j) = xor( rand<exp(-E/T), x(i,j) ); % flip the state if prob < MAC
	end
	
	T = T - dT; % decrease temperature
end

disp( [x; nan(1,6); xOrig] );

%% Brute forcing
%{
% generate all possible routes
routes = perms(2:5);
routes = [ones(size(routes,1),1) routes ones(size(routes,1),1)];

rteDist = NaN(length(routes),1);

for n = 1:length(rteDist) % loop over all possible routes
	a = routes(n, 1:end-1); % first index for distance matrix
	b = routes(n, 2:end);   % second index for distance matrix
	
	% calculate total distance on the n-th trip
	rteDist(n) = sum( d(sub2ind(size(d), a(:), b(:) ))  );
end

m = min(rteDist); % minimum distance of all routes
z = find(rteDist == m); % find all routes matching minimum distance
%}