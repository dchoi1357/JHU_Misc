%% Configuration
% cdir = 'C:\Users\johnw\Documents\JHU\JHU_Misc';
cdir = 'F:\Profiles\johnwu\Documents\JHU\NeuralNetwork\TSP_Boltzmann';
cd(cdir);
clc; % clear screen;

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

%% Allocate state space and performance monitoring vairables
x = false(5,6); % pre-allocate matrix representing route-state
x(:,2:5) = rand(5,4) < 0.5; % randomly generate initial state
x(1,[1,6]) = true; % force first and last epoch to city A
xOrig = x;
minDist = 200; % storing the lowest energy reached, 200 > longest possible route
bestX = nan(5,6); % storing the best route state so far

distHist = minDist * ones(maxIter,1); % best dist history array

%%
for it = 1 : maxIter % iterate set number of times
	[i,j] = deal(randi(5), 1+randi(4)); % select coordinate of x to change
	
	% marginal chg in energy of turning on/off this perceptron
	E = (-x(i,j)*2 + 1) * EnergyAtPt(D, x, i, j, pen); % negate if currently on
	
	% accept if energy decrease or rand num < Metropolis Accep. Criterion
	if E < 0 || rand < exp(-E/T)
		x(i,j) = ~x(i,j);
		rej = false;
	else
		rej = true;
	end
	
	% following section is used to record the best distance seen so far only if
	% route discovered is valid and a newly discovered one
	if all(sum(x,1)==1) && all(sum(x(:,1:end-1),2)==1) && (~rej)
		cityIdx = (1:5) * x; % array of cities in route, eg. [1 3 2 4 5 1]
		a = cityIdx(1:end-1); % first index for distance matrix
		b = cityIdx(2:end);   % second index for distance matrix
		% calculate total distance on the n-th trip
		currDist = sum( D( sub2ind(size(D),a(:),b(:)) )  );
		tmpStr = num2cell(cities(cityIdx));
		fprintf('Route discovered: %s->%s->%s->%s->%s->%s, ', z{:} )
		fprintf('total dist: %u\n', currDist);
		if currDist < minDist % if distance better than best distance so far
			fprintf('Dist of route, %u, better than previous best, %u\n', ...
				currDist, minDist);
			fprintf('Recording as best route so far\n');
			bestX = x; % record the current route
			minDist = currDist; % record the current distance as best distance
			distHist(it:end) = minDist;
		else
			fprintf('Current dist not better than previous best, %u\n', minDist)
		end
	end
	T = T - dT; % decrease temperature
end
plot(1:maxIter, distHist) % plot history of best distance

%% Brute forcing (only used to check answers)

routes = perms(2:5); % generate all possible routes
routes = [ones(size(routes,1),1) routes ones(size(routes,1),1)];
rteDist = NaN(length(routes),1); % array for distance of all routes

for n = 1:length(rteDist) % loop over all possible routes
	a = routes(n, 1:end-1); % first index for distance matrix
	b = routes(n, 2:end);   % second index for distance matrix
	
	% calculate total distance on the n-th trip
	rteDist(n) = sum( d(sub2ind(size(d), a(:), b(:) ))  );
end

bruteMin = min(rteDist); % minimum distance of all routes
minRtes = find(rteDist == bruteMin); % find all routes matching minimum distance

if bruteMin == minDist
	fprintf('\nBest route from simulated annealing confirmed best by ');
	fprintf('brute force algorithm\n');
else
	fprintf('\nBest route from simulated annealing failed to converge to ');
	fprintf('best possible route\n');
end
