%% Configuration
cdir = 'C:\Users\johnw\Documents\JHU\JHU_Misc';
cd(cdir);

%%
% distance
d = [1e6 10 20 5 18; 10 1e6 15 32 10; 20 15 1e6 25 16;
	5 32 25 1e6 35; 18 10 16 35 1e6];
cities = ['A', 'B', 'C', 'D', 'E'];
n = numel(cities);

p = [1, 0]; % parameter of sigmoidal membership function
T = 100;
it = 1; % counter for current iteration
maxIter = 1000; % maximum 1000 iterations
r = rand(maxIter, 3); % pre-gen rands for speed, need 3 rands per iteration

%% 

% randomized initial route
tmp = horzcat(rand(5,1), (1:5)'); % generate 5 random number, augment with 1-5
tmp = sortrows(tmp); % sort the rands, where 2nd col put in order of 1st col
x = tmp(:,2)'; % the initial randomly generated route;

while it < maxIter
	
	
	it = it + 1;
end

%% Brute forcing

% generate all possible routes
routes = perms(1:5);
routes(:,end+1) = routes(:,1); % force return to home

rteDist = NaN(length(routes),1);

for n = 1:length(rteDist) % loop over all possible routes
	a = routes(n, 1:end-1); % first index for distance matrix
	b = routes(n, 2:end);   % second index for distance matrix
	
	% calculate total distance on the n-th trip
	rteDist(n) = sum( d(sub2ind(size(d), a(:), b(:) ))  );
end

m = min(rteDist); % minimum distance of all routes
z = find(rteDist == m); % find all routes matching minimum distance
