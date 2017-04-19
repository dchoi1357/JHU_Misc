% distance
d = [1e6 10 20 5 18; 10 1e6 15 32 10; 20 15 1e6 25 16;
	5 32 25 1e6 35; 18 10 16 35 1e6];
cities = ['A', 'B', 'C', 'D', 'E'];
n = numel(cities);

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