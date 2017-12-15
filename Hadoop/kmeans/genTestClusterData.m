N = 30;
c =  [0.8, 0.2, 0.2; 0.8, 0.8, 0.8; 0.2, 0.6, 0.3];
r = (rand(N, 3, 3) .* 0.5) - 0.25;

x = zeros(N*3, 3);

for n = 1:3
	x((n-1)*N+1 : n*N, :) = r(:,:,n) + ones(N,1) * c(n,:);
end

clr = ones(N,1) * [2:4];

%%
scatter3(x(:,1), x(:,2), x(:,3), [], clr(:), '*');
xlim([0, 1]);
ylim([0, 1]);
zlim([0, 1]);

%%
out = array2table(x);
out.name = arrayfun(@(x) sprintf('p%02u', x), 1:height(out), 'Uniform', 0)';
out = [out(:,end) out(:, 1:end-1)];

writetable(out, 'sample_points.csv', 'WriteVariableNames', 0)