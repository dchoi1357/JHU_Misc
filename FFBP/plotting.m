figs = cell(3,1);

%% TACA by GI and LAC
taca1 = (raw.TACA==1) ;
figs{1} = figure('OuterPosition', [50 50 600 300]);
ptSize = 75;
hold on;
scatter(raw.GI(taca1), raw.LAC(taca1), ptSize, 'k', 'filled') ; % TACA=1 
scatter(raw.GI(~taca1) , raw.LAC(~taca1), ptSize, 'k') ; 
hold off;
ylim([ 0.80 3.2] ) ;
set(gca, 'Ytick', 1 : 3) ;
xlabel('Gross Income');
ylabel('Local Affluence Code') ;
legend({'TACA==1', 'TACA==0'}, 'location', 'Southeast'); 
title('Targeted Advertising Code Assignment by GI and LAC' );

%% SOW by GI and LAC
figs{2} = figure('OuterPosition', [50 50 600 300]);
plotColors = {'k', 'b', 'r'};
hold on;
for nn = 1:3
	indx = (raw.LAC==nn);
	scatter(raw. GI(indx), raw. SOW(indx), plotColors{nn}, 'filled');
end
hold off; 
legend( {'LAC==1', 'LAC==2', 'LAC==3'}, 'location', 'Southeast'); 
xlabel('Gross Income');
ylabel ('Size of Wallet'); 
title ('SOW by GI, separated by LAC');

%% Visualize activation functions
figs{3} = figure('OuterPosition', [50 50 600 300]);
subplot(1,2,1);
ezplot(sigm, [-6, 6], figs{3})
grid on;

subplot(1,2,2);
ezplot(ramp, [-5, 10], figs{3})
grid on;