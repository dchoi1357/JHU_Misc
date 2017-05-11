%% TACA by GI and LAC
taca1 = (raw.TACA==1);
figure(1);
ptSize = 75;
hold on;
scatter(raw.GI(taca1), raw.LAC(taca1), ptSize, 'k', 'filled'); % TACA=1
scatter(raw.GI(~taca1), raw.LAC(~taca1), ptSize, 'k' );
hold off;
ylim([0.90 3.1]);
set(gca, 'Ytick', 1:3);
xlabel('Gross Income');
ylabel('Local Affluence Code');
legend({'TACA==1','TACA==0'}, 'location', 'Southeast');
title( 'Targeted Advertising Code Assignment by GI and LAC');

%% SOW by GI and LAC
figure(2);
plotColors = {'k', 'b', 'r'};
hold on;
for nn = 1:3
	indx = (raw.LAC==nn);
	scatter(raw.GI(indx), raw.SOW(indx), plotColors{nn}, 'filled');
end
hold off;
legend({'LAC==1','LAC==2','LAC==3'}, 'location', 'Southeast');
xlabel('Gross Income');
ylabel('Size of Wallet');
title('SOW by GI, separated by LAC');

%% clean up
close(1)
close(2);