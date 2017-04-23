function E = EnergyAtPt(D, x, i, j, pen)
% there are three parts of energy of each perceptron
% 1. The distance contribution from previous epoch
% 2. Contribution of distance to next epoch
% 3. Penalties from other cities in same epoch and same cities in other epoch
% 4. Offset (equal to half penalty)

[E1, E2, E3, E4] = deal(0); % Set default at 0 for easy debugging

E1 = D(i,:)*x(:,j-1);
E2 = x(:,j+1)' * D(:,i);
E3 = ( sum(x(:,j)) + sum(x(i,:)) - 2.*x(i,j) ) * pen;
E4 = pen/2;

E = -(E1 + E2 + E3)/2 + E4;