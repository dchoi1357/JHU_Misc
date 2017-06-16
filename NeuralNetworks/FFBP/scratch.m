%% Homework
d = 0.95;
eta = 0.1;
x = [1, 3];
w1 = [0.8, 0.1];
w2 = [0.5, 0.2];
w3 = [0.2, 0.7];


%% Outer layer
n1 = sigmf(x*w1', p);
n2 = sigmf(x*w2', p);

x_hid = [n1, n2];

n3 = sigmf(x_hid * w3', p);

dd = (d-n3)*(1-n3)*n3;

w3_old = w3;
w3 = w3 + eta*dd .* x_hid;


%% Hidden layers

dd_hid = (1-x_hid).*(x_hid) .* dd.*w3_old;
w1 = w1 + eta.*dd_hid(1) .* x;
w2 = w2 + eta.*dd_hid(2) .* x;