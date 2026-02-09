% test_pinn.m - Test PINN forward pass
% Load weights
load('pinn_weights.mat');

% Create weights structure
weights.W1 = W1; weights.b1 = b1;
weights.W2 = W2; weights.b2 = b2;
weights.W3 = W3; weights.b3 = b3;
weights.W4 = W4; weights.b4 = b4;
weights.W5 = W5; weights.b5 = b5;

% Test prediction
C_A0_test = 0.85;  % Initial concentration
u_test = 0.5;      % Input flow rate
dt_test = 1.0;     % Time step

CA_next = pinn_forward(C_A0_test, u_test, dt_test, weights);

fprintf('Test Prediction:\n');
fprintf('  C_A0 = %.4f\n', C_A0_test);
fprintf('  u    = %.4f\n', u_test);
fprintf('  dt   = %.4f\n', dt_test);
fprintf('  CA_next (predicted) = %.4f\n', CA_next);
