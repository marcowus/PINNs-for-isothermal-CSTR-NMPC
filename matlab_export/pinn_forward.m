function CA_next = pinn_forward(C_A0, u, dt, weights)
% PINN_FORWARD - Forward pass through PINN network
%
% Inputs:
%   C_A0    - Current concentration
%   u       - Input flow rate
%   dt      - Time step
%   weights - Structure containing W1-W5, b1-b5
%
% Output:
%   CA_next - Predicted concentration at next timestep

    % Prepare input vector
    x = [C_A0; u; dt];
    
    % Layer 1: Input -> Hidden1
    z1 = weights.W1 * x + weights.b1;
    a1 = tanh(z1);
    
    % Layer 2: Hidden1 -> Hidden2
    z2 = weights.W2 * a1 + weights.b2;
    a2 = tanh(z2);
    
    % Layer 3: Hidden2 -> Hidden3
    z3 = weights.W3 * a2 + weights.b3;
    a3 = tanh(z3);
    
    % Layer 4: Hidden3 -> Hidden4
    z4 = weights.W4 * a3 + weights.b4;
    a4 = tanh(z4);
    
    % Layer 5: Hidden4 -> Output
    z5 = weights.W5 * a4 + weights.b5;
    CA_next = z5;  % Linear output (no activation)
    
end
