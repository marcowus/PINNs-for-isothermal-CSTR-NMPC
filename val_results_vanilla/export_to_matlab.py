# export_to_matlab.py
import torch
import json
import numpy as np
import scipy.io as sio
import os

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
MODEL_PATH = 'train_results/trained_model.pth'
SAVE_DIR = 'matlab_export'
os.makedirs(SAVE_DIR, exist_ok=True)

# System constants 
C_Ai = 1.0
k = 0.028

# ────────────────────────────────────────────────
# MODEL DEFINITION 
# ────────────────────────────────────────────────
class PINN_CSTR(torch.nn.Module):
    def __init__(self, hidden_layers=4, neurons=64):
        super().__init__()
        layers = [torch.nn.Linear(3, neurons), torch.nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([torch.nn.Linear(neurons, neurons), torch.nn.Tanh()])
        layers.append(torch.nn.Linear(neurons, 1))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, C_A0, u, t):
        x = torch.cat([C_A0, u, t], dim=1)
        return self.network(x)

# ────────────────────────────────────────────────
# EXPORT FUNCTION
# ────────────────────────────────────────────────
def export_pinn_to_matlab():
    print("="*60)
    print("Exporting PINN Model to MATLAB Format")
    print("="*60)

    # Load trained model
    print("\n[1/3] Loading trained PINN model...")
    model = PINN_CSTR()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print("[OK] Model loaded successfully")

    # ──────────────────────────────────────────────
    # Extract weights and biases
    # ──────────────────────────────────────────────
    print("\n[2/3] Extracting network weights and biases...")
    
    weights = {}
    biases = {}
    layer_idx = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            layer_idx += 1
            weights[f'W{layer_idx}'] = param.detach().cpu().numpy()
            print(f"  W{layer_idx}: {param.shape}")
        elif 'bias' in name:
            biases[f'b{layer_idx}'] = param.detach().cpu().numpy()
            print(f"  b{layer_idx}: {param.shape}")

    # ──────────────────────────────────────────────
    # Prepare MATLAB structure
    # ──────────────────────────────────────────────
    print("\n[3/3] Preparing MATLAB .mat file...")
    
    matlab_data = {
        # Network architecture
        'num_layers': 5,  # 4 hidden + 1 output
        'input_dim': 3,   # [C_A0, u, dt]
        'hidden_dim': 64,
        'output_dim': 1,  # [C_A next]
        
        # Weights and biases for each layer
        'W1': weights['W1'],  # Input -> Hidden1
        'b1': biases['b1'],
        'W2': weights['W2'],  # Hidden1 -> Hidden2
        'b2': biases['b2'],
        'W3': weights['W3'],  # Hidden2 -> Hidden3
        'b3': biases['b3'],
        'W4': weights['W4'],  # Hidden3 -> Hidden4
        'b4': biases['b4'],
        'W5': weights['W5'],  # Hidden4 -> Output
        'b5': biases['b5'],
        
        # Activation function (for MATLAB implementation)
        'activation': 'tanh',  # String for reference
        
        # System constants
        'C_Ai': C_Ai,
        'k': k,
        
        # Input/output normalization info (if you add normalization later)
        'input_mean': np.array([0.0, 0.0, 0.0]),   # Placeholder
        'input_std': np.array([1.0, 1.0, 1.0]),    # Placeholder
        'output_mean': 0.0,
        'output_std': 1.0
    }
    
    # Save to .mat file
    mat_file = os.path.join(SAVE_DIR, 'pinn_weights.mat')
    sio.savemat(mat_file, matlab_data)
    print(f"[OK] Saved to: {mat_file}")

    # ──────────────────────────────────────────────
    # Save JSON metadata (for easy reading)
    # ──────────────────────────────────────────────
    metadata = {
        'architecture': {
            'layers': 5,
            'input_dim': 3,
            'hidden_dim': 64,
            'output_dim': 1,
            'activation': 'tanh'
        },
        'system_constants': {
            'C_Ai': float(C_Ai),
            'k': float(k)
        },
        'input_description': {
            'x1': 'C_A0 (current concentration)',
            'x2': 'u (input flow rate)',
            'x3': 'dt (time step)'
        },
        'output_description': {
            'y1': 'C_A_next (predicted concentration at next timestep)'
        },
        'matlab_usage': 'Load pinn_weights.mat and use forward_pass.m to evaluate the network'
    }
    
    json_file = os.path.join(SAVE_DIR, 'model_info.json')
    with open(json_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"[OK] Metadata saved to: {json_file}")

    # ──────────────────────────────────────────────
    # Create MATLAB forward pass script
    # ──────────────────────────────────────────────
    matlab_script = """function CA_next = pinn_forward(C_A0, u, dt, weights)
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
"""
    
    matlab_script_file = os.path.join(SAVE_DIR, 'pinn_forward.m')
    with open(matlab_script_file, 'w') as f:
        f.write(matlab_script)
    print(f"[OK] MATLAB script saved to: {matlab_script_file}")

    # ──────────────────────────────────────────────
    # Create test/example MATLAB script
    # ──────────────────────────────────────────────
    test_script = """% test_pinn.m - Test PINN forward pass
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

fprintf('Test Prediction:\\n');
fprintf('  C_A0 = %.4f\\n', C_A0_test);
fprintf('  u    = %.4f\\n', u_test);
fprintf('  dt   = %.4f\\n', dt_test);
fprintf('  CA_next (predicted) = %.4f\\n', CA_next);
"""
    
    test_script_file = os.path.join(SAVE_DIR, 'test_pinn.m')
    with open(test_script_file, 'w') as f:
        f.write(test_script)
    print(f"[OK] Test script saved to: {test_script_file}")

    # ──────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("EXPORT COMPLETE!")
    print("="*60)
    print(f"\nFiles created in '{SAVE_DIR}/':")
    print("  1. pinn_weights.mat      - Network weights (load in MATLAB)")
    print("  2. pinn_forward.m        - Forward pass function")
    print("  3. test_pinn.m           - Test script")
    print("  4. model_info.json       - Metadata and documentation")
    print("\n" + "="*60)
    print("\nNext steps for MATLAB NMPC:")
    print("  1. Copy files to your MATLAB working directory")
    print("  2. Run 'test_pinn.m' to verify the model works")
    print("  3. Use 'pinn_forward.m' in your NMPC optimization loop")
    print("="*60)

if __name__ == '__main__':
    export_pinn_to_matlab()