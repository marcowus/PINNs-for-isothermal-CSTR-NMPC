import numpy as np
import pandas as pd
from scipy.integrate import odeint

# Parameters
C_Ai = 1.0
k = 0.028

def coupled_dynamics(state, t, v_in):
    x_val, u_val = state
    # \dot x = u(C_Ai - x) - k x
    dxdt = u_val * (C_Ai - x_val) - k * x_val
    # \dot u = v
    dudt = v_in
    return [dxdt, dudt]

def simulate_data():
    # Simulation settings
    dt = 0.1  # Sampling time (seconds)
    n_trajectories = 200
    steps_per_traj = 100

    data_list = []

    print(f"Generating training data: {n_trajectories} trajectories...")

    for i in range(n_trajectories):
        # Random initial state
        x0 = np.random.uniform(0.0, 1.0)
        u0 = np.random.uniform(0.0, 2.0)

        current_x = x0
        current_u = u0

        for step in range(steps_per_traj):
            # Random control increment v
            # u is dilution rate, usually positive. Let's keep it in [0, 3] roughly
            v = np.random.uniform(-0.05, 0.05)

            # Boundary avoidance for u
            if current_u > 2.5:
                v = np.random.uniform(-0.05, 0.0)
            elif current_u < 0.1:
                v = np.random.uniform(0.0, 0.05)

            # Simulate one step (Zero Order Hold on v)
            t_span = [0, dt]
            sol = odeint(coupled_dynamics, [current_x, current_u], t_span, args=(v,))

            next_x, next_u = sol[-1]

            data_list.append({
                'traj_id': i,
                'x_k': current_x,
                'u_k': current_u,
                'v_k': v,
                'x_next': next_x,
                'u_next': next_u,
                'dt': dt
            })

            current_x = next_x
            current_u = next_u

    df_train = pd.DataFrame(data_list)
    df_train.to_csv('koopman_data.csv', index=False)
    print(f"Saved 'koopman_data.csv' with {len(df_train)} samples.")

    # 2. Test Data (A single long trajectory for comparison)
    print("Generating test data 'cstr_simulation_data.csv'...")
    test_data = []
    current_x = 0.5
    current_u = 1.0
    t = 0.0

    # Generate 500 steps
    for step in range(500):
        # Varying input to excite dynamics
        # u(t) = 1.0 + 0.5 * sin(0.05 * t)
        # We drive u by setting v appropriately
        target_u = 1.0 + 0.5 * np.sin(0.05 * t)

        # Proportional control to follow target_u profile, or just derivative
        # v ~ d(target_u)/dt
        # d/dt (1 + 0.5 sin(w t)) = 0.5 w cos(w t)
        v = 0.5 * 0.05 * np.cos(0.05 * t)

        # Add some noise to v?
        # v += np.random.normal(0, 0.001)

        t_span = [0, dt]
        sol = odeint(coupled_dynamics, [current_x, current_u], t_span, args=(v,))
        next_x, next_u = sol[-1]

        test_data.append({
            'time': t,
            'u_input': current_u,
            'CA_concentration': current_x,
            'v_input': v
        })

        current_x = next_x
        current_u = next_u
        t += dt

    df_test = pd.DataFrame(test_data)
    df_test.to_csv('cstr_simulation_data.csv', index=False)
    print("Saved 'cstr_simulation_data.csv'.")

if __name__ == "__main__":
    simulate_data()
