import cvxpy as cp
import numpy as np
from scipy.linalg import solve_discrete_are

def compute_LQR_gain(A, B, Q, R, Cx=None):
    # Construct Q_z for the lifted state
    # If Q is scalar, we assume it penalizes the state x = Cx*z
    if Cx is not None:
        if np.isscalar(Q):
            # Q_z = Cx^T * Q * Cx
            Q_z = Q * np.outer(Cx, Cx)
        else:
            Q_z = Cx.T @ Q @ Cx
    else:
        # If no Cx provided, assume Q is already for z
        if np.isscalar(Q):
            Q_z = Q * np.eye(A.shape[0])
        else:
            Q_z = Q

    # Ensure R is 2D
    if np.isscalar(R):
        R_mat = np.array([[R]])
    else:
        R_mat = R

    # Solve DARE
    # Hack: Scale A slightly to move eigenvalues off unit circle if needed
    # Check eigenvalues
    evals = np.linalg.eigvals(A)
    if np.any(np.abs(np.abs(evals) - 1.0) < 1e-4):
        # print("Warning: Eigenvalues on unit circle. Scaling A for LQR design.")
        A_design = A * 0.995
    else:
        A_design = A

    P = solve_discrete_are(A_design, B, Q_z, R_mat)

    # Compute K
    # K = (R + B^T P B)^-1 B^T P A
    R_plus_BTPB = R_mat + B.T @ P @ B
    K = np.linalg.inv(R_plus_BTPB) @ (B.T @ P @ A)

    return K, P

class LinearMPC:
    def __init__(self, A, B, Q, R, P, N, x_min, x_max, u_min, u_max, v_min, v_max, Cx, Cu):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.P = P
        self.N = N
        self.x_min = x_min
        self.x_max = x_max
        self.u_min = u_min
        self.u_max = u_max
        self.v_min = v_min
        self.v_max = v_max
        self.Cx = Cx
        self.Cu = Cu

        self.nz = A.shape[0]
        self.nu = B.shape[1] if len(B.shape) > 1 else 1

        self.C_Ai_val = 1.0
        self.k_val = 0.028

    def solve(self, z0, x_sp):
        Z = cp.Variable((self.N + 1, self.nz))
        V = cp.Variable((self.N, self.nu))

        cost = 0
        constraints = [Z[0] == z0]

        # Calculate steady state for tracking
        # x_ss = x_sp
        # 0 = u(1-x) - kx -> u = kx/(1-x)
        if np.abs(self.C_Ai_val - x_sp) > 1e-6:
             u_ss = self.k_val * x_sp / (self.C_Ai_val - x_sp)
        else:
             u_ss = 0.0

        z_ss = np.zeros(self.nz)
        # z structure: [x, u, xu, 1, x2, u2]
        z_ss[0] = x_sp
        z_ss[1] = u_ss
        z_ss[2] = x_sp * u_ss
        z_ss[3] = 1.0
        if self.nz > 4:
            z_ss[4] = x_sp**2
            z_ss[5] = u_ss**2

        for k in range(self.N):
            # Stage cost: Q(x - x_sp)^2 + R v^2
            x_k = self.Cx @ Z[k]
            cost += self.Q * cp.sum_squares(x_k - x_sp) + self.R * cp.sum_squares(V[k])

            # Dynamics
            constraints.append(Z[k+1] == self.A @ Z[k] + self.B @ V[k])

            # Constraints
            u_k = self.Cu @ Z[k]

            constraints.append(u_k >= self.u_min)
            constraints.append(u_k <= self.u_max)
            constraints.append(x_k >= self.x_min)
            constraints.append(x_k <= self.x_max)

            constraints.append(V[k] >= self.v_min)
            constraints.append(V[k] <= self.v_max)

        # Terminal cost: (z_N - z_ss)^T P (z_N - z_ss)
        cost += cp.quad_form(Z[self.N] - z_ss, self.P)

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, verbose=False)

        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            # print(f"Warning: MPC Status {prob.status}")
            return None, None

        return V[0].value, Z.value
