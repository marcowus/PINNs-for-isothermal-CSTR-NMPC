import numpy as np
from .mpc_qp import LinearMPC, compute_LQR_gain

class TubeMPC:
    def __init__(self, A, B, Q, R, N, x_min, x_max, u_min, u_max, v_min, v_max, Cx, Cu, w_bar):
        """
        w_bar: bound on disturbance |w| <= w_bar
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.N = N
        self.x_min = x_min
        self.x_max = x_max
        self.u_min = u_min
        self.u_max = u_max
        self.v_min = v_min
        self.v_max = v_max
        self.Cx = Cx
        self.Cu = Cu
        self.w_bar = w_bar

        # 1. Compute feedback gain K_fb for error system (using same LQR as nominal?)
        self.K_fb, self.P_nominal = compute_LQR_gain(A, B, Q, R, Cx=Cx)

        # 2. Compute rho = spectral radius of (A - B K_fb)
        # Note: LQR u = -K x. So A_cl = A - B K
        self.A_cl = A - B @ self.K_fb
        evals = np.linalg.eigvals(self.A_cl)

        # Filter out eigenvalues close to 1.0 (constant mode)
        rho_candidates = np.abs(evals)
        mask_unstable_constant = (rho_candidates > 0.999) & (rho_candidates < 1.001)

        if np.any(mask_unstable_constant):
            filtered_evals = rho_candidates[~mask_unstable_constant]
            if len(filtered_evals) > 0:
                self.rho = np.max(filtered_evals)
            else:
                self.rho = 0.0
        else:
            self.rho = np.max(rho_candidates)

        if self.rho >= 1.0:
            print(f"Warning: Closed loop system unstable! rho={self.rho}")
            self.rho = 0.99 # Fallback to avoid division by zero if unstable (though LQR should stabilize)

        # 3. Compute invariant set size (scalar bound)
        # |e| <= w_bar / (1 - rho)
        if self.rho < 0.999:
            self.epsilon_max = w_bar / (1.0 - self.rho)
        else:
            self.epsilon_max = w_bar * 100 # Large value

        # 4. Compute margins
        # Margin for x: |Cx e| <= |Cx| |e|
        self.margin_x = np.linalg.norm(Cx, 2) * self.epsilon_max

        # Margin for u: |Cu e| <= |Cu| |e|
        self.margin_u = np.linalg.norm(Cu, 2) * self.epsilon_max

        # Margin for v: |K_fb e| <= |K_fb| |e|
        # v_error = - K_fb e
        self.margin_v = np.linalg.norm(self.K_fb, 2) * self.epsilon_max

        print(f"Tube MPC Initialized: rho={self.rho:.4f}, eps={self.epsilon_max:.4f}")
        print(f"Margins: x={self.margin_x:.4f}, u={self.margin_u:.4f}, v={self.margin_v:.4f}")

        # 5. Initialize Nominal MPC with tightened constraints
        self.nominal_mpc = LinearMPC(
            A, B, Q, R, self.P_nominal, N,
            x_min + self.margin_x, x_max - self.margin_x,
            u_min + self.margin_u, u_max - self.margin_u,
            v_min + self.margin_v, v_max - self.margin_v,
            Cx, Cu
        )

        self.z_nom = None # State of nominal system

    def solve(self, z_current, x_sp):
        """
        Solve Tube MPC.
        z_current: actual measured state
        """
        # Initialize z_nom if first run
        if self.z_nom is None:
            self.z_nom = z_current.copy()

        # Solve nominal MPC for z_nom
        v_nom_seq, z_nom_seq = self.nominal_mpc.solve(self.z_nom, x_sp)

        if v_nom_seq is None:
            print("Tube MPC: Nominal MPC failed (Infeasible?).")
            return None, None

        # Compute actual control input
        # v = v_nom - K_fb (z_current - z_nom)
        v_nom = v_nom_seq[0] if v_nom_seq.ndim > 0 else v_nom_seq

        # v_control needs to be shape (1,) or scalar
        v_control = v_nom - self.K_fb @ (z_current - self.z_nom)

        # Update nominal state for next step
        # z_nom_next = A z_nom + B v_nom
        # We can use the prediction from MPC: z_nom_seq[1]
        self.z_nom = z_nom_seq[1]

        return v_control, z_nom_seq
