import numpy as np
import cvxpy as cp
import pandas as pd
from .lifting import KoopmanLifting

class KoopmanGenerator:
    def __init__(self, data_path):
        self.data_path = data_path
        self.lifting = KoopmanLifting()

        # System parameters
        self.k = 0.028
        self.C_Ai = 1.0

    def fit(self, lambda_reg=1e-5):
        # 1. Load data
        df = pd.read_csv(self.data_path)

        # X, U, V are from data
        X = df['x_k'].values
        U = df['u_k'].values
        V = df['v_k'].values

        n_samples = len(X)

        # 2. Compute lifted state Z
        Z = self.lifting.lift(X, U)

        # 3. Compute target derivatives Z_dot analytically (using known physics)
        # \dot x = u(C_Ai - x) - k x
        X_dot = U * (self.C_Ai - X) - self.k * X
        # \dot u = v
        U_dot = V
        # \dot(xu) = \dot x u + x \dot u
        XU_dot = X_dot * U + X * U_dot
        # \dot 1 = 0
        One_dot = np.zeros(n_samples)
        # \dot(x^2) = 2 x \dot x
        X2_dot = 2 * X * X_dot
        # \dot(u^2) = 2 u \dot u = 2 u v
        U2_dot = 2 * U * V

        # Order MUST MATCH lifting.feature_names
        # ['x', 'u', 'xu', '1', 'x2', 'u2']
        Z_dot = np.column_stack([X_dot, U_dot, XU_dot, One_dot, X2_dot, U2_dot])

        # 4. Formulate optimization problem
        n_z = self.lifting.n_z
        K = cp.Variable((n_z, n_z))
        L = cp.Variable((n_z, 1))

        # Prediction: Z @ K.T + V @ L.T
        prediction = Z @ K.T + V.reshape(-1, 1) @ L.T

        loss = cp.sum_squares(Z_dot - prediction) + lambda_reg * cp.sum_squares(K)

        constraints = []

        # Row 0 (x): \dot x = -k x + u - xu
        # Indices: x=0, u=1, xu=2, 1=3, x2=4, u2=5
        # K[0] = [-k, 1, -1, 0, 0, 0]
        constraints.append(K[0, 0] == -self.k)
        constraints.append(K[0, 1] == 1.0)
        constraints.append(K[0, 2] == -1.0)
        constraints.append(K[0, 3] == 0.0)
        constraints.append(K[0, 4] == 0.0)
        constraints.append(K[0, 5] == 0.0)
        constraints.append(L[0, 0] == 0.0)

        # Row 1 (u): \dot u = v
        constraints.append(K[1, :] == 0.0)
        constraints.append(L[1, 0] == 1.0)

        # Row 3 (1): \dot 1 = 0
        constraints.append(K[3, :] == 0.0)
        constraints.append(L[3, 0] == 0.0)

        # Rows 2 (xu), 4 (x2), 5 (u2) are free

        prob = cp.Problem(cp.Minimize(loss), constraints)
        prob.solve(solver=cp.OSQP, verbose=False)

        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print(f"Warning: Optimization status: {prob.status}")

        self.K_matrix = K.value
        self.L_matrix = L.value

        # Calculate residual on training set
        residual = Z_dot - (Z @ self.K_matrix.T + V.reshape(-1, 1) @ self.L_matrix.T)
        self.max_residual = np.max(np.linalg.norm(residual, axis=1))

        return self.K_matrix, self.L_matrix
