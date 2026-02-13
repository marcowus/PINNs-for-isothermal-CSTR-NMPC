import numpy as np

class KoopmanLifting:
    def __init__(self):
        self.feature_names = ['x', 'u', 'xu', '1', 'x2', 'u2']
        self.n_z = len(self.feature_names)

    def lift(self, X, U):
        """
        Lift state x and input u to z space.
        Args:
            X: (N,) array of x (C_A)
            U: (N,) array of u
        Returns:
            Z: (N, n_z) array
        """
        X = np.atleast_1d(X).flatten()
        U = np.atleast_1d(U).flatten()

        if len(X) != len(U):
            raise ValueError("X and U must have same length")

        N = len(X)
        Z = np.zeros((N, self.n_z))

        # z = [x, u, xu, 1, x2, u2]
        Z[:, 0] = X
        Z[:, 1] = U
        Z[:, 2] = X * U
        Z[:, 3] = 1.0
        Z[:, 4] = X**2
        Z[:, 5] = U**2

        return Z

    def get_feature_indices(self):
        return {name: i for i, name in enumerate(self.feature_names)}

    def get_output_matrices(self):
        """
        Returns Cx and Cu such that x = Cx z, u = Cu z
        """
        Cx = np.zeros(self.n_z)
        Cu = np.zeros(self.n_z)

        # x is index 0
        Cx[0] = 1.0
        # u is index 1
        Cu[1] = 1.0

        return Cx, Cu
