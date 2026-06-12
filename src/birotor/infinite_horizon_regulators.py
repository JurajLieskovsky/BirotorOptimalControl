import scipy
import numpy as np
import cvxpy as cp

class _LQR:
    def __init__(self, f, df, x_eq, u_eq, q, r):
        self.x_eq = x_eq
        self.u_eq = u_eq

        self.q = q
        self.r = r

        assert all(abs(x_eq - f(0, x_eq, u_eq)) <= 1e-8)

        self.A, self.B = df(0, x_eq, u_eq)

        self.P = scipy.linalg.solve_discrete_are(self.A, self.B, self.q, self.r)

    def running_cost(self, x, u, _):
        dx = x - self.x_eq
        du = u - self.u_eq

        return dx.T @ self.q @ dx + du.T @ self.r @ du

    def final_cost(self, x):
        dx = x - self.x_eq

        return dx.T @ self.P @ dx


class LQR(_LQR):
    def __init__(self, f, df, x_eq, u_eq, q, r):
        super().__init__(f, df, x_eq, u_eq, q, r)

        self.K = np.linalg.solve(
            r + self.B.T @ self.P @ self.B, self.B.T @ self.P @ self.A
        )

    def input(self, x, _):
        return self.u_eq - self.K @ (x - self.x_eq)


class MPC(_LQR):
    def __init__(
        self,
        f,
        df,
        x_eq,
        u_eq,
        q,
        r,
        M,
        u_min=-np.inf,
        u_max=np.inf,
        pos_min=-np.inf * np.ones(2),
        pos_max=np.inf * np.ones(2),
        penalty=1e3,
    ):
        super().__init__(f, df, x_eq, u_eq, q, r)

        self.x = cp.Variable((6, M + 1))
        self.u = cp.Variable((2, M))

        slack = cp.Variable((2, M + 1))

        self.x_init = cp.Parameter(6)

        constraints = [
            self.x[:, 1:] == self.A @ self.x[:, :-1] + self.B @ self.u,
            self.x[:, 0] == self.x_init,
            self.u >= u_min - u_eq[:, np.newaxis],
            self.u <= u_max - u_eq[:, np.newaxis],
            self.x[:2, :] + slack >= pos_min[:, np.newaxis] - x_eq[:2, np.newaxis],
            self.x[:2, :] + slack <= pos_max[:, np.newaxis] - x_eq[:2, np.newaxis],
        ]

        LQ = np.linalg.cholesky(self.q)
        LR = np.linalg.cholesky(self.r)

        objective = cp.Minimize(
            cp.sum_squares(LQ.T @ self.x[:, :-1])
            + cp.sum_squares(LR.T @ self.u)
            + cp.quad_form(self.x[:, -1], self.P)
            + penalty * cp.norm1(slack)
        )

        self.problem = cp.Problem(objective, constraints)

    def input(self, x, _):
        self.x_init.value = x - self.x_eq
        self.problem.solve(solver=cp.OSQP, warm_starting=True, polish=False)

        return self.u_eq + self.u.value[:, 0]  # ty:ignore[not-subscriptable]
