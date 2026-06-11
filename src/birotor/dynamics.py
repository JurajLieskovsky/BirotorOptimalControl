import jax
import jax.numpy as np

g = 9.81
mass = 1
moi = 1
arm = 0.1


# equilibrium input
u_eq = mass * g / 2 * np.ones(2)


# continuous-time dynamics
def f(_, x, u, w=0):
    return np.array(
        [
            x[3],
            x[4],
            x[5],
            -np.sin(x[2]) * (u[0] + u[1]) / mass + w,
            np.cos(x[2]) * (u[0] + u[1]) / mass - g,
            arm * (u[0] - u[1]) / moi,
        ]
    )


def df(t, x, u):
    return jax.jacobian(f, argnums=(1, 2))(t, x, u)


# discrete-time dynamics
def rk4_f(k, x, u, dt):
    k1 = f(dt * (k), x, u)
    k2 = f(dt * (k + 1 / 2), x + dt / 2 * k1, u)
    k3 = f(dt * (k + 1 / 2), x + dt / 2 * k2, u)
    k4 = f(dt * (k + 1), x + dt * k3, u)

    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_df(k, x, u, dt):
    return jax.jacobian(rk4_f, argnums=(1, 2))(k, x, u, dt)
