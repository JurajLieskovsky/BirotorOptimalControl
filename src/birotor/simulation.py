import scipy
import numpy as np
import matplotlib.pyplot as plt

from . import dynamics


def input_saturation(u, u_min, u_max):
    if u < u_min:
        return u_min
    elif u > u_max:
        return u_max
    else:
        return u


def simulate(
    nstep,
    timestep,
    x0,
    controller,
    running_cost,
    final_cost,
    u_min=-np.inf,
    u_max=np.inf,
    w_scale=0
):
    solver = scipy.integrate.ode(dynamics.f)
    solver.set_integrator("dopri5")
    solver.set_initial_value(x0)

    ts = np.zeros(nstep + 1)
    cs = np.zeros(nstep + 1)
    xs = [np.zeros(6) for _ in range(nstep + 1)]
    us = [np.zeros(2) for _ in range(nstep + 1)]

    ts[0] = 0.0
    xs[0] = solver.y

    for k in range(nstep):
        u = np.array(
            [input_saturation(input, u_min, u_max) for input in controller(solver.y, k)]
        )
        solver.set_f_params(u, np.random.normal(0.0, w_scale))
        solver.integrate(solver.t + timestep)

        us[k] = u
        xs[k + 1] = solver.y
        ts[k + 1] = solver.t

        cs[k] = running_cost(xs[k], us[k], k)

    us[nstep] = us[nstep - 1]

    cs[nstep] = final_cost(xs[nstep])

    return ts, xs, us, cs


def plot_trajectory(xs):
    fig, ax = plt.subplots()

    ax.plot([x[0] for x in xs], [x[1] for x in xs])

    ax.set_xlabel(r"$y$ [m]")
    ax.set_ylabel(r"$z$ [m]")

    ax.set_aspect('equal')

    return fig, ax


def plot_states_and_inputs(ts, xs, us):
    fig, ax = plt.subplots(3)

    state_labels = [
        r"$y$ [m]",
        r"$z$ [m]",
        r"$\theta$ [rad]",
        r"$\dot{y}$ [m/s]",
        r"$\dot{z}$ [m/s]",
        r"$\dot{\theta}$ [rad/s]",
    ]
    input_labels = [
        r"$u_L$ [N]",
        r"$u_R$ [N]",
    ]

    for i, lbl in enumerate(state_labels[:3]):
        ax[0].plot(ts, [x[i] for x in xs], label=lbl)

    for i, lbl in enumerate(state_labels[3:]):
        ax[1].plot(ts, [x[i + 3] for x in xs], label=lbl)

    for i, lbl in enumerate(input_labels):
        ax[2].step(ts, [u[i] for u in us], where="post", label=lbl)

    ax[2].set_xlabel("$t$ [s]")

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    return fig, ax
