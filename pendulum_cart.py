import time
from math import sin, pi, cos

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from numpy.linalg import linalg
from scipy import signal

from controls import lqr_c, discretize_euler, lqr_d, discretize

matplotlib.use("TkAgg")

from sim import sim_pendulum_cart

# Example: Inverted Pendulum on Cart http://databookuw.com/databook.pdf#page=372
if __name__ == "__main__":
    dt = .01

    m = 1
    M = 5
    L = 2
    d = 1
    g = -9.8

    x0 = np.matrix([[-2],  # x
                    [0],  # x_dot
                    [pi+.1],  # theta
                    [0]])  # theta_dot

    s = 1  # pendulum up (1) or down (-1)

    # Linearized system matrices
    A = np.matrix([[0, 1, 0, 0],
                   [0, -d / M, s * m * g / M, 0],
                   [0, 0, 0, 1],
                   [0, -s * d / (M * L), -s * (m + M) * g / (M * L), 0]])

    B = np.matrix([[0],
                   [1 / M],
                   [0],
                   [s * 1 / (M * L)]])

    Ad, Bd = discretize(A, B, dt)

    poles = [-2, -2.1, -2.2, -2.3]

    # K = signal.place_poles(A, B, poles).gain_matrix

    Q = np.diag([10, 10, 10, 100])
    R = np.matrix([[.001]])

    K = lqr_d(Ad, Bd, Q, R)

    print("K:", K)

    x_ref = np.matrix([[2],  # x
                       [0],  # x_dot
                       [pi],  # theta
                       [0]])  # theta_dot


    x_traj = []
    thetas = []
    u_traj = []
    ts = []

    # plt.pause(2)
    for t in np.arange(0, 10, dt):
        u = -K * (x0 - x_ref)
        # u = np.matrix([[0]])
        x0, x_dot, theta, theta_dot = sim_pendulum_cart(m, M, L, d, g, x0, u, dt)
        x0 = np.matrix([[x0],
                        [x_dot],
                        [theta],
                        [theta_dot]])
        x_traj.append(float(x0[0]))
        thetas.append(float(x0[2]))
        u_traj.append(float(u))
        ts.append(t)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    line, = ax.plot([], [])
    line1, = ax.plot([], [], color='black')
    line2, = ax.plot([], [], 'o', color='black')
    plt.xlim([-5, 5])
    plt.ylim([-3, 3])
    plt.grid()


    def animate(i):
        x = x_traj[i]
        theta = thetas[i] - pi / 2
        line.set_xdata([x, x + 2 * cos(theta)])
        line.set_ydata([0, 2 * sin(theta)])
        line1.set_xdata([x - .5, x + .5])
        line1.set_ydata([0, 0])
        line2.set_xdata([x, x + 2 * cos(theta)])
        line2.set_ydata([0.05, 2 * sin(theta)])
        return line, line1, line2


    anim = FuncAnimation(fig, animate, frames=len(x_traj), interval=dt * 1000, blit=True)

    plt.show()
