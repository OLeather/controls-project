import time
from math import sin, pi, cos

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from numpy.linalg import linalg
from scipy import signal

from controls import lqr_c

matplotlib.use("TkAgg")

from sim import sim_pendulum_cart

# Example: Inverted Pendulum on Cart http://databookuw.com/databook.pdf#page=372
if __name__ == "__main__":
    dt = 0.01

    m = 1
    M = 5
    L = 2
    d = 1
    g = -9.8

    s = 1  # pendulum up (1) or down (-1)

    x = np.matrix([[-2],  # x
                   [0],  # x_dot
                   [pi+.1],  # theta
                   [0]])  # theta_dot

    # Linearized system matrices
    A = np.matrix([[0, 1, 0, 0],
                   [0, -d / M, s * m * g / M, 0],
                   [0, 0, 0, 1],
                   [0, -s * d / (M * L), -s * (m + M) * g / (M * L), 0]])

    B = np.matrix([[0],
                   [1 / M],
                   [0],
                   [s * 1 / (M * L)]])

    poles = [-2, -2.1, -2.2, -2.3]

    # K = signal.place_poles(A, B, poles).gain_matrix

    Q = np.diag([50, 50, 10, 100])
    R = np.matrix([[.001]])
    K = lqr_c(A, B, Q, R)

    print("K:", K)

    x_ref = np.matrix([[2],  # x
                       [0],  # x_dot
                       [pi],  # theta
                       [0]])  # theta_dot

    print("Eigs:", linalg.eig((A - B * K))[0])

    xs = []
    thetas = []
    us = []
    ts = []

    # plt.pause(2)
    for t in np.arange(0, 10, dt):
        u = -K * (x - x_ref)
        # u = np.matrix([[0]])
        x, x_dot, theta, theta_dot = sim_pendulum_cart(m, M, L, d, g, x, u, dt)
        x = np.matrix([[x],
                       [x_dot],
                       [theta],
                       [theta_dot]])
        xs.append(float(x[0]))
        thetas.append(float(x[2]))
        us.append(float(u))
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
        x = xs[i]
        theta = thetas[i] - pi / 2
        line.set_xdata([x, x + 2 * cos(theta)])
        line.set_ydata([0, 2 * sin(theta)])
        line1.set_xdata([x - .5, x + .5])
        line1.set_ydata([0, 0])
        line2.set_xdata([x, x + 2 * cos(theta)])
        line2.set_ydata([0.05, 2 * sin(theta)])
        return line, line1, line2


    anim = FuncAnimation(fig, animate, frames=len(xs), interval=dt*1000, blit=True)

    plt.show()
