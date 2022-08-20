import time
from math import sin, pi, cos

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import linalg
from scipy import signal

matplotlib.use("TkAgg")

from sim import sim_pendulum_cart


def plot_pendulum_cart(x, theta, line, line1, line2):
    theta -= pi / 2
    line.set_xdata([x, x + 3*cos(theta)])
    line.set_ydata([0, 3*sin(theta)])
    line1.set_xdata([x-.5, x+.5])
    line1.set_ydata([0, 0])
    line2.set_xdata([x, x+3*cos(theta)])
    line2.set_ydata([0.05, 3*sin(theta)])
    # ax.plot([x, x + cos(theta)], [0, sin(theta)], label="x")
    plt.pause(0.01)


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
                   [pi],  # theta
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

    K = signal.place_poles(A, B, poles).gain_matrix

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

    fig, ax = plt.subplots()

    line, = ax.plot([], [])
    line1, = ax.plot([], [], color='black')
    line2, = ax.plot([], [], 'o', color='black')
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.grid()
    plot_pendulum_cart(float(x[0]), float(x[2]), line, line1, line2)
    plt.pause(2)
    for t in np.arange(0, 10, dt):
        u = -K * (x - x_ref)
        # print("x:", x, "u:", u)
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
        plot_pendulum_cart(float(x[0]), float(x[2]), line, line1, line2)
        # plt.xlim([-5, 5])
        # plt.ylim([-1, 2])
        # plt.grid()

    plt.show()

    # fig, ax = plt.subplots()
    # line1, = ax.plot(ts, xs, label="x")
    # line2, = ax.plot(ts, thetas, label="theta")
    # line3, = ax.plot(ts, us, label="u")
    # ax.legend(handles=[line1, line2, line3])
    # ax.grid(True)
    # plt.show()
