from math import sin, pi
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import linalg
from scipy import signal

from sim import sim_pendulum

# Example: Inverted Pendulum http://databookuw.com/databook.pdf#page=355
if __name__ == "__main__":
    dt = 0.01

    g = -9.8
    L = 2
    d = 1

    x = np.matrix([[0],  # theta
                   [0]])  # theta_dot

    s = -1  # pendulum up (1) or down (-1)

    A = np.matrix([[0, 1],
                   [s * g / L, d]])

    B = np.matrix([[0],
                   [1]])

    poles = [-2, -2.1]

    # K = np.matrix([[10, 5]])

    K = signal.place_poles(A, B, poles).gain_matrix

    print("K:", K)

    x_ref = np.matrix([[pi],  # theta
                       [0]])  # theta_dot

    print("Eigs:", linalg.eig((A - B * K))[0])

    xs = []
    ts = []
    for t in np.arange(0, 10, dt):
        u = -K * (x - x_ref)
        theta, theta_dot = sim_pendulum(g, L, d, x, u, dt)
        x = np.matrix([[theta],
                       [theta_dot]])
        xs.append(float(x[0]))
        ts.append(t)

    fig, ax = plt.subplots()
    ax.plot(ts, xs)
    ax.grid(True)
    plt.show()
