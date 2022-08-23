from math import cos, sin, pi

from matplotlib import pyplot as plt


def setup_pendulum_cart_plot():
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    line1, = ax.plot([], [])
    line2, = ax.plot([], [], color='black')
    line3, = ax.plot([], [], 'o', color='black')
    plt.xlim([-6, 6])
    plt.ylim([-3, 3])
    plt.grid()
    return line1, line2, line3, fig


def plot_pendulum_cart(x, theta, line1, line2, line3):
    theta -= pi / 2

    line1.set_xdata([x, x + 2 * cos(theta)])
    line1.set_ydata([0, 2 * sin(theta)])
    line2.set_xdata([x - .5, x + .5])
    line2.set_ydata([0, 0])
    line3.set_xdata([x, x + 2 * cos(theta)])
    line3.set_ydata([0.05, 2 * sin(theta)])

    return line1, line2, line3
