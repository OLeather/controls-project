from math import sin


def sim_pendulum(g, L, d, x0, u, dt):
    theta_dot_dot = -g / L * sin(x0[0]) + u[0] - d * x0[1]
    theta_dot = x0[1] + theta_dot_dot * dt
    theta = x0[0] + theta_dot * dt
    return float(theta), float(theta_dot)