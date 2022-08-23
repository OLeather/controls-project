from math import sin, cos

# x0 = [[theta],
#       [theta_dot]]
def sim_pendulum(L, d, g, x0, u, dt):
    theta_dot_dot = -g / L * sin(x0[0]) + u[0] - d * x0[1]
    theta_dot = x0[1] + theta_dot_dot * dt
    theta = x0[0] + theta_dot * dt
    return float(theta), float(theta_dot)


# x0 = [[x],
#       [x_dot],
#       [theta],
#       [theta_dot]]
def sim_pendulum_cart(m, M, L, d, g, x0, u, dt):
    x = x0[0]
    x_dot = x0[1]
    theta = x0[2]
    theta_dot = x0[3]

    x_dot_dot = (-m**2 * L**2 * g * cos(theta) * sin(theta) + m * L**2 * (m * L * theta_dot**2 * sin(theta) - d * x_dot) + m * L**2 * u) / (m * L**2 * (M + m * (1 - cos(theta)**2)))

    theta_dot_dot = ((m + M) * m * g * L * sin(theta) - m * L * cos(theta) * (m * L * theta_dot**2 * sin(theta) - d * x_dot) - m * L * cos(theta) * u) / (m * L**2 * (M + m * (1 - cos(theta)**2)))

    x_dot = x_dot + x_dot_dot * dt
    x = x + x_dot * dt

    theta_dot = theta_dot + theta_dot_dot * dt
    theta = theta + theta_dot * dt

    return float(x), float(x_dot), float(theta), float(theta_dot)
