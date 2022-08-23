from math import pi

import cvxpy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from visualization import plot_pendulum_cart, setup_pendulum_cart_plot
from controls import discretize
from sim import sim_pendulum_cart

matplotlib.use("TkAgg")

dt = .1

# model parameters
m = 1  # pendulum mass
M = 5  # cart mass
L = 2  # pendulum length
d = 1  # pendulum friction
g = -9.8  # gravity

s = 1  # pendulum up (1) or down (-1)

# Linearized system matrices in continuous time
A = np.matrix([[0, 1, 0, 0],
               [0, -d / M, s * m * g / M, 0],
               [0, 0, 0, 1],
               [0, -s * d / (M * L), -s * (m + M) * g / (M * L), 0]])

B = np.matrix([[0],
               [1 / M],
               [0],
               [s * 1 / (M * L)]])

# Discretize A and B matrices to form X_k+1 = Ad*X_k + Bd*U_k
Ad, Bd = discretize(A, B, dt)

x0 = np.matrix([[-2],  # x
                [0],  # x_dot
                [pi - .1],  # theta
                [0]])  # theta_dot

x_goal = np.matrix([[2],  # x
                    [0],  # x_dot
                    [pi],  # theta
                    [0]])  # theta_dot

# State error weights
Q = np.diag([10.0, 10.0, 10.0, 10.0])

# Sum of forces penalty weight
R = np.diag([0.05])


def mpc(x0):
    x0 = np.asarray(x0)

    T = 30

    # Setup optimizer variables
    x_traj = cvxpy.Variable((len(x0), T + 1))  # Optimized state over time horizon
    u_traj = cvxpy.Variable((1, T))  # Optimized control input over time horizon

    cost = 0 # Cost variable
    constraints = [] # Optimizer constraints

    for t in range(T):
        # Quad form cost variable to minimize total state error and total force
        cost += cvxpy.quad_form(x_traj[:, t + 1], Q)
        cost += cvxpy.quad_form(u_traj[:, t], R)

        # X_k+1 = Ad*X_k + Bd*U_k
        constraints += [x_traj[:, t + 1] == Ad @ x_traj[:, t] + Bd @ u_traj[:, t]]

    # Initial state = x0 - x_goal
    constraints += [x_traj[:, 0] == x0[:, 0] - np.asarray(x_goal)[:, 0]]

    # Minimize cost function for optimal control trajectory u
    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)

    # Solve
    prob.solve(verbose=False)

    # Return first control input in trajectory to apply to the system
    return u_traj.value[0, 1]


line1, line2, line3, fig = setup_pendulum_cart_plot()


def run_sim(i):
    global x0
    # Get optimal control input u from MPC
    u = mpc(x0)

    # Simulate using control input
    x_, x_dot, theta, theta_dot = sim_pendulum_cart(m, M, L, d, g, x0, u, dt)

    # Update x0
    x0 = np.matrix([[x_],
                    [x_dot],
                    [theta],
                    [theta_dot]])

    # Plot
    plot_pendulum_cart(float(x0[0]), float(x0[2]), line1, line2, line3)

    return line1, line2, line3


anim = FuncAnimation(fig, run_sim, frames=1000, interval=1, blit=True)

plt.show()
