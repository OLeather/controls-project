from multiprocessing import Process
from multiprocessing import Queue
from threading import Thread

import matplotlib
from casadi import *
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import visualization
from visualization import plot_pendulum_cart, setup_pendulum_cart_plot

matplotlib.use("TkAgg")
# model parameters
m = 1  # pendulum mass
M = 5  # cart mass
L = 2  # pendulum length
d = 1  # pendulum friction
g = -9.8  # gravity
u_max = 100  # max force

# MPC parameters
T = 3  # Time horizon
N = 10  # Number of control intervals

# State symbols
x1 = MX.sym('x1')  # x
x2 = MX.sym('x2')  # x_dot
x3 = MX.sym('x3')  # theta
x4 = MX.sym('x4')  # theta_dot

x = vertcat(x1, x2, x3, x4)

# Control input symbol
u = MX.sym('u', 1)

# Nonlinear inverted pendulum equations
x_dot = vertcat(x[1],
                (-m ** 2 * L ** 2 * g * cos(x[2]) * sin(x[2]) + m * L ** 2 * (
                        m * L * x[3] ** 2 * sin(x[2]) - d * x[1]) + m * L ** 2 * u) / (
                        m * L ** 2 * (M + m * (1 - cos(x[2]) ** 2))),
                x[3],
                ((m + M) * m * g * L * sin(x[2]) - m * L * cos(x[2]) * (
                        m * L * x[2] ** 2 * sin(x[2]) - d * x[1]) - m * L * cos(x[2]) * u) / (
                        m * L ** 2 * (M + m * (1 - cos(x[2]) ** 2))))

ode = {'x': x, 'p': u, 'ode': x_dot}
SysModel = integrator('F', 'rk', ode, {'tf': T / N})
SysModel1 = integrator('F', 'rk', ode, {'tf': .01})
# Setup CasADI optimizer
opti = casadi.Opti()

# Setup optimizer variables
x_traj_param = opti.variable(4, N + 1)  # Optimized state over time horizon
u_traj_param = opti.variable(1, N)  # Optimized control input over time horizon
x0_param = opti.parameter(4, 1)  # Initial state parameter
x_goal_param = opti.parameter(4, 1)  # Goal state parameter

# Setup minimization objective to minimize overall force used
opti.minimize(sumsqr(u_traj_param))

# Set optimization constraints
for k in range(0, N):
    # X_k+1 = SysModel(X_k, U_k)
    opti.subject_to(x_traj_param[:, k + 1] == SysModel(x0=x_traj_param[:, k], p=u_traj_param[:, k])["xf"])

# Initial state in trajectory = x0
opti.subject_to(x_traj_param[:, 0] == x0_param)
# Final state in trajectory = x_goal
opti.subject_to(x_traj_param[:, -1] == x_goal_param)
# Constrain u to between min and max bounds
opti.subject_to(Opti_bounded(-u_max, u_traj_param, u_max))

u_traj = None
x_traj = None


def mpc(x0, x_goal):
    # Set the initial guess to the previous predicted trajectory if exists
    if x_traj is not None and u_traj is not None:
        opti.set_initial(x_traj_param, x_traj)
        opti.set_initial(u_traj_param, u_traj)

    # Set optimization parameters
    opti.set_value(x0_param, x0)
    opti.set_value(x_goal_param, x_goal)

    # Solve optimization problem
    solver_options = {"ipopt.tol": 1e-6, "ipopt.print_level": 0, "print_time": 0}
    opti.solver('ipopt', solver_options)

    sol = opti.solve()

    return sol.value(u_traj_param), sol.value(x_traj_param)


if __name__ == "__main__":
    # Initial state and goal state variables
    x0 = np.matrix([[0], [0], [0], [0]])
    x_goal = np.matrix([[0], [0], [0], [0]])

    # Setup pendulum cart plot
    line1, line2, line3, fig = setup_pendulum_cart_plot()

    t = 0


    # Run simulation and animate pendulum cart
    def run_sim(i):
        global t, x0, x_goal, u_traj, x_traj
        if i % int((T / N) / 0.01) == 0:
            print("Running MPC")
            u_traj, x_traj = mpc(x0, x_goal)

        t += 0.01
        print(t)


        # Setpoints
        if t > 1:
            x_goal = np.matrix([[0], [0], [pi], [0]])
        if t > 7.5:
            x_goal = np.matrix([[-4], [0], [pi], [0]])
        if t > 12.5:
            x_goal = np.matrix([[4], [0], [pi], [0]])

        x0 = SysModel1(x0=x0, p=u_traj[0])["xf"]

        plot_pendulum_cart(float(x0[0]), float(x0[2]), line1, line2, line3)

        return line1, line2, line3


    anim = FuncAnimation(fig, run_sim, frames=1000, interval=10, blit=True)
    plt.show()
