# Inverted Pendulum Controls
<p align="center">
  <img src="https://github.com/OLeather/controls-project/assets/43189206/f7a413cc-d8e8-4980-9d06-b5f8a8827379)" alt="Sublime's custom image"/>
</p>
<p align="center">Figure 1: Linear State Space Control of Inverted Pendulum</p>

I have been learning about modern control theory through online resources such as [Steve Bruntun's Controls Bootcamp](https://www.youtube.com/watch?v=Pi7l8mMjYVE&list=PLMrJAkhIeNNR20Mz-VpzgfQs5zrYi085m). As an exercise to apply my learning of state space and model predictive control, I worked on a project to stabilize an inverted pendulum on a cart -- a classic controls project. I did multiple exercises, including linear state space control, linearized model predictive control, and nonlinear model predictive control. The code can be found on my public repo <i class="fa fa-github"></i> [controls-project](https://github.com/OLeather/controls-project).

## Linear State Space

I derived the following linearized state space model of the inverted pendulum on the cart. The model is linearized in the up position to allow for balancing. 
```python
x = np.matrix([[-2],  # x
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
```

I then implemented both pole placement and a linear quadratic regulator to derive the control gain used in the state space control law. <i class="fa fa-github"></i> [pendulum_cart.py](https://github.com/OLeather/controls-project/blob/main/pendulum_cart.py)

```python
# Linear Quadratic Regulator control gain
K = lqr_d(Ad, Bd, Q, R)

# Control law
u = -K * (x - x_ref)
```

## MPC

I implemented model predictive control using the numerical optimization libraries [cvxpy](https://www.cvxpy.org/) and [CasADI](https://web.casadi.org/). For both applications, I implemented the model of the system (either linearized or nonlinear), and optimized the control laws over a finite time horizon. 

The linearized optimization problem used the linearized state space model derived for the state space application. It then optimized the control law over the time horizon to minimize state error and remain balanced. I used cvxpy, a convex optimization library, to implement the linear optimization. <i class="fa fa-github"></i> [pendulum_cart_mpc.py](https://github.com/OLeather/controls-project/blob/main/pendulum_cart_mpc.py)
```python
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
```

For the nonlinear optimization problem, I used the nonlinear ordinary differential equation representation of the dynamics of the pendulum cart. This allows for additional control, such as swing up and swing down, which were not possible with the linearized model.

```python
# Nonlinear model of the inverted pendulum on a cart
x_dot = vertcat(x[1],
                (-m ** 2 * L ** 2 * g * cos(x[2]) * sin(x[2]) + m * L ** 2 * (
                        m * L * x[3] ** 2 * sin(x[2]) - d * x[1]) + m * L ** 2 * u) / (
                        m * L ** 2 * (M + m * (1 - cos(x[2]) ** 2))),
                x[3],
                ((m + M) * m * g * L * sin(x[2]) - m * L * cos(x[2]) * (
                        m * L * x[2] ** 2 * sin(x[2]) - d * x[1]) - m * L * cos(x[2]) * u) / (
                        m * L ** 2 * (M + m * (1 - cos(x[2]) ** 2))))

ode = {'x': x, 'p': u, 'ode': x_dot}

# System model is an integration of the ode
SysModel = integrator('F', 'rk', ode, {'tf': T / N})
```

I used the numeric optimization library CasADI to setup the nonlinear optimization problem, and optimized using a similar method as linear MPC by minimizing the sum squared error over a time horizon to derive the next optimal control input. The optimization parameters I used are shown below:
```python
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
```

<p align="center">
  <img src="https://github.com/OLeather/controls-project/assets/43189206/4791d778-c9e3-49f0-bfe7-4c7dfdb08025)" alt="Sublime's custom image"/>
</p>
<p align="center">Figure 2: Nonlinear Model Predictive Control Swing Up and Balancing</p>
