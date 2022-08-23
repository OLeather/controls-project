import numpy as np
import scipy.linalg

# Returns the controllability matrix of the system
# C = [B AB A^2B ... A^(n-1)B]
from numpy import vectorize


def ctrb(A, B):
    return []


# Continuous time linear quadratic regulator
def lqr_c(A, B, Q, R):
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) * (B.T * P)
    return K


# Discrete time linear quadratic regulator
def lqr_d(A, B, Q, R):
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R + B.T * P * B) * (B.T * P * A)
    return K


# Discretization of A and B system matrices
def discretize(A, B, dt):
    Ad = scipy.linalg.expm(A * dt)
    Bd = dt * B  # TODO: non-approximation of Bd

    return Ad, Bd


# Euler approximation of discretization A and B system matrices
def discretize_euler(A, B, dt):
    Ad = np.linalg.inv(np.identity(len(A)) - A * dt)
    Bd = dt * B
    return Ad, Bd
