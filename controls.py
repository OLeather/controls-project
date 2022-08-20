import numpy as np
import scipy.linalg


# Returns the controllability matrix of the system
# C = [B AB A^2B ... A^(n-1)B]
def ctrb(A, B):
    return []


# Returns the control K matrix given a set of eigenvalues and system matrices
def place(A, B, eigs):
    return []


def lqr_c(A, B, Q, R):
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) * (B.T * P)
    return K


def lqr_d(A, B, Q, R):
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R + B.T * P * B) * (B.T * P * A)
    return K
