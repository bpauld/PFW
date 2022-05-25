from scipy.sparse.linalg import svds
import numpy as np

# Define linear optimization oracles and support function on the trace norm ball

def support_function_trace_ball(Z, max_norm=1):
    #u, s, vh = np.linalg.svd(Z)
    u, s, vh = svds(Z, k=1, which = "LM")
    return s[0] * max_norm


def oracle_trace_ball(X, max_norm=1):
    # return argmin_{s \in K} s^T u
    # where K = max_norm * trace_norm ball
    (n, m) = X.shape
    #u, s, vh = np.linalg.svd(X)
    u, s, vh = svds(X, k=1, which = "LM")
    u1 = u[:, 0].reshape((n, 1))
    v1 = vh[0, :].reshape((1, m))
    Z = np.dot(u1, v1)
    return - max_norm * Z

def stochastic_oracle_trace_ball(X, epsilon, variance, max_norm=1):
    # return argmax_{s \in K} s^T (X + epsilon * Delta)
    # where Delta is distributed according to a normal distribution with mean 0 and variance = variance
    # where K = max_norm * trace_ball
    
    (n, m) = X.shape
    Delta = np.random.normal(0, np.sqrt(variance), size=(n, m)) #for now
    #u, s, vh = np.linalg.svd(X + epsilon * Delta)
    u, s, vh = svds(X + epsilon * Delta, k=1, which = "LM")
    u1 = u[:, 0].reshape((n, 1))
    v1 = vh[0, :].reshape((1, m))
    Z = np.dot(u1, v1)
    return max_norm * Z