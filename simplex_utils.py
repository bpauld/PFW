import numpy as np

# Define linear optimization oracles and support function on the simplex

def oracle_simplex(u):
    # return argmin_{s \in K} s^T u
    # where K = simplex
    
    d = u.shape
    j = np.argmin(u)
    result = np.zeros(d)
    result[j] = 1
    return result

def stochastic_oracle_simplex(u, epsilon, variance=1):
    # return argmax_{s \in K} s^T (u + epsilon * Delta)
    # where Delta is distributed according to a Gumbel with loc = 0 and scale = variance
    # where K = simplex
    
    d = u.shape[0]
    Delta = np.random.gumbel(0, variance, d)
    vector = u + epsilon * Delta    
    j = np.argmax(vector)
    result = np.zeros(d)
    result[j] = 1
    return result

def support_function_simplex(y):
    return np.max(y)  