import numpy as np
from simplex_utils import oracle_simplex, stochastic_oracle_simplex, support_function_simplex


def FW_least_squares_simplex(A, b, x0, n_iter):
    xk = x0
    ATb = np.dot(A.T, b)
    
    
    fxk_list = []
    min_fxk_list = []
    pd_gap_list = []
    min_pd_gap_list = []
    min_fxk_so_far = np.inf
    min_pd_gap_so_far = np.inf

    for k in range(n_iter):
        grad_xk = np.dot(A.T, np.dot(A, xk) - b)
        yk = grad_xk
        
        fxk = 0.5 * np.linalg.norm(np.dot(A, xk) - b)**2
        pd_gap = fxk + 0.5 * np.dot(yk + ATb, np.linalg.solve(np.dot(A.T, A), yk + ATb)) - 0.5 * np.dot(b, b) + support_function_simplex(-yk)
        
        
        if fxk < min_fxk_so_far:
            min_fxk_so_far = fxk
        if pd_gap < min_pd_gap_so_far:
            min_pd_gap_so_far = pd_gap
            
        fxk_list.append(fxk)
        pd_gap_list.append(pd_gap)
        min_fxk_list.append(min_fxk_so_far)
        min_pd_gap_list.append(min_pd_gap_so_far)
    
    
        sk = oracle_simplex(grad_xk)
        eta_k = 2 / (k+2)
        xk = (1 - eta_k) * xk + eta_k * sk
        
        
    return fxk_list, min_fxk_list, pd_gap_list, min_pd_gap_list