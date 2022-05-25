from trace_ball_utils import support_function_trace_ball, oracle_trace_ball, stochastic_oracle_trace_ball
import numpy as np



def FW_least_squares_trace_ball(A, B, X0, n_iter, max_norm=1):
    Xk = X0
    ATB = np.dot(A.T, B)
    
    
    fXk_list = []
    min_fXk_list = []
    pd_gap_list = []
    min_pd_gap_list = []
    min_fXk_so_far = np.inf
    min_pd_gap_so_far = np.inf

    for k in range(n_iter):
        grad_Xk = np.dot(A.T, np.dot(A, Xk) - B)
        Yk = grad_Xk
        
        fXk = 0.5 * np.linalg.norm(np.dot(A, Xk) - B)**2
        pd_gap = fXk + 0.5 * np.trace(np.dot((Yk + ATB).T, np.linalg.solve(np.dot(A.T, A), Yk + ATB))) - 0.5 * np.trace(np.dot(B.T, B)) + support_function_trace_ball(-Yk, max_norm=max_norm)
        
        
        if fXk < min_fXk_so_far:
            min_fXk_so_far = fXk
        if pd_gap < min_pd_gap_so_far:
            min_pd_gap_so_far = pd_gap
            
        fXk_list.append(fXk)
        pd_gap_list.append(pd_gap)
        min_fXk_list.append(min_fXk_so_far)
        min_pd_gap_list.append(min_pd_gap_so_far)
    
    
        sk = oracle_trace_ball(grad_Xk, max_norm=max_norm)
        eta_k = 2 / (k+1)
        Xk = (1 - eta_k) * Xk + eta_k * sk
        
    return fXk_list, min_fXk_list, pd_gap_list, min_pd_gap_list    
