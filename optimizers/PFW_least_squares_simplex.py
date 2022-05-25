from update_Ak import update_Ak
import numpy as np
from simplex_utils import oracle_simplex, stochastic_oracle_simplex, support_function_simplex

def PFW_least_squares_simplex(A,
        b,
        epsilon,
        L,
        x0,
        m,
        n_iter,
        R_K=1,
        M_mu=1,
        x_optimal=None):
    
    
    ATb = np.dot(A.T, b)
    Ak = 0
    dk = 0
    
    if m=="theory":
        nb_stoch_grads = int(np.sqrt(1/epsilon))
    else:
        nb_stoch_grads = m
    
    fxk_list = []
    min_fxk_list = []
    dual_fct_yk_list = []
    pd_gap_list = []
    min_pd_gap_list = []
    ub_list = []
    min_fxk_so_far = np.inf
    min_pd_gap_so_far = np.inf
    
    xk = x0
    fx0 = 0.5 * np.linalg.norm(np.dot(A, xk) - b)**2
    grad_xk = np.dot(A.T, np.dot(A, xk) - b)
    yk = grad_xk
    
    
        
    mu_h = 1 / L
    mu_w = mu_h
    beta = R_K * M_mu / epsilon
    
    
    if x_optimal is not None:
        ub_constant_part = 2 * np.sqrt(epsilon * L) / nb_stoch_grads + epsilon
        ub_exp_constant_term = beta * (fx0 - 0.5 * np.linalg.norm(np.dot(A, x_optimal) - b)**2)
    
    
    for k in range(n_iter):
        #Update function value lists
        dual_fct_yk_value = 0.5 * np.dot(yk + ATb, np.linalg.solve(np.dot(A.T, A), yk + ATb)) - 0.5 * np.dot(b, b) + support_function_simplex(-yk)        
        fxk = 0.5 * np.linalg.norm(np.dot(A, xk) - b)**2
        pd_gap = fxk + dual_fct_yk_value
        
        if fxk < min_fxk_so_far:
            min_fxk_so_far = fxk
        if pd_gap < min_pd_gap_so_far:
            min_pd_gap_so_far = pd_gap
                   
        
       
        
        fxk_list.append(fxk)
        dual_fct_yk_list.append(dual_fct_yk_value)
        pd_gap_list.append(pd_gap)
        min_fxk_list.append(min_fxk_so_far)
        min_pd_gap_list.append(min_pd_gap_so_far)
        
        
        #compute theoretical upper bound
        if x_optimal is not None:
            ub_k = np.exp(-k * np.sqrt(mu_h) / (2 * (np.sqrt(beta) + np.sqrt(mu_h)))) * ub_exp_constant_term + ub_constant_part
            ub_list.append(ub_k)
        
        
        
        if True:
            if k%(int(n_iter / 10)) == 0:
                output = 'Iteration.: %d, dual gap: %.2e' % \
                     (k, pd_gap)
                output += ', Min dual gap: %e' % min_pd_gap_so_far
                print(output) 
        
        
        #proceed to algorithm
        Ak1 = update_Ak(Ak, mu_h, beta, mu_w)
        
        tau_k = 1 - Ak / Ak1
        vk = (1 - tau_k) * yk + tau_k * grad_xk
        
        #get m stochastic gradients
        gk = -stochastic_oracle_simplex(-vk, epsilon)
        for i in range(1, nb_stoch_grads):
            gk -= stochastic_oracle_simplex(-vk, epsilon)
        gk = gk / nb_stoch_grads
            
        
               
        dk = dk + (Ak1 - Ak) * gk
        xk = beta / (beta + Ak1) * x0 - dk / (beta + Ak1)
        grad_xk = np.dot(A.T, np.dot(A, xk) - b)
        yk = (1 - tau_k) * yk + tau_k * grad_xk
       
        Ak = Ak1
        
    return fxk_list, dual_fct_yk_list, pd_gap_list, min_fxk_list, min_pd_gap_list, ub_list