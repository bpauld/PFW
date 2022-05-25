from update_Ak import update_Ak
import numpy as np
from simplex_utils import oracle_simplex, stochastic_oracle_simplex, support_function_simplex

def RPFW_least_squares_simplex(A,
                               b,
                               epsilon_start,
                               epsilon_end,
                               epsilon_decrease,
                               L,
                               x0,
                               m,
                               max_iter,
                               R_K=1,
                               M_mu=1,
                               verbose=True):
    
    
    ATb = np.dot(A.T, b)
    
    fxk_list = []
    pd_gap_list = []
    min_fxk_list = []
    min_pd_gap_list = []
    
    mu_h = 1 / L
    mu_w = mu_h
    xk = x0
    
    grad_xk = np.dot(A.T, np.dot(A, xk) - b)
    yk = grad_xk    
    fxk = 0.5 * np.linalg.norm(np.dot(A, xk) - b)**2
    

    min_fxk_so_far = fxk
    min_pd_gap_so_far = fxk + 0.5 * np.dot(yk + ATb, np.linalg.solve(np.dot(A.T, A), yk + ATb)) - 0.5 * np.dot(b, b) + support_function_simplex(-yk)
    
    total_iter = 0
    
    epsilon = epsilon_start
     
    while epsilon >= epsilon_end:
        
        beta = R_K * M_mu / epsilon
        dk = 0
        Ak = 0
            
        n_iter =  int(np.sqrt(L) / np.sqrt(epsilon) * np.log(1/epsilon)) + 1
        
        
        
        if n_iter + total_iter > max_iter:
            n_iter = max_iter - total_iter + 1
        
        if m=="theory":
            nb_stoch_grads = int( 1 / np.sqrt(epsilon)) 
        else:
            nb_stoch_grads = m   
    
        
        if verbose:
            output = "For epsilon = %.2e, Running for %d iterations with m = %d parallel computers" % (epsilon, n_iter, nb_stoch_grads)
            print(output)
    
        for k in range(n_iter):
            
            #Update function value lists  
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


            if k % 1000 == 0:                
                if verbose:
                    output = 'Iteration.: %d, dual gap: %.2e' % \
                         (k + total_iter, pd_gap)
                    output += ', Min dual gap: %e' % min_pd_gap_so_far
                    output += ', Min fct value: %e' % min_fxk_so_far
                    print(output)
                
            
            
            
            #proceed to iteration

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

             
        
        
        total_iter += k + 1
        x0 = xk
        yk = grad_xk # necessary ?
        if total_iter > max_iter:
            break
        epsilon *= epsilon_decrease
    
    return fxk_list, min_fxk_list, pd_gap_list, min_pd_gap_list