from trace_ball_utils import support_function_trace_ball, oracle_trace_ball, stochastic_oracle_trace_ball
import numpy as np
from update_Ak import update_Ak

def RPFW_least_squares_trace_ball(A,
                                  B,
                                  variance,
                                  epsilon_start,
                                  epsilon_end,
                                  epsilon_decrease,
                                  L,
                                  X0,
                                  m,
                                  max_iter,
                                  R_K=1,
                                  M_mu=1,
                                  max_norm=1,
                                  verbose=True):
    
    
    ATB = np.dot(A.T, B)
    
    fXk_list = []
    pd_gap_list = []
    min_fXk_list = []
    min_pd_gap_list = []
    
    mu_h = 1 / L
    mu_w = mu_h
    Xk = X0
    
    
    grad_Xk = np.dot(A.T, np.dot(A, Xk) - B)
    Yk = grad_Xk    
    fXk = 0.5 * np.sum((np.dot(A, Xk) - B)**2)
    

    min_fXk_so_far = fXk
    min_pd_gap_so_far = fXk + 0.5 * np.trace(np.dot((Yk + ATB).T, np.linalg.solve(np.dot(A.T, A), Yk + ATB))) - 0.5 * np.trace(np.dot(B.T, B)) + support_function_trace_ball(-Yk, max_norm=max_norm)
    
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
            fXk = 0.5 * np.sum((np.dot(A, Xk) - B)**2)       
            pd_gap = fXk + 0.5 * np.trace(np.dot((Yk + ATB).T, np.linalg.solve(np.dot(A.T, A), Yk + ATB))) - 0.5 * np.trace(np.dot(B.T, B)) + support_function_trace_ball(-Yk, max_norm=max_norm)

            if fXk < min_fXk_so_far:
                min_fXk_so_far = fXk
            if pd_gap < min_pd_gap_so_far:
                min_pd_gap_so_far = pd_gap
                
                
            fXk_list.append(fXk)
            pd_gap_list.append(pd_gap)
            min_fXk_list.append(min_fXk_so_far)
            min_pd_gap_list.append(min_pd_gap_so_far)


            if k % 1000 == 0:                
                if verbose:
                    output = 'Iteration.: %d, dual gap: %.2e' % \
                         (k + total_iter, pd_gap)
                    output += ', Min dual gap: %e' % min_pd_gap_so_far
                    output += ', Min fct value: %e' % min_fXk_so_far
                    print(output)
                
            
            
            
            #proceed to iteration

            Ak1 = update_Ak(Ak, mu_h, beta, mu_w)

            tau_k = 1 - Ak / Ak1
            vk = (1 - tau_k) * Yk + tau_k * grad_Xk

            #get m stochastic gradients
            gk = -stochastic_oracle_trace_ball(-vk, epsilon, variance=variance)
            for i in range(1, nb_stoch_grads):
                gk -= stochastic_oracle_trace_ball(-vk, epsilon, variance=variance)
            gk = gk / nb_stoch_grads


            dk = dk + (Ak1 - Ak) * gk
            Xk = beta / (beta + Ak1) * X0 - dk / (beta + Ak1)
            grad_Xk = np.dot(A.T, np.dot(A, Xk) - B)
            Yk = (1 - tau_k) * Yk + tau_k * grad_Xk
            
            #print(np.linalg.norm(Xk, "nuc"))
            Ak = Ak1

             
        
        
        total_iter += k + 1
        X0 = Xk
        Yk = grad_Xk # necessary ?
        if total_iter > max_iter:
            break
        epsilon *= epsilon_decrease
    
    return fXk_list, min_fXk_list, pd_gap_list, min_pd_gap_list