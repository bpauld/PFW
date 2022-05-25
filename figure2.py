import numpy as np
import matplotlib.pyplot as plt

from update_Ak import update_Ak
from simplex_utils import oracle_simplex, stochastic_oracle_simplex, support_function_simplex
from trace_ball_utils import support_function_trace_ball, oracle_trace_ball, stochastic_oracle_trace_ball
from optimizers.FW_least_squares_simplex import FW_least_squares_simplex
from optimizers.RPFW_least_squares_simplex import RPFW_least_squares_simplex
from optimizers.FW_least_squares_trace_ball import FW_least_squares_trace_ball
from optimizers.RPFW_least_squares_trace_ball import RPFW_least_squares_trace_ball



############################################################################################




if __name__ == "__main__":
    
    # SIMPLEX EXPERIMENTS
    
    # Generate data
    seed = 43
    np.random.seed(seed)
    n = 200
    d = 50
    A = np.random.randn(n, d)
    L = np.max(np.linalg.svd(A)[1])**2
    x_star = np.random.rand(d)
    sum_entries_minimizer = 1.5
    x_star = sum_entries_minimizer * x_star / np.sum(x_star)
    b = np.dot(A, x_star)
    x0 = np.random.rand(d)
    x0 = x0 / np.sum(x0)

    


    max_iter = int(1e5)
    lists_FW_simplex = FW_least_squares_simplex(A, b, x0, max_iter)

    M_mu_theory = np.sqrt(d)
    M_mu_1 = 1
    epsilon_start = 1
    epsilon_end = 0
    epsilon_decrease = 0.5

    lists_AFW_1_simplex_M_theory = RPFW_least_squares_simplex(A,
                                                              b,
                                                              epsilon_start=epsilon_start,
                                                              epsilon_end=epsilon_end,
                                                              epsilon_decrease=epsilon_decrease,
                                                              L=L,
                                                              x0=x0,
                                                              m=1,
                                                              max_iter=max_iter,
                                                              R_K=1,
                                                              M_mu=M_mu_theory,
                                                              verbose=True)


    lists_AFW_theory_simplex_M_theory = RPFW_least_squares_simplex(A,
                                                                   b,
                                                                   epsilon_start=epsilon_start,
                                                                   epsilon_end=epsilon_end,
                                                                   epsilon_decrease=epsilon_decrease,
                                                                   L=L,
                                                                   x0=x0,
                                                                   m="theory",
                                                                   max_iter=max_iter,
                                                                   R_K=1,
                                                                   M_mu=M_mu_theory,
                                                                   verbose=True)

    lists_AFW_1_simplex_M_1 = RPFW_least_squares_simplex(A,
                                                         b,
                                                         epsilon_start=epsilon_start,
                                                         epsilon_end=epsilon_end,
                                                         epsilon_decrease=epsilon_decrease,
                                                         L=L,
                                                         x0=x0,
                                                         m=1,
                                                         max_iter=max_iter,
                                                         R_K=1,
                                                         M_mu=M_mu_1,
                                                         verbose=True)


    lists_AFW_theory_simplex_M_1 = RPFW_least_squares_simplex(A,
                                                              b,
                                                              epsilon_start=epsilon_start,
                                                              epsilon_end=epsilon_end,
                                                              epsilon_decrease=epsilon_decrease,
                                                              L=L,
                                                              x0=x0,
                                                              m="theory",
                                                              max_iter=max_iter,
                                                              R_K=1,
                                                              M_mu=M_mu_1,
                                                              verbose=True)
    
    
    # TRACE NORM BALL EXPERIMENTS
    # Generate data
    seed = 43
    np.random.seed(seed)
    p = 10
    q = 8
    A = np.random.randn(p, p)
    L = np.linalg.norm(np.dot(A.T, A), 'fro')
    X_star = np.random.randn(p, q)
    norm_global_minimizer = 1.5
    X_star = norm_global_minimizer * X_star / np.linalg.norm(X_star, 'nuc')

    B = np.dot(A, X_star)

    X0 = np.random.randn(p, q)
    X0 = 0.5 * X0 / np.linalg.norm(X0, 'nuc')
    
    
    max_iter =  int(1e5)
    M_mu_theory = np.sqrt(p*q)
    M_mu_1 = 1
    epsilon_start = 1
    epsilon_end = 0
    epsilon_decrease = 0.5

    lists_FW_matrix = FW_least_squares_trace_ball(A=A, B=B, X0=X0, n_iter=max_iter, max_norm=1)


    lists_AFW_1_matrix_M_theory = RPFW_least_squares_trace_ball(A=A,
                                                                variance=1,
                                                                B=B,
                                                                epsilon_start=epsilon_start,
                                                                epsilon_end=epsilon_end,
                                                                epsilon_decrease=epsilon_decrease,
                                                                L=L,
                                                                X0=X0,
                                                                m=1,
                                                                max_iter=max_iter,
                                                                R_K=1,
                                                                M_mu=M_mu_theory,
                                                                max_norm=1,
                                                                verbose=True)

    lists_AFW_theory_matrix_M_theory = RPFW_least_squares_trace_ball(A=A,
                                                                     variance=1,
                                                                     B=B,
                                                                     epsilon_start=epsilon_start,
                                                                     epsilon_end=epsilon_end,
                                                                     epsilon_decrease=epsilon_decrease,
                                                                     L=L,
                                                                     X0=X0,
                                                                     m="theory",
                                                                     max_iter=max_iter,
                                                                     R_K=1,
                                                                     M_mu=M_mu_theory,
                                                                     max_norm=1,
                                                                     verbose=True)
    


    lists_AFW_1_matrix_M_1 = RPFW_least_squares_trace_ball(A=A,
                                                           variance=1,
                                                           B=B,
                                                           epsilon_start=epsilon_start,
                                                           epsilon_end=epsilon_end,
                                                           epsilon_decrease=epsilon_decrease,
                                                           L=L,
                                                           X0=X0,
                                                           m=1,
                                                           max_iter=max_iter,
                                                           R_K=1,
                                                           M_mu=M_mu_1,
                                                           max_norm=1,
                                                           verbose=True)

    lists_AFW_theory_matrix_M_1 = RPFW_least_squares_trace_ball(A=A,
                                                                variance=1,
                                                                B=B,
                                                                epsilon_start=epsilon_start,
                                                                epsilon_end=epsilon_end,
                                                                epsilon_decrease=epsilon_decrease,
                                                                L=L,
                                                                X0=X0,
                                                                m="theory",
                                                                max_iter=max_iter,
                                                                R_K=1,
                                                                M_mu=M_mu_1,
                                                                max_norm=1,
                                                                verbose=True)
    
    
    
    
    # Plot experiments
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(30,6))

    ax1.yaxis.grid(color='gray', linewidth=0.3)
    ax1.xaxis.grid(color='gray', linewidth=0.3)
    ax2.yaxis.grid(color='gray', linewidth=0.3)
    ax2.xaxis.grid(color='gray', linewidth=0.3)
    ax3.yaxis.grid(color='gray', linewidth=0.3)
    ax3.xaxis.grid(color='gray', linewidth=0.3)
    ax4.yaxis.grid(color='gray', linewidth=0.3)
    ax4.xaxis.grid(color='gray', linewidth=0.3)



    l1 = ax1.loglog(lists_FW_simplex[-1], color="orange")
    l2 = ax1.loglog(lists_AFW_1_simplex_M_theory[-1], color="red")
    l3 = ax1.loglog(lists_AFW_theory_simplex_M_theory[-1], color="blue")
    ax1.set_ylabel("Minimum dual gap")
    ax1.set_xlabel("Iteration")
    ax1.set_title(r"(a) $M = \sqrt{d}$")


    l4 = ax2.loglog(lists_FW_simplex[-1], color="orange")
    l5 = ax2.loglog(lists_AFW_1_simplex_M_1[-1], color="red")
    l6 = ax2.loglog(lists_AFW_theory_simplex_M_1[-1], color="blue")
    ax2.set_xlabel("Iteration")
    ax2.set_title(r"(b) $M = 1$")

    l7 = ax3.loglog(lists_FW_matrix[-1], color="orange")
    l8 = ax3.loglog(lists_AFW_1_matrix_M_theory[-1], color="red")
    l9 = ax3.loglog(lists_AFW_theory_matrix_M_theory[-1], color="blue")
    ax3.set_xlabel("Iteration")
    ax3.set_title(r"(c) $M = \sqrt{pq}$")


    l10 = ax4.loglog(lists_FW_matrix[-1], color="orange")
    l11 = ax4.loglog(lists_AFW_1_matrix_M_1[-1], color="red")
    l12 = ax4.loglog(lists_AFW_theory_matrix_M_1[-1], color="blue")
    ax4.set_xlabel("Iteration")
    ax4.set_title(r"(d) $M = 1$")
    line_labels = ["Frank-Wolfe", r"Algorithm 3 with $m_\alpha=1$", r"Algorithm 3 with $m_\alpha=\frac{1}{\sqrt{\alpha}}$"]
    fig.legend([l1, l2, l3],     # The line objects
               labels=line_labels,   # The labels for each line
               #loc="best",   # Position of legend
               loc="lower center",
               borderaxespad=0.1,    # Small spacing around legend box
               ncol=len(line_labels),
               fontsize="large"
               )
    plt.show()


    