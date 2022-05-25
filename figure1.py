import numpy as np
import matplotlib.pyplot as plt
from optimizers.PFW_least_squares_simplex import PFW_least_squares_simplex
from simplex_utils import oracle_simplex
  
    
#########################################################################

if __name__ == "__main__":
    
    # Generate data
    seed = 42
    np.random.seed(seed)
    n = 200
    d = 50
    A = np.random.randn(n, d)
    L = np.max(np.linalg.svd(A)[1])**2
    x_star = np.random.rand(d)
    sum_entries_minimizer = 1.5
    x_star = sum_entries_minimizer * x_star / np.sum(x_star)
    b = np.dot(A, x_star)

    #######################################################################


    # run Frank-Wolfe to obtain approximate solution
    x0 = np.random.rand(d)
    x0 = x0 / np.sum(x0)
    n_iter_fw = 10000
    xk = x0

    fxk = 0.5 * np.linalg.norm(np.dot(A, xk) - b)**2
    for k in range(n_iter_fw):
        grad_xk = np.dot(A.T, np.dot(A, xk) - b)

        sk = oracle_simplex(grad_xk)

        eta_k = 2 / (k+1)

        xk = (1 - eta_k) * xk + eta_k * sk

    x_sol = xk

    
    
    # run AFW for different values of epsilon and different values of m

    dic_lists = {}
    M_mu = np.sqrt(d)

    for epsilon in [1e-2, 1e-3]:
        for m in [1, "theory"]:
            lists = PFW_least_squares_simplex(A,
                                              b,
                                              epsilon=epsilon,
                                              L=L,
                                              x0=x0,
                                              m=m,
                                              n_iter=100000,
                                              R_K=1,
                                              M_mu=M_mu,
                                              x_optimal=x_sol)

            dic_lists["lists_" + str(epsilon) + "_" + str(m)] = lists



    ########################################################################

    # plot figures

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,4.5))
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(color='gray', linewidth=0.3)
    ax1.xaxis.grid(color='gray', linewidth=0.3)
    ax2.yaxis.grid(color='gray', linewidth=0.3)
    ax2.xaxis.grid(color='gray', linewidth=0.3)


    l1 = ax1.loglog(dic_lists["lists_0.01_1"][-2], color="red")
    l2 = ax1.loglog(dic_lists["lists_0.01_1"][-1], color="red", linestyle="dotted")
    l3 = ax1.loglog(dic_lists["lists_0.01_theory"][-2], color="blue")
    l4 = ax1.loglog(dic_lists["lists_0.01_theory"][-1], color="blue", linestyle="dotted")
    ax1.set_ylabel("Minimum dual gap")
    ax1.set_xlabel("Iteration")

    l5 = ax2.loglog(dic_lists["lists_0.001_1"][-2], color="red")
    l6 = ax2.loglog(dic_lists["lists_0.001_1"][-1], color="red", linestyle="dotted")
    l7 = ax2.loglog(dic_lists["lists_0.001_theory"][-2], color="blue")
    l8 = ax2.loglog(dic_lists["lists_0.001_theory"][-1], color="blue", linestyle="dotted")
    ax2.set_ylabel("Minimum dual gap")
    ax2.set_xlabel("Iteration")
    line_labels = [r"Algorithm 2 with $m=1$", r"Upper Bound for $m=1$",
                   r"Algorithm 2 with $m=\frac{1}{\sqrt{\alpha}}$", r"Upper Bound for $m=\frac{1}{\sqrt{\alpha}}$"]
    fig.legend([l1, l2, l3, l4],     # The line objects
               labels=line_labels,   # The labels for each line
               loc="upper center",   # Position of legend
               #loc="center right",
               borderaxespad=0.1,    # Small spacing around legend box
               ncol=len(line_labels),
               fontsize="large"
               )

    plt.savefig("test.pdf")
    plt.show()