import numpy as np

def update_Ak(Ak, mu_h, beta, mu_w):
    alpha = np.sqrt(beta * mu_h)
    Ak1 = Ak * (mu_h + 2 * beta + alpha) + beta * mu_w
    Ak1 +=  np.sqrt( (beta * mu_w + mu_h * Ak)**2 + 4*Ak*(beta*beta*mu_w + Ak*mu_h*beta) + 2*beta*mu_w*Ak*alpha + Ak*Ak*alpha*alpha + 2*Ak*Ak*mu_h*alpha)
    Ak1 = Ak1 / (2*(beta + alpha))
    return Ak1