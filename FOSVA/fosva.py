import matplotlib.pyplot as plt
import numpy as np


def multi_fosva(eps_p_fun, eps_m_fun, alpha_fun, len_x, grad, random_point_generator, n_iterations):
    ans = [] 
    for _ in range(len_x):
        ans.append({'u': [0], 'v': [0]})
    eps_p = eps_p_fun(0)
    eps_m = eps_m_fun(0)
    alpha = alpha_fun(0)
    for it in range(n_iterations):
        print(it)
        # TAKE A RANDOM POINT:
        vec_s = random_point_generator()
        posGrad, negGrad = grad(vec_s)
        # UPDATE FUNC:
        for i in range(len_x):
            pi_p = posGrad[i]
            pi_m = negGrad[i]
            s = vec_s[i]
            u = ans[i]['u']
            nu = ans[i]['v']
            u_new, nu_new = _run_fosva_iteration(
                u, nu, s, alpha, pi_m, pi_p, eps_m, eps_p
            )
            ans[i]['u'] = u_new
            ans[i]['v'] = nu_new
        
        eps_p = eps_p_fun(it + 1) # Check, it is not correct the update.
        eps_m = eps_m_fun(it + 1) 
        alpha = alpha_fun(it + 1)

    return ans


def _run_fosva_iteration(u, nu, s, alpha, pi_m, pi_p, eps_m, eps_p):
    # DEFINE SMOOTHING INTERVAL
    u_old = [ele for ele in u]
    if s not in u:
        u.append(s)
    u.sort()
    nu = update_nu(nu, u_old, u)

    nu_new_p = (1-alpha) * np.array(nu) + alpha * np.ones_like(nu)*pi_p
    nu_new_m = (1-alpha) * np.array(nu) + alpha * np.ones_like(nu)*pi_m

    pos_s = u.index(s)
    for k in range(0, min(pos_s, len(nu)) ):
        if nu[k] < nu_new_p[k]:
            nu[k] = nu_new_p[k]
    if pos_s < len(nu):
        for k in range(pos_s, len(nu)):
            if nu[k] > nu_new_m[k]:
                nu[k] = nu_new_m[k]
    
    return u, nu


def fosva(eps_p_fun, eps_m_fun, alpha_fun, grad_p, grad_m, range_low, range_high, n_iterations):
    # INIT
    u = [0] # breakpoints
    nu = [0] # slopes
    eps_p = eps_p_fun(0)
    eps_m = eps_m_fun(0)
    alpha = alpha_fun(0)
    for i in range(n_iterations - 1):
        # COLLECT GRADIENT INFORMATION
        s = np.random.uniform(
            range_low,
            range_high
        )
        pi_p = grad_p(s)
        pi_m = grad_m(s)

        u, nu = _run_fosva_iteration(u, nu, s, alpha, pi_m, pi_p, eps_m, eps_p)        

        # At each iteration, we can apply declining step size for stability.
        eps_p = eps_p_fun(i + 1) 
        eps_m = eps_m_fun(i + 1) 
        alpha = alpha_fun(i + 1)
    return u, nu

def update_nu(nu, u_old, u_new):
    nu_new = np.zeros_like(u_new, dtype=float)
    pos_u_old = 0
    
    for i in range(len(u_new)):
        if len(u_old) <= pos_u_old:
            nu_new[i] = nu[pos_u_old - 1]
        else:
            if u_new[i] == u_old[pos_u_old]:
                nu_new[i] = nu[pos_u_old]
                pos_u_old += 1
            else:
                nu_new[i] = nu[pos_u_old - 1]
    return nu_new


def piecewise_function(x, breaks, slopes):
    coeff = [(slopes[0], 0)]
    for i in range(1, len(slopes)):
        q = coeff[-1][1] + (slopes[i-1]-slopes[i]) * breaks[i]
        coeff.append( (slopes[i], q) )
    return np.piecewise(x, [x >= b for b in breaks], [lambda x=x, m=c[0], q=c[1]: m*x + q for c in coeff])
