import matplotlib.pyplot as plt
import numpy as np


def multi_fosva(alpha_fun, len_x, grad, random_point_generator, n_iterations):
    """Multidimensional FOSVA
    """
    # Initialize
    ans = [] 
    for _ in range(len_x):
        ans.append({'u': [0], 'v': [0]})
    alpha = alpha_fun(0)
    # for each iteration
    for it in range(n_iterations):
        # take a random point
        vec_s = random_point_generator()
        # evaluate the right and left slope in that point
        posGrad, negGrad = grad(vec_s)
        # update the values in the vectors:
        for i in range(len_x):
            pi_p = posGrad[i]
            pi_m = negGrad[i]
            s = vec_s[i]
            u = ans[i]['u']
            nu = ans[i]['v']
            u_new, nu_new = _run_fosva_iteration(
                u, nu, s, alpha, pi_m, pi_p
            )
            ans[i]['u'] = u_new
            ans[i]['v'] = nu_new
        # At each iteration, we can apply declining step size for stability.
        alpha = alpha_fun(it + 1)

    return ans


def fosva(alpha_fun, grad_p, grad_m, range_low, range_high, n_iterations):
    """One dimensional FOSVA
    """
    u = [0] # breakpoints
    nu = [0] # slopes
    alpha = alpha_fun(0)
    for i in range(n_iterations - 1):
        # generate a random point
        s = np.random.uniform(
            range_low,
            range_high
        )
        # evaluate the right and left slope in that point
        pi_p = grad_p(s)
        pi_m = grad_m(s)
        u, nu = _run_fosva_iteration(u, nu, s, alpha, pi_m, pi_p)
        # At each iteration, we can apply declining step size for stability.
        alpha = alpha_fun(i + 1)
    return u, nu


def _run_fosva_iteration(u, nu, s, alpha, pi_m, pi_p):
    # duplicate the array u
    u_old = [ele for ele in u]
    # add the point s if it is not in the array u
    if s not in u:
        u.append(s)
    # sort the u array
    u.sort()
    # update accordingly the vector of slopes with a new component
    nu = update_nu(nu, u_old, u)
    # Update the value of the slope:
    nu_new_p = (1-alpha) * np.array(nu) + alpha * np.ones_like(nu) * pi_p
    nu_new_m = (1-alpha) * np.array(nu) + alpha * np.ones_like(nu) * pi_m
    # Update the array of slopes nu in order to have a decreasing vector
    # start by computing the pos of the new point s in the vector u
    pos_s = u.index(s)
    # update all the values to the right of pos_s if smaller nu_new_p[k]
    for k in range(0, min(pos_s, len(nu)) ):
        if nu[k] < nu_new_p[k]:
            nu[k] = nu_new_p[k]
    # if the new point was not in the last position
    if pos_s < len(nu):
        # update all the values to the left of pos_s if greter than nu_new_m[k]
        for k in range(pos_s, len(nu)):
            if nu[k] > nu_new_m[k]:
                nu[k] = nu_new_m[k]
    return u, nu


def update_nu(nu, u_old, u_new):
    """
    It return the vector of slopes updated, i.e. with one new component,
    """
    # create a new vector of slopes
    nu_new = np.zeros_like(u_new, dtype=float)
    pos_u_old = 0
    # for each position in u_new
    for i in range(len(u_new)):
        if len(u_old) <= pos_u_old:
            # if the position is the last, nu_new must be equal to the value if the previous position
            nu_new[i] = nu[pos_u_old - 1]
        else:
            # if  u_new[i] == u_old[pos_u_old] then the array u has not been modified
            if u_new[i] == u_old[pos_u_old]:
                # then, nu_new must be equal to nu
                nu_new[i] = nu[pos_u_old]
                # we can consider a new position
                pos_u_old += 1
            else:
                # otherwise, nu_new must be equal to the value if the previous position
                nu_new[i] = nu[pos_u_old - 1]
    return nu_new


def piecewise_function(x, breaks, slopes):
    """It evaluates the piecewise linear function
    (described by a set of breaks and slopes) in the point x.
    """
    coeff = [(slopes[0], 0)]
    for i in range(1, len(slopes)):
        q = coeff[-1][1] + (slopes[i-1]-slopes[i]) * breaks[i]
        coeff.append( (slopes[i], q) )
    return np.piecewise(x, [x >= b for b in breaks], [lambda x=x, m=c[0], q=c[1]: m*x + q for c in coeff])
