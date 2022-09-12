import json
import pickle
import numpy as np
from FOSVA import *

def compute_gradient(instance, inventory, prb, demand, eps):
    """ It computes right and left slopes using prb.
    In the paper prb is a TS problem.
    """
    instance.inventory = inventory
    # nominal value
    ofNom, _ , _ = prb.solve(instance, demand)
    # inizialization
    posGrad = np.zeros(instance.n_components)
    negGrad = np.zeros(instance.n_components)

    for i in range(instance.n_components):
        # inventory modification
        instance.inventory[i] += eps
        # computation
        ofTmp, _ , _ = prb.solve(instance, demand)
        # saving the result
        posGrad[i] = (ofTmp - ofNom)/eps
        # inventory modification
        instance.inventory[i] -= 2*eps # one eps to nominal, one eps for left grad
        # computation
        ofTmp, _ , _ = prb.solve(instance, demand)
        # saving the result
        negGrad[i] = (ofNom - ofTmp)/eps # notice the sign
        # back to nominal
        instance.inventory[i] += eps

    return posGrad, negGrad


def run_multifosva_ato(instance, prb, demand, step_derivative, eps_p_fun, eps_m_fun, alpha_fun, n_iterations_fosva, save_pkl=None, save_json=None):
    # create a function to compute the gradient
    grad = lambda inventory: compute_gradient(
        instance, inventory, prb, demand, step_derivative
    )
    # create a function generate a random inventory
    def random_point_generator():
        mean_demand_component = np.mean(demand, axis=1) @ instance.gozinto
        return np.random.uniform(
                0.05, 4, size=mean_demand_component.shape
            ) * mean_demand_component
    # run the multi_fosva function which computes FOSVA for multi dimensional functions
    ans = multi_fosva(
        eps_p_fun=eps_p_fun,
        eps_m_fun=eps_m_fun,
        alpha_fun=alpha_fun,
        grad=grad,
        random_point_generator=random_point_generator,
        len_x=instance.n_components,
        # len_x=instance.n_items,
        n_iterations=n_iterations_fosva
    )
    # if needed dave a pickel file
    if save_pkl:
        filehandler = open(save_pkl,"wb")
        pickle.dump(ans, filehandler)
    # if needed dave a json file
    if save_json:
        # to print the json value we define this class to print numpy array
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)
        with open('./results/ans.json', 'w') as outfile:
            json.dump(
                ans, outfile,
                cls=NumpyEncoder
            )

    return ans
