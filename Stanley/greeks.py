import const 
import copy
import math
import numpy as np
import structure_price


def delta(num_paths, alphas, K, phi, dt, t, T1, F = const.F):
    F_up = copy.deepcopy(F)
    F_down = copy.deepcopy(F)
    bp = 0.0001
    F_up[4] += bp
    F_down[4] -= bp
    price_up = structure_price.knock_in_caplet(num_paths, alphas, K, phi, dt, t, T1, F_up)
    price_down = structure_price.knock_in_caplet(num_paths, alphas, K, phi, dt, t, T1, F_down)
    d = (price_up - price_down) / 2
    #print("delta", d)
    return d

#print(delta(1000000, const.alphas, 0.0609, const.phi, 0.25, 0, 1))


def gamma(num_paths, alphas, K, phi, dt, t, T1, F = const.F):
    F_up = copy.deepcopy(F)
    F_down = copy.deepcopy(F)
    bp = 0.0001
    F_up[4] += bp
    F_down[4] -= bp
    d_up = delta(num_paths, alphas, K, phi, dt, t, T1, F_up)
    d_down = delta(num_paths, alphas, K, phi, dt, t, T1, F_down)
    g = (d_up - d_down) / 2
    print("gamma", g)
    return g
    
#print(gamma(1000, const.alphas, 0.0609, const.phi, 0.25, 0, 1))