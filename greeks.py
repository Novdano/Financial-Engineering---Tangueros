import monte_carlo as mc
import const
import math
import numpy as np

def delta(n_sim, n_step, alpha, theta, phi, rho, s_t, sigma_t, t, T):
    n_step_left = math.floor((T-t) * n_step / T)
    mc_s_1, mc_p_1 = mc.mc_df (n_sim, n_step, alpha, theta, phi, rho, s_t-0.5, sigma_t, T-t)
    mc_s_2, mc_p_2 = mc.mc_df (n_sim, n_step, alpha, theta, phi, rho, s_t+0.5, sigma_t, T-t)
    r = const.r
    res = math.exp(-r*(T-t)) * np.sum(mc_p_1) - math.exp(-r*(T-t)) * np.sum(mc_p_2)
    return res

def vega(n_sim, n_step, alpha, theta, phi, rho, s_t, sigma_t, t, T):
    n_step_left = math.floor((T-t) * n_step / T)
    mc_s_1, mc_p_1 = mc.mc_df (n_sim, n_step, alpha, theta, phi, rho, s_t, sigma_t*1.005, T-t)
    mc_s_2, mc_p_2 = mc.mc_df (n_sim, n_step, alpha, theta, phi, rho, s_t, sigma_t*0.995, T-t)
    r = const.r
    res = math.exp(-r*(T-t)) * np.sum(mc_p_1) - math.exp(-r*(T-t)) * np.sum(mc_p_2)
    return res