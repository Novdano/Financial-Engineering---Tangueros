import monte_carlo as mc
import const
import math
import numpy as np

#1 dollar shock to stock
def delta(n_sim, n_step, alpha, theta, phi, rho, s_t, sigma_t, t, T):
    n_step_left = math.floor((T-t) * n_step / T)
    mc_s_up, mc_p_up = mc.mc_df (n_sim, n_step_left, alpha, theta, phi, rho, s_t + 0.01, sigma_t, T-t)
    mc_s_down, mc_p_down = mc.mc_df (n_sim, n_step_left, alpha, theta, phi, rho, s_t - 0.01, sigma_t, T-t)
    r = const.r
    res = math.exp(-r*(T-t)) * np.mean(mc_p_up) - math.exp(-r*(T-t)) * np.mean(mc_p_down)
    std_error = math.sqrt(np.var(mc_p_up) + np.var(mc_p_down))
    #print(math.exp(-r*(T-t)) * np.mean(mc_p_up), math.exp(-r*(T-t)) * np.mean(mc_p_down), std_error)
    return res/(2 * 0.01)

#1 vol point shock
def vega(n_sim, n_step, alpha, theta, phi, rho, s_t, sigma_t, t, T):
    n_step_left = math.floor((T-t) * n_step / T)
    mc_s_up, mc_p_up = mc.mc_df (n_sim, n_step_left, alpha, theta, phi, rho, s_t, sigma_t + 0.5, T-t)
    mc_s_down, mc_p_down = mc.mc_df (n_sim, n_step_left, alpha, theta, phi, rho, s_t, sigma_t - 0.5, T-t)
    r = const.r
    res = math.exp(-r*(T-t)) * np.mean(mc_p_up) - math.exp(-r*(T-t)) * np.mean(mc_p_down)
    std_error = math.sqrt(np.var(mc_p_up) + np.var(mc_p_down))
    #print(math.exp(-r*(T-t)) * np.mean(mc_p_up), math.exp(-r*(T-t)) * np.mean(mc_p_down), std_error)
    return res


def gamma(n_sim, n_step, alpha, theta, phi, rho, s_t, sigma_t, t, T):
    n_step_left = math.floor((T-t) * n_step / T)
    delta_1 = delta(n_sim, n_step_left, alpha, theta, phi, rho, s_t+0.01, sigma_t, t, T)
    delta_2 = delta(n_sim, n_step_left, alpha, theta, phi, rho, s_t-0.01, sigma_t, t, T)
    return (delta_1 - delta_2)/(2*0.01)


N_SIM = 1000
N_STEP = 1000   
ALPHA = 10.97858327
THETA = 0.027225000000000003
PHI = 0.01362476
RHO = -0.55156066

#print(gamma(N_SIM, N_STEP, ALPHA, THETA, PHI, RHO, 100, 0.11, 0, 1))
#print(delta(N_SIM, N_STEP, ALPHA, THETA, PHI, RHO, 100, 0.11, 0, 1))
#print(vega(N_SIM, N_STEP, ALPHA, THETA, PHI, RHO, 100, 0.11, 0, 1))
