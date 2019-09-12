import numpy as np 
from scipy.optimize import minimize
import math
import const
import monte_carlo as mc 
from random import random

def var_swap_replication(n_sim, n_step, alpha, theta, phi, rho, s_0, sigma_0, T):
    strike_percentage =  np.array([50,55,60,65,70,75,80,120,125,130,135,140,145,150]) * 0.01
    weight = np.array([20.00, 16.53, 13.89, 11.83,10.20, 8.89,7.81, 3.47, 3.20, 2.96,2.74,2.55,2.38,2.22]) * 0.01 #(10,)
    forward = s_0 * math.exp(const.r * T)
    print(len(strike_percentage))
    n_opt = len(strike_percentage)
    p_0 = np.zeros((n_opt,))
    for i in range (len(p_0)):
        strike_i = strike_percentage[i] * forward
        is_call = 1
        if i <= 5:
            is_call = 0
        s, p = mc.mc_vanilla(n_sim, n_step, alpha, theta, phi, rho, forward, sigma_0, strike_i, T, is_call)
        p_0[i] = np.mean(p) * math.exp(-const.r * T)
    p_0_percentage = p_0 / forward
    replication_price = np.sum(p_0_percentage * weight) * 2 / T
    var_swap_strike = math.sqrt(replication_price / math.exp(-const.r * T))
    return replication_price, var_swap_strike

#var_swap_strike = var_swap_replication(100, 100, 2, 0.08, 0.2, -0.5, const.s_0, const.atm_iv_1m, 1)
var_swap_strike = 0.265604

def mc_portfolio(n_sim, n_step, alpha, theta, phi, rho, s_0, sigma_0, T, s_max=const.s_0):
    s = np.zeros((n_sim, n_step))
    var_swap_T = np.zeros((n_sim,1))
    p = np.zeros((n_sim,1))
    mu = const.r
    for i in range (n_sim):
        s[i][0] = s_0
        v_t = sigma_0
        rv = 0
        d_t = T / n_step
        for j in range (n_step-1):
            if (v_t < 0):
                v_t = 0
            d_w_t_1 = np.random.normal(0, 1, 1)
            d_s_t = mu * s[i][j] *d_t + (v_t**0.5) * s[i][j] * d_t**0.5 * d_w_t_1
            s[i][j+1] = max(s[i][j]+d_s_t[0],1)
            R_t = math.log(s[i][j+1]/s[i][j])
            rv += (R_t)**2
            if(s[i][j+1] >= s_max):
                s_max = s[i][j+1]
            d_w_t_2 = np.random.normal(0, 1, 1)
            d_w_t_3= rho * d_w_t_1 + (1-rho**2)**0.5 * d_w_t_2
            d_v_t = alpha * ( theta - v_t) * d_t + phi * (v_t**0.5) * d_t**0.5 * d_w_t_3
            v_t += d_v_t
        drawdown = (s_max-s[i][-1])/s_max
        if (drawdown > 0.1):
            p[i] = max((rv - (const.bs_vol+0.05)**2),0) * const.notional
        var_swap_T[i] = (rv - var_swap_strike**2)
    s_T = s[:,-1].reshape(-1,1)
    return s_T, p, var_swap_T



def portfolio_var(n_stock, n_var_swap,n_sim, n_step, alpha, theta, phi, rho, s_0, sigma_0, T, s_max=const.s_0):
    s_T, p, var_swap_T = mc_portfolio(n_sim, n_step, alpha, theta, phi, rho, s_0, sigma_0, T, s_max=const.s_0)
    #print("s_T: ",s_T,"p:", p,"var_swap_T", var_swap_T)
    return np.var(n_stock * s_T + n_var_swap * var_swap_T - p)

def portfolio_return(n_stock, n_var_swap,n_sim, n_step, alpha, theta, phi, rho, s_0, sigma_0, T, s_max=const.s_0):
    s_T, p, var_swap_T = mc_portfolio(n_sim, n_step, alpha, theta, phi, rho, s_0, sigma_0, T, s_max=const.s_0)
    return np.mean(n_stock * s_T + n_var_swap * var_swap_T - p)

def minimize_var(n_stock, n_var_swap, n_sim, n_step, alpha, theta, phi, rho, s_0, sigma_0, T, s_max=const.s_0):
    constraints = { "type": "eq", 'fun': portfolio_return, 'args': (n_sim, n_step, alpha, theta, phi, rho, s_0, sigma_0, T, s_max, )  }
    ret = minimize( portfolio_var, [n_stock, n_var_swap], args=(n_sim, n_step, alpha, theta, phi, rho, s_0, sigma_0, T, s_max,),
        method="SLSQP", constraints = constraints, bounds = [(-1,0.5), (-1,0.5)] )
    print(ret)
    return ret.x

#############################################################################
# Simulated Annealing
#############################################################################
N_SIM = 100
N_STEP = 100
ALPHA = 10.97858327
THETA = 0.12214962
PHI = 0.00001
RHO = -0.55156066
S_0 = const.s_0
SIGMA_0 = const.atm_iv_1m
TIME = 1

TARGET_RETURN = 0

def acceptence_probability(old_var, new_var, T):
    return math.exp((old_var-new_var)/T)


def neighbor(n_stock, n_var_swap):
    sd_s = 0.1
    sd_v = 0.1
    new_n_stock = np.random.normal(n_stock,sd_s,1)[0]
    new_n_var_swap = np.random.normal(n_var_swap,sd_v,1)[0]
    mean = portfolio_return(n_stock, n_var_swap, N_SIM, N_STEP, ALPHA, THETA, PHI, RHO, S_0, SIGMA_0, TIME)
    while(mean < 0):
        sd_s /= 2
        sd_v /=2
        new_n_stock = np.random.normal(n_stock,sd_s,1)[0]
        new_n_var_swap = np.random.normal(n_var_swap,sd_v,1)[0]
        mean = portfolio_return(new_n_stock, new_n_var_swap, N_SIM, N_STEP, ALPHA, THETA, PHI, RHO, S_0, SIGMA_0, TIME)
    return new_n_stock, new_n_var_swap

def sim_anneal(n_stock, n_var_swap,n_sim, n_step, alpha, theta, phi, rho, s_0, sigma_0, T, s_max=const.s_0):
    old_var = portfolio_var(n_stock, n_var_swap,n_sim, n_step, alpha, theta, phi, rho, s_0, sigma_0, T, s_max=const.s_0)
    T = 1
    T_min = 10**(-4)
    a = 0.9
    while(T > T_min):
        for i in range (20):
            new_n_stock, new_n_var_swap = neighbor(n_stock, n_var_swap)
            new_var = portfolio_var(new_n_stock, new_n_var_swap,n_sim, n_step, alpha, theta, phi, rho, s_0, sigma_0, T)
            accept_prob = acceptence_probability(old_var, new_var, T)
            if(accept_prob > random()):
                n_stock, n_var_swap = new_n_stock, new_n_var_swap
                old_var = new_var
                print(n_stock, n_var_swap, old_var)
        T = T * a
    return n_stock, n_var_swap, old_var





# def bs_v_0(s_0,k,sigma,T):
#         d1 = (log(s_0/k) + (r + sigma^2/2)*T)/(sigma * sqrt(T))
#         d2 = (log(S_0/k) + (r - sigma^2/2)*T)/(sigma * sqrt(T)) 
#         price = S_0 * pnorm(d1) - K*pnorm(d2) * exp(-r * T)
#     return 

# def v_swap_strike(s_0,k, v_0, T):
#     sigma_lower = 0
#     sigma_upper = 1
#     sigma = sigma_lower
#     v_1 = bs_v_0(s_0,k,sigma_lower,T)
#     v_2 = bs_v_0(s_0,k,sigma_upper,T)
#     diff = v_1 - v_0
#     while(abs(diff)>0.000001):
#         sigma = 0.5*(sigma_lower + sigma_upper)
#         V = bs_v_0(s_0,k,sigma,T)
#         diff = V - v_0
#         if(V>v_0):
#             sigma_upper = sigma
#         else:
#             sigma_lower = sigma
#     return sigma

#var_swap_replication(100, 100, 2, 0.08, 0.2, -0.5, const.s_0, const.atm_iv_1m, 1)

# def optimize(prices, symbols, target_return=0.1):
#     normalized_prices = prices / prices.ix[0, :]
#     init_guess = np.ones(len(symbols)) * (1.0 / len(symbols))
#     bounds = ((0.0, 1.0),) * len(symbols)
#     weights = minimize(get_portfolio_risk, init_guess,
#                        args=(normalized_prices,), method='SLSQP',
#                        options={'disp': False},
#                        constraints=({'type': 'eq', 'fun': lambda inputs: 1.0 - np.sum(inputs)},
#                                     {'type': 'eq', 'args': (normalized_prices,),
#                                      'fun': lambda inputs, normalized_prices:
#                                      target_return - get_portfolio_return(weights=inputs,
#                                                                           normalized_prices=normalized_prices)}),
#                        bounds=bounds)
#     return weights.x