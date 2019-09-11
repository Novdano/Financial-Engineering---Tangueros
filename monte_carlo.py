import const
import math 
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp


mc_p = []
mc_s = []
def collect_results(result):
    global mc_p
    global mc_s
    mc_s.append(result[0])
    mc_p.append(result[1])


def one_simulation(n_step, alpha, theta, phi, rho, s_0, sigma_0, strike, T, option=const.PUT):
    miu = const.r       
    s = np.zeros(n_step+1)
    s[0] = s_0
    v_t = sigma_0
    rv = 0
    d_t = T / n_step
    for j in range(n_step):
        if (v_t < 0):
            v_t = 0
        d_w_t_1 = np.random.normal(0, 1, 1)
        d_s_t = miu * s[j] *d_t + (v_t**0.5) * s[j] * d_t**0.5 * d_w_t_1
        s[j+1] = s[j]+d_s_t[0]
        d_w_t_2 = np.random.normal(0, 1, 1)
        d_w_t_3= rho * d_w_t_1 + (1-rho**2)**0.5 * d_w_t_2
        #d_w_t_3 = d_w_t_2
        #print(d_w_t_1, d_w_t_2)
        d_v_t = alpha * ( theta - v_t ) * d_t + phi * (v_t**0.5) * d_t**0.5 *d_w_t_3
        v_t += d_v_t[0]
    if option == const.PUT:
        payoff = strike - s[-1]
    elif option == const.CALL:
        payoff = s[-1] - strike
    if( payoff < 0 ):
        payoff = 0
    return s, payoff  

#alpha is the mean reversion rate
#theta is the long term vol avg
#phi is vol of vol
def mc_vanilla(n_sim, n_step, alpha, theta, phi, rho, s_0, sigma_0, strike, T, option=const.PUT):
    #initiate pool
    #pool = mp.Pool(mp.cpu_count())
    ret_s = []
    ret_p = []
    for i in range (n_sim):
        s, payoff = one_simulation(n_step, alpha, theta, phi, rho, s_0, sigma_0, strike, T, option)
        ret_s.append(s)
        ret_p.append(payoff)
        #pool.apply_async(one_simulation, args=(n_step, alpha, theta, phi, rho, s_0, sigma_0, strike, T, option), callback=collect_results)
    #pool.close()
    #pool.join()
    return ret_s, ret_p


def mc_df(n_sim, n_step, alpha, theta, phi, rho, s_0, sigma_0, T, s_max=const.s_0):
    s = np.zeros((n_sim, n_step))
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
            #jump = np.random.normal(const.jump_m,const.jump_v,1)
            #d_n_t = np.random.normal(0, 1, 1)
            #d_s_t = mu * s[i][j] *d_t + (v_t**0.5) * s[i][j] * d_t**0.5 * d_w_t_1 \
                        #+ s[i][j]*(math.exp(jump)-1)*d_n_t*d_t**0.5
            d_s_t = mu * s[i][j] *d_t + (v_t**0.5) * s[i][j] * d_t**0.5 * d_w_t_1
            s[i][j+1] = s[i][j]+d_s_t[0]
            R_t = np.log(s[i][j+1]/s[i][j])
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
    return s,p

def check_s_martingale(s):
    discounted_s = (np.sum(s,axis=0)[-1] / len(s)) * math.exp(- const.r)
    return discounted_s

def plot_mc(s):
    plt.plot(mc_s.T)
    plt.show()

def p_var(n_sim, n_step, alpha, theta, phi, rho, s_0, sigma_0, T, s_max=const.s_0):
    mc_s, mc_p = mc_df(n_sim, n_step, alpha, theta, phi, rho, s_0, sigma_0, T, s_max)
    var_low = np.quantile(mc_p,0.05)
    var_high = np.quantile(mc_p,0.95)
    mean = np.mean(mc_p)
    sd = np.std(mc_p)
    return var_low, var_high, mean, sd, mean/sd

#mc_s, mc_p = mc_df(100, 1000, 2, 0.08, 0.2, -0.5, const.s_0, const.atm_iv_1m, 1)
#mc_s, mc_p = mc_vanilla(1000, 100, 0.01, 0.1, 0.02, -0.3, const.s_0, const.atm_iv_1m, 100, 1)
#m = np.array(mc_s)
#print(np.mean(m, axis=0))
#print(np.mean(m, axis=0)[-1] * (1 + const.r/100)**(-100))
#plt.plot(mc_s.T)
#plt.show()
