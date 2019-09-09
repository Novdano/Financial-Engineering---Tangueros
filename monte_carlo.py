import const
import math 
import numpy as np
import matplotlib.pyplot as plt
# np.standard_normal(1)

#alpha is the mean reversion rate
#theta is the long term vol avg
#phi is vol of vol
def mc_vanilla(n_sim, n_step, alpha, theta, phi, rho, s_0, sigma_0, strike, T, option=const.PUT):
    miu = const.r
    s_T = 0
    p_T = 0
    for i in range (n_sim):
        s_t = s_0
        v_t = sigma_0
        rv = 0
        d_t = T / n_step
        for j in range (n_step):
            if (v_t < 0):
                v_t = 0
            d_w_t_1 = np.random.normal(0, d_t**0.5, 1)
            d_w_t_1 = d_w_t_1[0]
            d_s_t = miu * s_t *d_t + (v_t**0.5) * s_t * d_w_t_1
            s_t += d_s_t
            d_w_t_2 = np.random.normal(0, d_t**0.5, 1)
            d_w_t_2 = d_w_t_2[0]
            #d_w_t_3= rho * d_w_t_1 + (1-rho**2)**0.5 * d_w_t_2
            d_w_t_3 = d_w_t_2
            #print(d_w_t_1, d_w_t_2)
            d_v_t = alpha * ( theta - v_t ) * d_t + phi * (v_t**0.5) * d_w_t_3
            v_t += d_v_t
        if option == const.PUT:
            payoff = strike - s_t
        elif option == const.CALL:
            payoff = s_t - strike
        if( payoff < 0 ):
            payoff = 0
        p_T += payoff
        s_T += s_t
    s_T /= n_sim
    p_T /= n_sim
    s_T *= math.exp(-const.r)
    p_T *= math.exp(-const.r)
    return s_T, p_T


def mc_df(n_sim, n_step, alpha, theta, phi, rho, s_0, sigma_0, T):
    s = np.zeros((n_sim, n_step))
    p = np.zeros((n_sim,1))
    mu = const.r
    for i in range (n_sim):
        s[i][0] = s_0
        v_t = sigma_0
        rv = 0
        s_max = const.s_0
        d_t = T / n_step
        for j in range (n_step-1):
            d_w_t_1 = np.random.normal(0, d_t**0.5, 1)
            d_s_t = mu * s[i][j] *d_t + (v_t**0.5) * s[i][j] * d_w_t_1
            s[i][j+1] = s[i][j]+d_s_t[0]
            R_t = np.log(s[i][j+1]/s[i][j])
            rv += (R_t)**2
            if(s[i][j+1] >= s_max):
                s_max = s[i][j+1]
            d_w_t_2 = np.random.normal(0, d_t**0.5, 1)
            d_v_t = alpha * ( theta - v_t) * d_t + phi * (v_t**0.5) * d_w_t_2
            v_t += d_v_t
            rv /= n_step
        drawdown = (s_max-s[i][-1])/s_max
        if (drawdown > 0.1):
            p[i] = max((rv - (const.bs_vol+0.05)**2),0) * const.notional   
    return s

mc_s = mc_df(10, 4, 0.01, 0.1, 0.02, -0.3, const.s_0, const.atm_iv_1m, 1)
plt.plot(mc_s.T)
plt.show()
