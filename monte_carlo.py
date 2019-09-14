import const
import math 
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import pandas

N_SIM = 100
N_STEP = 100
ALPHA = 10.97858327
THETA = 0.027225000000000003
PHI = 0.01362476
RHO = -0.55156066
S_0 = const.s_0
SIGMA_0 = const.atm_iv_1m
TIME = 1

#alpha is the mean reversion rate
#theta is the long term vol avg
#phi is vol of vol
def mc_vanilla(n_sim, n_step, alpha, theta, phi, rho, s_0, sigma_0, strike, T, option=const.CALL):
    miu = const.r
    s = np.zeros((n_sim, n_step))
    p = np.zeros((n_sim,1))
    for i in range (n_sim):
        s[i][0] = s_0
        v_t = sigma_0**2
        rv = 0
        d_t = T / n_step
        for j in range(n_step-1):
            if (v_t < 0):
                v_t = 0
            d_w_t_1 = np.random.normal(0, 1, 1)
            d_s_t = miu * s[i][j] *d_t + (v_t**0.5) * s[i][j] * d_t**0.5 *d_w_t_1
            s[i][j+1] = s[i][j]+d_s_t[0]
            d_w_t_2 = np.random.normal(0, 1, 1)
            d_w_t_3= rho * d_w_t_1 + (1-rho**2)**0.5 * d_w_t_2
            #d_w_t_3 = d_w_t_2
            #print(d_w_t_1, d_w_t_2)
            d_v_t = alpha * ( theta - v_t ) * d_t + phi * (v_t**0.5) * d_t**0.5 * d_w_t_3
            v_t += d_v_t
        if option == const.PUT:
            payoff = strike - s[i][-1]
        elif option == const.CALL:
            payoff = s[i][-1] - strike
        if( payoff < 0 ):
            payoff = 0
        p[i] = payoff
    return s, p


def mc_df(n_sim, n_step, alpha, theta, phi, rho, s_0, sigma_0, T, s_max=const.s_0):
    s = np.zeros((n_sim, n_step))
    p = np.zeros((n_sim,1))
    mu = const.r
    s_max_list = np.zeros((n_sim,1))
    for i in range (n_sim):
        s[i][0] = s_0
        v_t = sigma_0**2
        rv = 0
        d_t = T / n_step
        s_max = s_0
        for j in range (n_step-1):
            # if (v_t < 0):
            #     v_t = 0
            d_w_t_1 = np.random.normal(0, 1, 1)
            #jump = np.random.normal(const.jump_m,const.jump_v,1)
            #d_n_t = np.random.normal(0, 1, 1)
            #d_s_t = mu * s[i][j] *d_t + (v_t**0.5) * s[i][j] * d_t**0.5 * d_w_t_1 \
                        #+ s[i][j]*(math.exp(jump)-1)*d_n_t*d_t**0.5
            d_s_t = mu * s[i][j] * d_t + (v_t**0.5) * s[i][j] * (d_t**0.5) * d_w_t_1
            s[i][j+1] = s[i][j]+d_s_t[0]
            R_t = np.log(s[i][j+1]/s[i][j])
            rv += (R_t)**2
            if(s[i][j+1] >= s_max):
                s_max = s[i][j+1]
            d_w_t_2 = np.random.normal(0, 1, 1)
            d_w_t_3= rho * d_w_t_1 + (1-rho**2)**0.5 * d_w_t_2
            d_v_t = alpha * (theta - v_t) * d_t + phi * (v_t**0.5) * (d_t**0.5) * d_w_t_3
            v_t += d_v_t
        print(i, s[i][-1], s_max, (s_max-s[i][-1])/s_max, math.sqrt(rv))
        s_max_list[i] = s_max
        drawdown = (s_max-s[i][-1])/s_max
        #print(drawdown)
        if (drawdown > 0.1):
            p[i] = max((math.sqrt(rv) - (const.bs_vol+0.05)),0) * const.notional 
    print(np.count_nonzero(p)/n_sim)  
    return s, p

def mc_report(n_sim, n_step, alpha, theta, phi, rho, s_0, sigma_0, T, s_max=const.s_0):
    s = np.zeros((n_sim, n_step))
    p = np.zeros((n_sim,1))
    mu = const.r
    v_T = np.zeros((n_sim,1))
    rv_T = np.zeros((n_sim,1))
    for i in range (n_sim):
        s[i][0] = s_0
        v_t = sigma_0**2
        rv = 0
        d_t = T / n_step
        s_max = s_0
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
        v_T[i] = v_t
        rv_T = rv
        drawdown = (s_max-s[i][-1])/s_max
        #print(drawdown)
        if (drawdown > 0.1):
            p[i] = max((math.sqrt(rv) - (const.bs_vol+0.05)),0) * const.notional 
    s_T = s[:,-1]
    print('simulated vol at T, annualized realized vol, stock at T')
    print("mean: ",np.mean(v_T), np.mean(rv_T), np.mean(s_T))
    print("sd: ",np.stx(v_T), np.std(rv_T), np.std(s_T))

    return v_T, rv_T, s_T

def check_s_martingale(s):
    discounted_s = np.sum(s[:,-1] )/len(s[:,-1] ) * math.exp(-const.r)
    return discounted_s

def plot_p(p, filename):
    plt.hist(p, 50,facecolor='g')
    plt.ylabel('Frequency')
    plt.xlabel("Payoff")
    plt.grid(True)
    plt.title("Histogram of Structure Payoff")
    plt.savefig(filename)
    plt.show()

def plot(s):
    plt.plot(s.T)
    plt.show()

#returns the 5% VaR and 95% VaR
def p_var(n_sim, n_step, alpha, theta, phi, rho, s_0, sigma_0, T, s_max=const.s_0):
    mc_s, mc_p = mc_df(n_sim, n_step, alpha, theta, phi, rho, s_0, sigma_0, T, s_max)
    var_low = np.quantile(mc_p,0.05)
    var_high = np.quantile(mc_p,0.95)
    mean = np.mean(mc_p)
    sd = np.std(mc_p)
    return const.charge - var_low, const.charge - var_high, mean - const.charge

#(0.15293256546944, 0.07980132902222221, -0.11260251422444387)
#There's 5% chance that client may lose $0.1529 per 1$
#There's a 95% chance that the client may lose $0.07980 oper 1$


# print("running...")
# mc_s, mc_p = mc_df(10000, 10000, ALPHA, THETA, PHI, RHO, const.s_0, const.atm_iv_1m, 1)
# np.save("mc_s_10000_10000",mc_s)
# np.save("mc_p_10000_10000",mc_p)
# print("saved")

#s100, p100, max100 = mc_df(100, 100, ALPHA, THETA, PHI, RHO, const.s_0, const.atm_iv_1m, 1)
#s1000,p1000, max1000 = mc_df(100, 1000, ALPHA, THETA, PHI, RHO, const.s_0, const.atm_iv_1m, 1)
s1000,p1000 = mc_df(1000, 10000, ALPHA, THETA, PHI, RHO, const.s_0, const.atm_iv_1m, 1)







