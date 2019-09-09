import const as con
import numpy as np
import matplotlib.pyplot as plt
# np.standard_normal(1)

#alpha is the mean reversion rate
#theta is the long term vol avg
#phi is vol of vol
'''def mc(n_sim, n_step, alpha, theta, phi):
    miu = con.r
    s_T = 0
    p_T = 0
    for i in range (n_sim):
        s_t = con.s_0
        v_t = con.atm_iv_1m
        rv = 0
        s_max = con.s_0
        s_min = con.s_0
        drawdown = 0
        d_t = 1 / n_step
        for j in range (n_step):
            d_w_t_1 = np.random.normal(0, d_t**0.5, 1)
            d_s_t = miu * s_t *d_t + (v_t**0.5) * s_t * d_w_t_1
            s_t_1 = s_t
            s_t += d_s_t[0]
            R_t = np.log(s_t/s_t_1)
            rv += (R_t)**2
            print(s_t_1, s_t)
            if(s_t >= s_max):
                cur_drawdown = (s_max - s_min) / s_max
                print(s_max, s_min)
                print(cur_drawdown)
                if (cur_drawdown > drawdown):
                    drawdown = cur_drawdown
                s_max = s_t
                s_min = s_t
            elif (s_t < s_min):
                s_min = s_t

            d_w_t_2 = np.random.normal(0, d_t**0.5, 1)
            d_v_t = alpha * ( theta - v_t) * d_t + phi * (v_t**0.5) * d_w_t_2
            v_t += d_v_t
            rv /= n_step
            if (drawdown > 0.1):
                p_T += (rv - (con.bs_vol+0.05)**2) * con.notional
        
        s_T += s_t
    s_T /= n_sim
    p_T /= n_sim
    return s_T, p_T


print(mc_stock(1, 10, 0.01, 0.1, 0.02))'''

def mc_df(n_sim, n_step, alpha, theta, phi):
    s = np.zeros((n_sim, n_step))
    p = np.zeros((n_sim,1))
    mu = con.r
    for i in range (n_sim):
        s[i][0] = con.s_0
        v_t = con.atm_iv_1m
        rv = 0
        s_max = con.s_0
        d_t = 1 / n_step
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
            p[i] = max((rv - (con.bs_vol+0.05)**2),0) * con.notional   
    return s

mc_s = mc_df(10, 4, 0.01, 0.1, 0.02)
plt.plot(mc_s.T)
plt.show()
