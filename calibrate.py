import numpy as np
from scipy.optimize import minimize
import monte_carlo as mc
import const

#alpha is rate of reversion, theta is long term vol, phi is vol of vol
def loss_function( params ):
    alpha, theta, phi, rho = params
    mse = 0
    num = 0
    for t in const.T:
        for i in range(len(const.strikes)):
            for j in range(len(const.strikes[i])):
                k = const.strikes[i][j]
                p = const.prices[i][j]
                mc_s, mc_p = mc.mc_vanilla(10, 100, alpha, theta, phi, rho, const.s_0, 
                            const.atm_iv_1m, k, t, const.CALL )
                pred_p = np.mean(mc_p)
                mse += (pred_p - p) ** 2
                num += 1
    return mse / num


def calibrate_sv():
    initial_params = [1, 1, 1, -0.3]
    return( minimize( loss_function, initial_params, tol=1e-6 ) )

print(calibrate_sv())