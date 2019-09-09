from scipy.optimize import minimize
import monte_carlo
import const

#alpha is rate of reversion, theta is long term vol, phi is vol of vol
def loss_function( params ):
    alpha_0, theta_0, phi_0 = params
    mse = 0
    num = 0
    for t in T:
        for i in range(len(const.strikes)):
            for j in range(len(const.strikes[i])):
                k = strikes[i][j]
                p = prices[i][j]
                mse += (vanilla_monte_carlo( s_0, sigma_0, k, t ) - p) ** 2
                num += 1
    return mse / num


def calibrate_sv():
    initial_params = [0, 0, 0]
    return( minimize( loss_function, initial_params, tol=1e-6 ) )
