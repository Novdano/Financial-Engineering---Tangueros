import numpy as np
from scipy.optimize import minimize, basinhopping
import monte_carlo as mc
import const

set_alpha = 1
set_theta = 1
set_phi = 0.08
set_rho = -0.3

#alpha is rate of reversion, theta is long term vol, phi is vol of vol
def loss_function( params ):
    alpha, theta, phi = params
    mse = 0
    num = 0
    for i in range(len(const.T)):
        t = const.T[i]
        for j in range(len(const.strikes[i])):
            k = const.strikes[i][j]
            p = const.prices[i][j]
            n_sim = 1000
            n_step = 100
            mc_s, mc_p = mc.mc_vanilla(n_sim, n_step, alpha, theta, phi, -0.3, const.s_0, 
                        const.atm_iv_1m, k, t, const.CALL )
            pred_p = np.mean(mc_p) * (1 + const.r/n_step)**(-n_step)
            mse += (pred_p - p) ** 2
            num += 1
    return mse / num

def loss_function_term_structure( params ):
    alpha, theta = params
    mse = 0
    num = 0
    global set_phi
    global set_rho
    for i in range(len(const.T)):
        t = const.T[i]
        atm_k = const.strikes[i][2]
        atm_p = const.prices[i][2]
        n_sim = 1000
        n_step = 1000
        mc_s, mc_p = mc.mc_vanilla(n_sim, n_step, alpha, theta, set_phi, set_rho, const.s_0, 
                    const.atm_iv_1m, atm_k, t, const.CALL )
        pred_p = np.mean(mc_p) * (1 + const.r/n_step)**(-n_step)
        mse += (pred_p - atm_p) ** 2
        num += 1
    return mse / num



def loss_function_smile( params ):
    phi = params[0]
    rho = params[1]
    mse = 0
    num = 0
    global set_alpha
    global set_theta
    for i in range(len(const.T)):
        t = const.T[i]
        atm_k = const.strikes[i][2]
        atm_p = const.prices[i][2]
        n_sim = 1000
        n_step = 1000
        mc_s, mc_p = mc.mc_vanilla(n_sim, n_step, set_alpha, set_theta, phi, rho, const.s_0, 
                    const.atm_iv_1m, atm_k, t, const.CALL )
        pred_p = np.mean(mc_p) * (1 + const.r/n_step)**(-n_step)
        mse += (pred_p - atm_p) ** 2
        num += 1
    return mse / num


'''def calibrate_sv(initial_params):
    constraints = ( {'type': 'ineq', 'fun': lambda x: x[0]},
                    {'type': 'ineq', 'fun': lambda x: x[1]},
                    {'type': 'ineq', 'fun': lambda x: x[2]} )
    bounds = ((0, None), (0, 1), (0, 1))
    return(minimize(loss_function, initial_params, method="Anneal",
        tol=1e-6, bounds=bounds))'''


def calibrate_sv(initial_params):
    constraints = ( {'type': 'ineq', 'fun': lambda x: x[0]},
                    {'type': 'ineq', 'fun': lambda x: x[1]},
                    {'type': 'ineq', 'fun': lambda x: x[2]} )
    minimizer_kwargs = {"method": "BFGS"}
    bounds = ((0, None), (0, 1), (0, 1))
    return( basinhopping(loss_function, initial_params, minimizer_kwargs=minimizer_kwargs, niter=200) )


def main():
    bound_term = ((0, None), (0, 1))
    bound_smile = ((0,1), (-1,1))
    global set_alpha, set_theta, set_phi, set_rho
    #calibrate alpha and theta
    ts_param = minimize( loss_function_term_structure, [set_alpha, set_theta], tol=1e-8, bounds=bound_term)
    print(ts_param)
    set_alpha, set_theta = ts_param.x
    smile_param = minimize( loss_function_smile, [set_phi, set_rho], tol=1e-8, bounds=bound_smile)
    print(smile_param)
    set_phi, set_rho = smile_param.x
    print( set_alpha, set_phi, set_theta, set_rho )

main()

#print(calibrate_sv([1,1,1]))

