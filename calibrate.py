import numpy as np
from scipy.optimize import minimize, basinhopping, least_squares
import monte_carlo as mc
import const

set_alpha = 2
set_theta = 0.1
set_phi = 0.05
set_rho = -0.4

#alpha is rate of reversion, theta is long term vol, phi is vol of vol
def loss_function( params ):
    alpha, phi, theta, rho = params
    #theta = set_theta
    mse = 0
    num = 0
    for j in range(len(const.strikes[0])):
        for i in range(len(const.T)):
            t = const.T[i]
            k = const.strikes[i][j]
            p = const.prices[i][j]
            n_sim = 1000
            n_step = 1000
            mc_s, mc_p = mc.mc_vanilla(n_sim, n_step, alpha, theta, phi, rho, const.s_0, 
                        const.atm_iv_1m, k, t, const.CALL )
            pred_p = np.mean(mc_p) * (1 + const.r/n_step)**(-n_step)
            print(pred_p, p)
            mse += (pred_p - p) ** 2
            num += 1
    return mse / num


def loss_function_vec( params ):
    alpha, phi, rho = params
    theta = set_theta
    mse = 0
    num = 0
    residual = []
    for i in range(len(const.T)):
        t = const.T[i]
        for j in range(len(const.strikes[i])):
            k = const.strikes[i][j]
            p = const.prices[i][j]
            n_sim = 1000
            n_step = 1000
            mc_s, mc_p = mc.mc_vanilla(n_sim, n_step, alpha, theta, phi, rho, const.s_0, 
                        const.atm_iv_1m, k, t, const.CALL )
            pred_p = np.mean(mc_p) * (1 + const.r/n_step)**(-n_step)
            print(pred_p, p)
            residual.append(pred_p - p)
            mse += (pred_p - p) ** 2
            num += 1
    return residual

def loss_function_term_structure( params ):
    alpha, theta = params
    mse = 0
    num = 0
    global set_phi
    global set_rho
    residual = []
    for i in range(len(const.T)):
        t = const.T[i]
        k = const.strikes[i][1]
        p = const.prices[i][1]
        n_sim = 1000
        n_step = 100
        mc_s, mc_p = mc.mc_vanilla(n_sim, n_step, alpha, theta, set_phi, set_rho, const.s_0, 
                    const.atm_iv_1m, k, t, const.CALL )
        pred_p = np.mean(mc_p) * (1 + const.r/n_step)**(-n_step)
        print(pred_p, p)
        residual.append(pred_p - p)
        mse += (pred_p - p) ** 2
        num += 1
    return mse / num

def loss_function_term_structure_vec( params ):
    alpha = params
    theta = set_theta
    mse = 0
    num = 0
    global set_phi
    global set_rho
    residual = []
    for i in range(len(const.T)):
        t = const.T[i]
        k = const.strikes[i][1]
        p = const.prices[i][1]
        n_sim = 1000
        n_step = 1000
        mc_s, mc_p = mc.mc_vanilla(n_sim, n_step, alpha, theta, set_phi, set_rho, const.s_0, 
                    const.atm_iv_1m, k, t, const.CALL )
        pred_p = np.mean(mc_p) * (1 + const.r/n_step)**(-n_step)
        print(pred_p, p)
        residual.append(pred_p - p)
        mse += (pred_p - p) ** 2
        num += 1
    print("")
    return residual


def loss_function_smile( params ):
    phi = params[0]
    rho = params[1]
    mse = 0
    num = 0
    global set_alpha
    global set_theta
    residual = []
    for i in range(len(const.T)):
        t = const.T[i]
        for j in range(len(const.strikes[i])):
            k = const.strikes[i][j]
            p = const.prices[i][j]
            n_sim = 100
            n_step = 100
            mc_s, mc_p = mc.mc_vanilla(n_sim, n_step, set_alpha, set_theta, phi, rho, const.s_0, 
                        const.atm_iv_1m, k, t, const.CALL )
            pred_p = np.mean(mc_p) * (1 + const.r/n_step)**(-n_step)
            print(pred_p, p)
            residual.append(pred_p - p)
            mse += (pred_p - p) ** 2
            num += 1
    return mse / num 

def loss_function_smile_vec( params ):
    phi = params[0]
    rho = params[1]
    mse = 0
    num = 0
    global set_alpha
    global set_theta
    residual = []
    for i in range(len(const.T)):
        t = const.T[i]
        for j in range(len(const.strikes[i])):
            k = const.strikes[i][j]
            p = const.prices[i][j]
            n_sim = 100
            n_step = 100
            mc_s, mc_p = mc.mc_vanilla(n_sim, n_step, set_alpha, set_theta, phi, rho, const.s_0, 
                        const.atm_iv_1m, k, t, const.CALL )
            pred_p = np.mean(mc_p) * (1 + const.r/n_step)**(-n_step)
            #print(pred_p, p)
            residual.append(pred_p - p)
            mse += (pred_p - p) ** 2
            num += 1
    return residual

'''def calibrate_sv(initial_params):
    constraints = ( {'type': 'ineq', 'fun': lambda x: x[0]},
                    {'type': 'ineq', 'fun': lambda x: x[1]},
                    {'type': 'ineq', 'fun': lambda x: x[2]} )
    bounds = ((0, None), (0, 1), (0, 1))
    return(minimize(loss_function, initial_params, method="Anneal",
        tol=1e-6, bounds=bounds))'''


def calibrate_sv(initial_params):
    minimizer_kwargs = {"method": "BFGS"}
    bounds = ((0, None),  (0, 1), (-1,1))
    return( basinhopping(loss_function, initial_params, minimizer_kwargs=minimizer_kwargs) )


def main_minim():
    bound_term = ([0], [np.inf])
    bound_smile = ([0,-1], [1,1])
    global set_alpha, set_theta, set_phi, set_rho
    #calibrate alpha and theta
    ts_param = least_squares( loss_function_term_structure_vec, 
                        [set_alpha], 
                        method="dogbox", verbose=2, diff_step = 0.1,
                        bounds=bound_term)
    print(ts_param)
    set_alpha = ts_param.x
    smile_param = least_squares( loss_function_smile_vec, 
                        [set_phi, set_rho], 
                        method="dogbox", verbose=2, diff_step = 0.1,
                        bounds=bound_smile)
    print(smile_param)
    set_phi, set_rho = smile_param.x
    print( set_alpha, set_theta, set_phi, set_rho )

def lsq_term():
    bound_term = ([0, 0],[np.inf, 1])
    global set_alpha, set_theta
    ls_output = least_squares( loss_function_term_structure_vec, 
                        [set_alpha, set_theta], 
                        method="dogbox", verbose=2, diff_step = 0.1,
                        bounds=bound_term)
    set_alpha, set_theta = ls_output.x
    cost = loss_function_term_structure([set_alpha, set_theta])
    print(ls_output)
    print( set_alpha, set_theta, set_phi, set_rho )
    print( "Loss:%d" %(cost))

def main_lsq():
    bound_term = ([0, 0])
    bound_smile = ([0,-1], [1,1])
    full_bound = ([0,0,-1], [np.inf, 1, 1])
    global set_alpha, set_theta, set_phi, set_rho
    cost = loss_function([set_alpha, set_phi, set_rho])
    ls_output = least_squares( loss_function_vec, 
                        [set_alpha, set_phi, set_rho], 
                        verbose=2, diff_step = 0.1,
                        bounds=full_bound)
    set_alpha, set_phi, set_rho = ls_output.x
    cost = loss_function([set_alpha, set_phi, set_rho])
    print(ls_output)
    print( set_alpha, set_phi, set_rho )
    print( "Loss:%d" %(cost))
    #calibrate alpha and theta
    '''
    loss = loss_function([set_alpha, set_theta, set_phi, set_rho]) 
    while( loss_function([set_alpha, set_theta, set_phi, set_rho]) > 1 ):
        ts_param = least_squares( loss_function_term_structure_vec, [set_alpha, set_theta], 
                        method="dogbox", ftol=0.01, verbose=2, diff_step = 0.1,
                        bounds=bound_term)
        print(ts_param)
        set_alpha, set_theta = ts_param.x
        smile_param = least_squares( loss_function_smile_vec, [set_phi, set_rho], 
                        method="dogbox", ftol=0.01, verbose=2, diff_step = 0.1,
                        bounds=bound_smile)
        print(smile_param)
        set_phi, set_rho = smile_param.x
        print( set_alpha, set_theta, set_phi, set_rho )
        print( "Loss:%d" %(loss_function([set_alpha, set_theta, set_phi, set_rho])))
    '''
#main_minim()
#main_minim()
#main_lsq()
print(loss_function( [10.97858327,  0.0214962,  0.01362476, -0.55156066] ))

#lsq_term()

#print(calibrate_sv([set_alpha, set_phi, set_rho]))

