import const
import math
import copy
import numpy as np

def caplet_price(num_paths, alphas, phi, dt, t, T1):
    payoffs = []
    for path in range(num_paths):
        tau = t
        #index i is bond price for 0, 0.25*(i+1)
        discount = 1
        curr_bond_price = copy.deepcopy(const.P)
        v1 = [0 for i in range(len(curr_bond_price))]
        v2 = [0 for i in range(len(curr_bond_price))]
        #loop move bond price forward by 1 time step
        while(tau < T1):
            risk_free_index = int(tau/dt)
            for i in range(len(curr_bond_price)):
                v1[i] += - (alphas[risk_free_index] * phi * dt)
                v2[i] += - (alphas[risk_free_index] * (math.exp(-2 * ( dt*(i+1) - tau - dt )) - 0.5))
            P_risk_free_bond = curr_bond_price[risk_free_index]
            r = -math.log(P_risk_free_bond)/dt
            #print(r)
            discount = discount * math.exp(-r * dt)
            tau += dt
            #for every bond, we want to move the bond price forward by 1 step
            for i in range(len(curr_bond_price)):
                Z1 = np.random.normal(1)
                Z2 = np.random.normal(1)
                curr_bond_price[i] = curr_bond_price[i] * (1 + r * dt + 
                                    v1[i] * math.sqrt(dt) * Z1 + 
                                    v2[i] * math.sqrt(dt)* Z2)
        #print(curr_bond_price)
        risk_free_index = int(tau/dt)
        three_m_bond_price = curr_bond_price[risk_free_index]
        r = -math.log(three_m_bond_price)/dt
        #print(r)
        payoff = max((r - const.F[risk_free_index]),0) * discount 
        payoffs.append(payoff)
    return np.mean(payoffs)



def calibrate_alpha(alphas, calibrate_phi, lr0, tol, decay):
    #each one calibrate the alpha
    for i in range(4):
        T1 = (i + 1)*0.25
        mse = (caplet_price(1000, alphas, phi, 0.25, 0, T1) - const.mkt_caplet_price[i])**2
        lr = lr0
        while(mse > tol):
            old_alpha = alphas[i]
            curr_alpha_up = alphas[i] * (1 + lr)
            curr_alpha_down = alphas[i] * (1 - lr)
            model_price_old = caplet_price(1000, alphas, phi, 0.25, 0, T1)
            alphas[i] = curr_alpha_up
            model_price_up = caplet_price(1000, alphas, phi, 0.25, 0, T1)
            alphas[i] = curr_alpha_down
            model_price_down = caplet_price(1000, alphas, phi, 0.25, 0, T1)
            mkt_price = const.mkt_caplet_price[i]
            mse_up = (model_price_up - mkt_price)**2
            mse_down = (model_price_down - mkt_price)**2
            mse_old = (model_price_old- mkt_price)**2
            if(mse_down > mse_old and mse_up > mse_old):
                break
            if(mse_up > mse_down):
                alphas[i] = curr_alpha_down
                mse = mse_down
            elif(mse_up < mse_down):
                alphas[i] = curr_alpha_up
                mse = mse_up
            lr /= decay
    return alphas


def calibrate_phi(alphas, phi, lr0, tol, decay):
    



def calibrate(alphas, phi, lr0 = 0.1, tol = 1e-4, decay = 1.1):
    calibrate_alpha(alphas, phi, lr0, tol, decay)
    calibrate_phi(alphas, phi, lr0, tol, decay)
    return(alphas, phi)




for i in range(10):
    alphas = [0.1 for i in range(4)]
    #alphas[0] = 0.0005
    phi = 1
    alphas, phi = calibrate(alphas, phi)
    print(alphas)








