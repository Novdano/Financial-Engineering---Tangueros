import const
import math
import copy
import numpy as np

def caplet_price(num_paths, alphas, phi, dt, t, T1):
    payoffs = []
    total_corr = 0
    for path in range(num_paths):
        tau = t
        #index i is bond price for 0, 0.25*(i+1)
        discount = 1
        curr_bond_price = copy.deepcopy(const.P)
        v1 = [0 for i in range(len(curr_bond_price))]
        v2 = [0 for i in range(len(curr_bond_price))]
        change_short = []
        change_long = []
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
            prev_short_yield = -math.log(curr_bond_price[risk_free_index])/dt
            prev_long_yield = -math.log(curr_bond_price[risk_free_index + 3])
            Z1 = np.random.standard_normal(1)[0]
            Z2 = np.random.standard_normal(1)[0]
            #for every bond, we want to move the bond price forward by 1 step
            for i in range(len(curr_bond_price)):
                curr_bond_price[i] = curr_bond_price[i] * (1 + r * dt + 
                                    v1[i] * math.sqrt(dt) * Z1 + 
                                    v2[i] * math.sqrt(dt)* Z2)
                if(curr_bond_price[i] <= 0):
                    curr_bond_price[i] = 0.0000001
            short_yield = -math.log(curr_bond_price[risk_free_index + 1])/dt
            long_yield = -math.log(curr_bond_price[risk_free_index + 4])
            change_short.append(short_yield - prev_short_yield)
            change_long.append(long_yield - prev_long_yield)
        #print(curr_bond_price)
        risk_free_index = int(tau/dt)
        three_m_bond_price = curr_bond_price[risk_free_index]
        r = -math.log(three_m_bond_price)/dt
        #print(r)
        payoff = max((r - const.F[risk_free_index]),0) * discount 
        payoffs.append(payoff)
        correlation = np.corrcoef(change_short, change_long)
        correlation = correlation[0][1]
        total_corr += correlation
    return np.mean(payoffs), total_corr/num_paths



def calibrate_alpha(alphas, calibrate_phi, lr0, tol, decay):
    #each one calibrate the alpha
    for i in range(4):
        print(i)
        T1 = (i + 1)*0.25
        mse, correlation = caplet_price(1000, alphas, phi, 0.25, 0, T1)
        mse  = abs(mse - const.mkt_caplet_price[i])
        lr = lr0
        while(mse > tol):
            old_alpha = alphas[i]
            print(alphas[i], mse)
            curr_alpha_up = alphas[i] * (1 + lr)
            curr_alpha_down = alphas[i] * (1 - lr)
            model_price_old, correlation = caplet_price(1000, alphas, phi, 0.25, 0, T1)
            alphas[i] = curr_alpha_up
            model_price_up, correlation = caplet_price(1000, alphas, phi, 0.25, 0, T1)
            alphas[i] = curr_alpha_down
            model_price_down, correlation = caplet_price(1000, alphas, phi, 0.25, 0, T1)
            mkt_price = const.mkt_caplet_price[i]
            mse_up = abs(model_price_up - mkt_price)
            mse_down = abs(model_price_down - mkt_price)
            mse_old = abs(model_price_old- mkt_price)
            # if(mse_down > mse_old and mse_up > mse_old):
            #     break
            if(mse_up > mse_down):
                alphas[i] = curr_alpha_down
                mse = mse_down
                lr /= decay
            elif(mse_up < mse_down):
                alphas[i] = curr_alpha_up
                mse = mse_up
                lr /= decay
    return alphas


def calibrate_phi(alphas, phi, lr0, tol, decay):
    actual_corr = 0.81
    curr_phi = phi
    curr_model_price, curr_corr = caplet_price(1000, alphas, phi, 0.25, 0, 1) 
    mse_old = abs(curr_corr - actual_corr)
    lr = lr0
    print(mse_old)
    while(mse_old > tol):
        phi_up = phi * (1+lr)
        phi_down = phi * (1 - lr)
        model_price_up, corr_up = caplet_price(1000, alphas, phi_up, 0.25, 0, 1) 
        model_price_down, corr_down = caplet_price(1000, alphas, phi_down, 0.25, 0, 1) 
        mse_up = abs(corr_up - actual_corr)
        mse_down = abs(corr_down - actual_corr)
        if(mse_up > mse_down):
            phi = phi_down
            mse_old = mse_down
            curr_corr = corr_down
            lr /= decay
        elif(mse_up < mse_down):
            phi = phi_up
            mse_old = mse_up
            curr_corr = corr_up
            lr /= decay
        print(phi, mse_old, curr_corr)
    return(phi)

def check_price(alphas, phi):
    c1, _ = caplet_price(1000, alphas, phi, 0.25, 0, 0.25)
    c2, _ = caplet_price(1000, alphas, phi, 0.25, 0, 0.5)
    c3, _ = caplet_price(1000, alphas, phi, 0.25, 0, 0.75)
    c4, _ = caplet_price(1000, alphas, phi, 0.25, 0, 1)
    print(c1, c2, c3, c4)
    print(const.mkt_caplet_price[0], const.mkt_caplet_price[1], const.mkt_caplet_price[2], const.mkt_caplet_price[3])


def calibrate(alphas, phi, lr0 = 0.1, tol = 0.000001, decay = 1.1):
    alphas = calibrate_alpha(alphas, phi, lr0, 1e-5, decay)
    #phi = calibrate_phi(alphas, 0.1, lr0, tol, 1.0000001)
    return(alphas, phi)




#alphas = [0.1 for i in range(4)]
#alphas = [0.0007933531054793789, 0.0002741856578150153, 6.562750490064817e-05, 8.149990323380371e-05] #phi = 3.3
#alphas = [0.000998623968725623, 0.0003094669015269544, 4.905818435642512e-05, 5.2634248541918144e-05] #phi = 2.64298
#alphas = [0.0012834699645740623, 0.00026616290811232765, 2.452909217821256e-05, 3.57987667293167e-05] #phi = 2.1
#alphas = [0.0013001425982507085, 0.00026882453719345094, 2.379844944555294e-05, 3.5426848362429415e-05] #phi = 2.25
#alphas = [0.0014160124495893158, 0.0003252776900040757, 1.6726724772166804e-05, 2.409686451733034e-05] #phi = 1.81
#alphas = [0.0013873581504275968, 0.00028980197022567325, 1.6726724772166804e-05, 2.3858543879246853e-05] #phi = 2.24
#alphas = [0.001236048883025226, 0.00038562137070394557, 1.963581759360895e-05, 2.3858543879246853e-05] #phi = 1.95
#alphas = [0.001236048883025226, 0.00038562137070394557, 1.5595565521263067e-05, 1.780439606846251e-05]
alphas =  [0.0013596537713277488, 0.00038562137070394557, 1.871467862551568e-05, 1.4567233146923872e-05]
phi = 1.94
#alphas, phi = calibrate(alphas, phi)
#print(caplet_price(1000, alphas, phi, 0.25, 0, 1))
#print("alphas", alphas)
#check_price(alphas, phi)








