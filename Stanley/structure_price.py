import const 
import copy
import math
import numpy as np

def knock_in_caplet(num_paths, alphas, K, phi, dt, t, T1):
    payoffs = []
    prices = []
    knock_in = 0
    three_m_rate = []
    one_y_rate = []
    spread = []
    for path in range(num_paths):
        tau = t
        #index i is bond price for 0, 0.25*(i+1)
        discount = 1
        curr_bond_price = copy.deepcopy(const.P)
        v1 = [0 for i in range(len(curr_bond_price))]
        v2 = [0 for i in range(len(curr_bond_price))]
        three_m_bond_yield = -math.log(curr_bond_price[0])/dt
        one_y_bond_yield = -math.log(curr_bond_price[3])
        initial_spread = one_y_bond_yield - three_m_bond_yield
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
            Z1 = np.random.standard_normal(1)[0]
            Z2 = np.random.standard_normal(1)[0]
            #for every bond, we want to move the bond price forward by 1 step
            for i in range(len(curr_bond_price)):
                curr_bond_price[i] = curr_bond_price[i] * (1 + r * dt + 
                                    v1[i] * math.sqrt(dt) * Z1 + 
                                    v2[i] * math.sqrt(dt)* Z2)
                if(curr_bond_price[i] <= 0):
                    curr_bond_price[i] = 0.0000001
        #print(curr_bond_price)
        risk_free_index = int(tau/dt)
        three_m_bond_price = curr_bond_price[risk_free_index]
        three_m_bond_yield = -math.log(curr_bond_price[risk_free_index])/dt
        one_y_bond_yield = -math.log(curr_bond_price[risk_free_index+3])
        final_spread = one_y_bond_yield - three_m_bond_yield
        three_m_rate.append(three_m_bond_yield)
        one_y_rate.append(one_y_bond_yield)
        spread.append(final_spread)
        r = -math.log(three_m_bond_price)/dt
        percentage_change = (final_spread - initial_spread)/initial_spread
        if (percentage_change < -0.5):
            knock_in += 1
            price = max((r - const.F[risk_free_index]),0) * discount 
        else:
            price = 0
        payoffs.append(price / discount)
        prices.append(price)
    #np.savetxt( "1000000path_structure_payoff.csv" ,payoffs )
    np.savetxt( "1000000path_3m_yield.csv" ,three_m_rate )
    np.savetxt( "1000000path_1y_yield.csv" ,one_y_rate )
    np.savetxt( "1000000path_spread.csv" ,spread )
    #print(knock_in/num_paths)
    return np.mean(price), np.std(price), np.mean(three_m_rate), np.mean(one_y_rate), np.mean(spread),\
            np.std(three_m_rate), np.std(one_y_rate), np.std(spread)

print(knock_in_caplet(1000000, const.alphas, 0.0609, const.phi, 0.25, 0, 1 ))