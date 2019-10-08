import const 
import copy
import math
import numpy as np
import greeks

def static_hedging(num_paths, F, P, alphas, phi, dt, t, T1):
    #capital for hedging
    K = F[4]
    #portfolio contains short structure, delta hedged
    curr_delta = greeks.delta(num_paths, alphas, K, phi, dt, t, T1)
    forward_bought = -curr_delta
    initial_treasury_capital = -curr_delta * P[4]
    initial_capital = const.fair_price + initial_treasury_capital
    #simulation
    final_capitals = []
    knock_in = 0
    three_m_rate = []
    one_y_rate = []
    spread = []
    for path in range(num_paths):
        tau = t
        #index i is bond price for 0, 0.25*(i+1)
        discount = 1
        curr_bond_price = copy.deepcopy(P)
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
            final_capital = max((r - K),0)
        else:
            final_capital = 0
        final_capital = -final_capital / discount + initial_treasury_capital * curr_bond_price[4]
        final_capitals.append(final_capital)
    return np.mean(final_capitals), np.std(final_capitals)

print(static_hedging(100000, const.F, const.P, const.alphas, const.phi, 0.25, 0, 1))

def get_forward_rate(P):
    F = []
    for i in range(len(P)):
        if (i == 0):
            F.append(-math.log(P[i])/const.dt)
        else:
            F.append( (math.log(P[i-1]) - math.log(P[i]))/const.dt )
    return(F)

def dynamic_hedging(num_paths, F, P, alphas, phi, dt, t, T1):
        #capital for hedging
    K = F[4]
    #portfolio contains short structure, delta hedged
    #simulation
    final_capitals = []
    knock_in = 0
    three_m_rate = []
    one_y_rate = []
    spread = []
    for path in range(num_paths):
        tau = t
        ################## hedging variables ################
        capital = 0
        curr_delta = 0
        ################## end of hedging variables ################
        discount = 1
        curr_bond_price = copy.deepcopy(P)
        v1 = [0 for i in range(len(curr_bond_price))]
        v2 = [0 for i in range(len(curr_bond_price))]
        three_m_bond_yield = -math.log(curr_bond_price[0])/dt
        one_y_bond_yield = -math.log(curr_bond_price[3])
        initial_spread = one_y_bond_yield - three_m_bond_yield
        #loop move bond price forward by 1 time step
        while(tau < T1):
            delta = greeks.delta(10000, alphas, K, phi, dt, tau, T1, get_forward_rate(curr_bond_price))
            forward_bought = -(delta - curr_delta)
            capital = capital - forward_bought * P[4]
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
            final_capital = max((r - K),0)
        else:
            final_capital = 0
        final_capital = -final_capital  + curr_delta * curr_bond_price[4] + capital + const.fair_price
        final_capitals.append(final_capital)
    return np.mean(final_capitals), np.std(final_capitals)

#print(dynamic_hedging(1000, const.F, const.P, const.alphas, const.phi, 0.25, 0, 1))