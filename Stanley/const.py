import math
from scipy.stats import norm

dt = 0.25

F = [
    0.0520, #0,0, 0.25
    0.0575, #0, 0.25, 0.5
    0.0591, #0, 0.5, 0.75
    0.0601, #0, 0.75, 1
    0.0609, #0, 1, 1.25
    0.0643, #0, 1.25, 1.5
    0.0667, #0, 1.5, 1.75
    0.0642, #0, 1.75, 2
    ]

P = []
for i in range(len(F)):
    if(i == 0):
        P.append(math.exp(-F[i] * dt))
    else:
        P.append(P[-1] * math.exp(-F[i]*dt))

P = [  
    0.9870841350202876, #P(0,0.25)
    0.9729962994896734, #P(0,0.5)
    0.9587259608921443, #P(0,0.75)
    0.9444287798676372, #P(0,1)
    0.9301587578854282, #P(0,1.25)
    0.9153259935998006, #P(0,1.5)
    0.9001894840360222, #P(0,1.75)
    0.8858567705204549  #P(0,2)
    ]

ISD = [
    0.2416,
    0.2228,
    0.2016,
    0.1897,
    0.1757,
    0.1609,
    0.1686
    ]

def black_atm_caplet_price(K, sigma, T, dt, r):
    d1 = (((sigma**2)/2) * T)/(sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    return( (K * N_d1 - K * N_d2)*math.exp(-r*T) * dt )

mkt_caplet_price = []
for i in range(len(ISD)):
    mkt_caplet_price.append(black_atm_caplet_price(F[i+1], ISD[i], 0.25*(i+1), dt, F[0]))

alphas = [0.0016337120054001582, 0.0003926087242999385, 1.8374411741415394e-05, 1.4852394803269864e-05]
phi = 1.4

fair_price = 0.000966
