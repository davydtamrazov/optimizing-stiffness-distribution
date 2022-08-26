#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 10:49:35 2020

@author: tamrazovd
"""

import numpy as np
import scipy.optimize as opt
from Objective import f, c, loaddata, interp_accel
from pathlib import Path

# Ground motion data location        
folder = Path("Groundmotions/")
filename = 'El_Centro_NS.csv'

# Analysis properties
g = 386.2  # in/s^2
delt = 0.02
n_modes = 3

# Read and interpolate ground motion record
t, a_g = loaddata(folder,filename)
t, a_g = interp_accel(t,a_g,delt)

# Scale record to correct units
a_g *= g

# System properties
N = 10
H = 1
k = 1 * np.ones(N)

kmin = 0.25
kmax = 1.75
m = 1/g
xi = 0.02

# alpha = 0.5
k0 = np.array([1]*N)
f0 = f(a_g,t,N,H,m,k0,xi,n_modes)
# obj = lambda k: np.array([(1-alpha)*f(a_g,t,N,H,m,k,xi,n_modes)/f0 + alpha*np.mean(k)/np.mean(k0)])

#%%
#x0 = np.array([0.75323259, 0.69917908, 0.65713586, 0.63583989, 0.43395031])    # MADS best point
#x0 = np.array([0.75937287, 0.70026099, 0.65511151, 0.63839159, 0.43543229]) # DE best point
#x0 = np.array([0.8243517,0.65613437,0.64746203,0.62576232,0.43076368]) # GA
#x0 = np.array([0.74366919, 0.69168375, 0.65295467, 0.64211327, 0.42865936]) # GA best point
#x0 = np.array( [0.73538383, 0.67632336, 0.63749348, 0.60857258, 0.44954663]) # GA

# alpha=0.4
#x0 = np.array([1.30830947, 1.19522758, 1.09027778, 1.03104585, 0.92419768,
               # 0.83569622, 0.71584255, 0.78232648, 0.70691206, 0.49976854]) #alpha=0.4

# alpha=0.5
# x0 = np.array([1.3380800207884933, 1.2694018467225305, 1.1481744223463515, 
#                 1.0002171481502065, 0.926447516986812, 0.8810409074850861, 0.8203836685044308, 
#                 0.6549612577257964, 0.6401504726775442, 0.45487532343326226]) # alpha=0.5

# alpha=0.3
# x0 =np.array([1.27696,1.19969,1.15932,1.06047,0.934736,1.01919,1.0225,1.0322,0.766446,0.639843]) #alpha=0.3

# alpha=0.2
# x0 = np.array([1.6750586494292872, 1.4634757272845027, 1.346673660920378, 1.222546710985911, 
#                1.1658001301626704, 1.101188816593765, 0.9637170386875966, 0.8702185691255293, 
#                0.7207826447690651, 0.5551776477376074]) # alpha=0.2

alpha=0.1
x0 = np.array([1.7499996488163618, 1.69283989211542, 1.5227972029790495, 1.6278099694878063, 
                1.4255824392817267, 1.450403024382135, 1.4199420083886805, 1.3159787134936436, 
                1.0639607683990246, 0.7511587698983654])

# alpha = 0
# x0 = np.array([1.7486038273001743, 1.6613747575803193, 1.5462763435511309, 1.542737980305038, 
#                1.494801581948357, 1.4680359044166613, 1.4867744964769325, 1.3173766402809846, 
#                0.9966324786377034, 0.8044894010542365])

gamma = 10
obj_cst = lambda k: np.array([(1-alpha)*f(a_g,t,N,H,m,k,xi,n_modes)/f0 + alpha*np.mean(k)/np.mean(k0)]) \
        + gamma*any(c(k,kmax,kmin)>0)

print(obj_cst(x0))

def callback_xk(xk):
    print(xk)

xopt = opt.minimize(obj_cst, x0, method='Nelder-Mead', callback=callback_xk, options={'disp':True, 'maxiter':10000, 'ftol':1e-6, 'gtol':1e-5})

