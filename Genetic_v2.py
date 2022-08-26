import numpy as np
from pathlib import Path
from Objective import f, c, loaddata, interp_accel
import matplotlib.pyplot as plt


#%%
# Ground motion data location        
folder = Path("Groundmotions/")
filename = 'El_Centro_NS.csv'

# Analysis properties
g = 386.2  # in/s^2
delt = 0.02

# Read and interpolate ground motion record
t, a_g = loaddata(folder,filename)
t, a_g = interp_accel(t,a_g,delt)

# Scale record to correct units
a_g *= g

# System properties
N = 10
n_modes = 3
H = 1
kmin = 0.25
kmax = 1.75
m = 1/g
xi = 0.02

#%% Optimization parameters
alpha = 0.4
k0 = np.array([1]*N)
f0 = f(a_g,t,N,H,m,k0,xi,n_modes)
obj = lambda k: np.array([(1-alpha)*f(a_g,t,N,H,m,k,xi,n_modes)/f0 + alpha*np.mean(k)/np.mean(k0)])


from deap import base, creator, tools


#%%

# Create fitness minimization criterion
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# Create individual
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

#%%

# Create fitness minimization criterion
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# Create individual
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

#%%
IND_SIZE=N

toolbox = base.Toolbox()

gen_k = lambda: np.random.uniform(kmin,kmax)
evaluate = lambda individual: np.array([obj(individual)])

toolbox.register("rand_k", gen_k)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                  toolbox.rand_k, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mate",tools.cxUniform,indpb = 0.5)
#toolbox.register("mate",tools.cxSimulatedBinaryBounded,eta = 0.5,low=kmin,up=kmax)
# toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("mutate",tools.mutPolynomialBounded, eta=0.3, low=kmin, up=kmax, indpb=0.3)
toolbox.register("select", tools.selTournament, tournsize=10)
toolbox.register("evaluate", evaluate)

pop = toolbox.population(n=500) # number of individuals in population
CXPB = 0.5  # crossover probability
# MUTPB = 0.2 # mutation probability
MUTPB = 0.5
NGEN = 50   # number of generations

xhist = []
fhist = []

# Evaluate the entire population
fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit
    # print(fit)

for g in range(NGEN):
    print('\n',g)
    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if np.random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if np.random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # The population is entirely replaced by the offspring
    pop[:] = offspring
    xbest = tools.selBest(pop,1)
    fbest = xbest[0].fitness.values[0]
    xhist.append(xbest)
    fhist.append(fbest)
    print(xbest, fbest)
    # print(offspring)
    # print(fitnesses)

#%%

x0 = tools.selBest(pop,1)
best_obj = x0[0].fitness.values[0]
print(best_obj)


#%%
#x0 = np.array([0.9013,0.761,0.59,0.626,1.237])
#x0 = np.array([0.6672078 , 0.5667836 , 0.55552381, 0.56547384, 0.45774851])
#x0 = np.array([0.8243517 , 0.65613437, 0.64746203, 0.62576232, 0.43076368])
# x0_test = np.array([1.6737663181076827, 1.4620228566776738, 1.3533691201857212, 1.2216417036532758, 1.16507884549066, 1.1002352674932094, 0.9618881958463086, 0.8673504618816205, 0.7171532045479849, 0.5525350134970812])

# import scipy.optimize as opt

# xhist_nm = []
# def callback_xk(xk):
#     print(xk)
#     xhist_nm.append(xk)

# #xopt = opt.fmin(obj, x0, xtol=1e-6, ftol=1e-6, maxiter=1500, disp=True, callback=callback_xk)
# lw = np.array([kmin] * N)
# up = np.array([kmax] * N)
# bounds=opt.Bounds(lw, up, keep_feasible=False)
# #xopt = opt.minimize(obj, x0, method='L-BFGS-B', bounds=bounds, callback=callback_xk, 
#                     #options={'disp':True, 'maxiter':1000, 'ftol':1e-6, 'gtol':1e-5})
# xopt = opt.minimize(obj, x0_test, method='Nelder-Mead', callback=callback_xk, options={'disp':True, 'maxiter':1000, 'ftol':1e-6, 'gtol':1e-5})
# xhist_nm = [list(x) for x in xhist_nm]

#%%

fhist = np.array([float(f) for f in fhist])

#%%
xhist = np.array([np.array(x[0]) for x in xhist])


