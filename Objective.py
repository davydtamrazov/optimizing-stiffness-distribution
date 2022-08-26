import numpy as np
from scipy.linalg import eigh
import csv

def f(a_g,t,N,H,m,k,xi,n_modes = 5):
    
    # Assemble mass and stiffness matrices
    M = m*np.eye(N)
    K = assemble_stiffness(k)
    
    # Initial position and velocity
    u0 = 0
    du0 = 0
    
    # Modal analysis
    w, T, phi, G = modal_analysis(N, M, K, H)
    
    # Compute modal displacement time histories
    D = np.empty((0,len(t)))
    for i in range(n_modes):
        Ti = T[i]
        u,du,ddut,Fs,PRD,t_PRD,PRV,t_PRV,PAA,t_PAA,PRV_pseudo,PAA_pseudo = \
         timeint(Ti,xi,t,a_g,u0,du0)
    
        D = np.concatenate((D,u.T))
        
    IDR = interstory_drift_ratio(H, phi, G, D, N, t, n_modes)
    
    # Return overall maximum IDR over all stories
    return np.amax(abs(IDR))

def c(k,kmax,kmin):
    
    kmax *= np.ones(len(k))
    kmin *= np.ones(len(k))
    
    c1 = k - kmax
    c2 = kmin - k
    
    return np.hstack((c1,c2))
 
def assemble_stiffness(k):
    n = len(k)
    K = np.zeros((n,n))
    for i in range(n-1):
        K[i,i] = k[i] + k[i+1]
        K[i,i+1] = -k[i+1]
        K[i+1,i] = -k[i+1]
    K[n-1,n-1] = k[n-1]
    return K

def modal_analysis(N, M, K, H):
    
    # Solve eigenvalue problem
    lmd, phi = eigh(K,M)

    # Modal frequencies and periods
    w = np.sqrt(lmd)
    T = 2*np.pi/w
    
    # Modal participation factors
    G = np.ones(N)
    for i in range(N):
        phi_n = phi[:,i]
        G[i] = (phi_n.T @ M @ np.ones(N)) / (phi_n.T @ M @ phi_n)
    
    # Effective modal shapes
    # ephi = phi @ np.diag(G)
    
    return w, T, phi, G

def interstory_drift_ratio(H, phi, G, D, N, t, n_modes = 5):
    
    # Effective mode shape for the considered modes
    ephi = (phi @ np.diag(G))[:,range(n_modes)]
    
    # Relative interstory drift ratios from effective mode shapes
    IDRr = np.array([(ephi[i,:]-ephi[i-1,:])/H \
                     if i != 0 else ephi[i,:]/H for i in range(N)])
    
    # Calculate interstory drift ratio for each floor at each time step using
    # SRSS combination rule

    IDR = np.zeros((N,np.shape(D)[1])) 
    for i in range(N):
        for md in range(n_modes):
            IDR[i,:] += (IDRr[i,md] * D[md,:])
            
    return IDR

def loaddata(folder,filename):
    
    '''
    Loads ground motion data and timestep vector from specified file name
    
    INPUT:
        folder: Path containing name of folder with data
        filename: String containing file name of .csv containing data
    
    OUTPUT:
        t: vector of timesteps from data file
        ddug: vector of ground acceleration from data file
    
    '''
    
    with open(folder/filename,newline='\n') as csvfile:
        # read raw file lines into data list
        data = [x for x in list(csv.reader(csvfile,delimiter=',')) if x]
        
    # Convert string data entries into numbers and store them in X, Y
    t = np.array([float(entry[0]) for entry in data[1:]])
    ddug = np.array([float(entry[1]) for entry in data[1:]])
        
    return t, ddug

def interp_accel(t,ddug,delt_new):
    
    # Create new timestep vector for suitable delta t
    t_new = np.arange(t[0],t[-1],delt_new);

    # Linear interpolation of given acceleration values using built-in fcn
    ddug_new = np.interp(t_new,t,ddug)

    return t_new, ddug_new

    

def timeint(T,xi,t,ddug,u0,du0):

    '''
    
    Time Integration: 
    Performs time integration for a 1DOF linear elastic dynamic system 
    using Iwan's Method
    
    INPUT:
        T: the natural period of the system
        xi: the damping ratio
        t: vector of timesteps for ground motion record
        ddug: vector of ground motion accelerations (u_g'')
        u0: initial displacement of system
        du0: initial velocity of system
    
    OUTPUT: 
        t: vector of timesteps over which  response quantities are computed
        u: the relative displacement time history
        du: the relative velocity time history
        ddut: the absolute acceleration time history
        PSD: the peak spectral displacement
        t_PSD: the time of occurrence of the PSD
        PSV: the peak spectral velocity
        t_PSV: the time of occurrence of the PSV
        PSA: the peak spectral absolute acceleration
        t_PSA: the time of occurrence of the PSA
        PSV_pseudo: the peak pseudo-velocity 
        PSA_pseudo: the peak pseudo-acceleration
    '''
    
    # Compute fictitious system properties based on input natural period
    wn = 2*np.pi/T
    m = 1           # mass (assumed=1)
    k = wn**2 * m   # stiffness
    c = 2*xi*m*wn   # damping
    
    # Find suitable timestep based on period and ground motion
    delt = t[1] - t[0] # the timestep size of the ground motion record  
    p = -ddug
    n = len(ddug)
    
    # Compute Iwan's method coefficients
    d = np.sqrt(1-xi**2)
    base = np.exp(-xi*wn*delt)
    wd = wn*d
    ss = np.sin(wd*delt)
    cs = np.cos(wd*delt)
    
    A = base *((xi/d)*ss + cs)
    B = base *((1/wd)*ss)
    C = (1/k) * (((2*xi)/(wn*delt)) + base*((((1-2*xi**2)/(wd*delt)) - \
         xi/d)*ss - (1+(2*xi)/(wn*delt))*cs))
    D = (1/k) * (1 - ((2*xi)/(wn*delt)) + \
         base*( ((2*xi**2-1)/(wd*delt))*ss + ((2*xi)/(wn*delt))*cs ))
    Ap = -base*( (wn/d)*ss)
    Bp = base*(cs - (xi/d)*ss)
    Cp = (1/k) * ((-1/delt)+ base*(((wn/d) + ((xi)/(delt*d)))*ss + \
           (1/delt)*cs))
    Dp = (1/(k*delt)) * (1 - base*((xi/d)*ss + cs))
    
    # Initialize output variables
    u = np.zeros([n,1]);
    du = np.zeros([n,1]);
    ddu = np.zeros([n,1]);
    ddut = np.zeros([n,1]);
    Fs =  np.zeros([n,1]);

    # Initial Conditions
    u[1] = u0;           # Initial displacement
    du[1] = du0;         # Initial velocity
    ddu[1] = 0;          # Initial acceleration (assumed)
    Fs[1] = k*u0;        # Initial restoring force
    
    # Time integration per Iwan's method
    for i in range(n-1):
        u[i+1] = A*u[i] + B*du[i] + C*p[i] + D*p[i+1]
        du[i+1] = Ap*u[i] + Bp*du[i] + Cp*p[i] + Dp*p[i+1]
        ddu[i+1] = (p[i+1] - c*du[i+1]- k*u[i+1])/m
        ddut[i+1] = ddug[i+1] + ddu[i+1]
        Fs[i+1] = k*u[i+1]
    
    # Compute spectral ordinates
    PRD = max(abs(u))           # Peak relative displacement
    PRV = max(abs(du))          # Peak relative velocity
    PAA = max(abs(ddut))        # Peak absolute acceleration
    PRV_pseudo = wn * PRD       # Peak pseudo velocity
    PAA_pseudo = wn**2 * PRD    # Peak pseudo acceleration
    
    # Calculate times of occurrence of peak values
    ind_PRD = np.where(abs(u)==PRD)[0]      # index of peak
    t_PRD = t[ind_PRD]
    ind_PRV = np.where(abs(du)==PRV)[0]     # index of peak
    t_PRV = t[ind_PRV]
    ind_PAA = np.where(abs(ddut)==PAA)[0]   # index of peak
    t_PAA = t[ind_PAA]

    return u,du,ddut,Fs,PRD,t_PRD,PRV,t_PRV,PAA,t_PAA,PRV_pseudo,PAA_pseudo
