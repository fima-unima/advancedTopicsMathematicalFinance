import numpy as np  # Import packages numpy and matplotlib
import matplotlib.pyplot as plt 

T = 2 # Time horizon
N = 1000 # Discretization number per time unit
#Variables for the processes:
x0 = 1
mu = 1
sigma = 1
y0 = 1
z0 = 0.25


def brownian_mot(N,T):  # Function to simulate a Brownian motion
    mu, sigma = 0, np.sqrt(1/N) # Mean and standard deviation 
    s = np.random.normal(mu, sigma, N*T) # N*T normal distributed random numbers 
    # Create realization of Brownian Motion 
    B = np.zeros(N*T+1)  # Vector creation
    for i in range(N*T):
        B[i+1]=B[i]+s[i] # Increments of BM on each time step
    return B   # Return Brownian motion

B = brownian_mot(N,T)

def geom_BM(x0,mu,sigma,N,T,B):
    X = np.zeros(N*T+1)
    X[0] = x0 # X starts at x0
    for i in range(N*T):
        X[i+1] = x0 * np.exp((mu-(sigma**2)/2)*((i+1)/N) + sigma*B[i+1]) # Definition of X
    return X # Return

X = geom_BM(x0,mu,sigma,N,T,B)

def procY(y0,N,T,B):
    Y = np.zeros(N*T+1)
    Y[0] = y0 
    for i in range(N*T):
        Y[i+1] = np.exp(((i+1)/N)**2/2) * (y0 + B[i+1]) # Definition
    return Y

Y = procY(y0,N,T,B)

def procZ(z0,mu,sigma,N,T,B):
    Z = np.zeros(N*T+1)
    integ = np.zeros(N*T+1) # store the value of the integral
    Z[0] = z0
    integ[0] = 0
    for i in range(N*T):
        integ[i+1] = integ[i] + (B[i+1]-B[i])*(np.exp(mu/N))
        Z[i+1] = ( np.sqrt(z0)*np.exp(mu*(i+1)/N) + sigma * integ[i+1] )**2
    return Z

Z = procZ(z0,mu,sigma,N,T,B)

#Plot
x = np.linspace(0, T, N*T+1)
plt.plot(x,X, label="Geometric BM")
plt.plot(x,Y, label="Process b)")
plt.plot(x,Z, label="Process c)")
plt.title("3 stochastic processes")
plt.xlabel('t') # Labeling x-axis
plt.legend(loc="best")
plt.show()