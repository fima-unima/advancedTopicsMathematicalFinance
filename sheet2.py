import numpy as np  # Import packages numpy and matplotlib
import matplotlib.pyplot as plt 
from scipy.stats import norm  # Import norm to get access to the normal distribution function

T = 5 # Time horizon
N = 1000 # Discretization number per time unit
S0 = 10
K = 9
sigma = 2


def brownian_mot(N,T):  # Function to simulate a Brownian motion
    mu, sigma = 0, np.sqrt(1/N) # Mean and standard deviation 
    s = np.random.normal(mu, sigma, N*T) # N*T normal distributed random numbers 
    # Create realization of Brownian Motion 
    B = np.zeros(N*T+1)  # Vector creation
    for i in range(N*T):
        B[i+1]=B[i]+s[i] # Increments of BM on each time step
    return B   # Return Brownian motion

for N in (10,100,1000):
    B = brownian_mot(N,T)
    S = S0 + sigma*B  # stock price

    V = np.zeros(N*T+1) # Variable for the value of the option 

    for i in range(N*T+1):
        V[i] = (S[i]-K)*norm.cdf( (S[i]-K)/(sigma*np.sqrt(T-i/N)) ) + sigma*np.sqrt(T-i/N)*norm.pdf( (S[i]-K)/(sigma*np.sqrt(T-i/N)) ) # Value of the option

    # Delta Hedging:
    phi = np.zeros(N*T+1)
    bank = np.zeros(N*T+1)
    value = np.zeros(N*T+1)
    phi[0] = norm.cdf( (S0-K)/(sigma*np.sqrt(T)) )  # innitial amount of investment in the risky asset
    bank[0] = V[0] - phi[0]*S0  # initial bank account
    value[0] = bank[0] + phi[0]*S0 # track portfolio value of our hedging strategy
    for i in range(N*T):
        value[i+1] = bank[i] + phi[i]*S[i+1] # new value at time point i+1
        phi[i+1] = norm.cdf( (S[i+1]-K)/(sigma*np.sqrt(T-i/N)) ) # updated amount in risky asset
        bank[i+1] = bank[i] - S[i+1]*(phi[i+1]-phi[i]) # updated bank account such that trading strategy is self-financing


    #Plot
    x = np.linspace(0, T, N*T+1)
    #plt.plot(x,S)
    plt.plot(x,S,label="Stock price")
    plt.plot(x,V,label="Value function of the Call option")
    plt.plot(x,value,label="Value of the replicating portfolio")
    plt.title("N=" + str(N))
    plt.xlabel('t') # Labeling x-axis
    plt.legend(loc="best")
    plt.show()