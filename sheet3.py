import numpy as np  # Import packages numpy and matplotlib
import matplotlib.pyplot as plt 
from scipy.stats import norm  # Import norm to get access to the normal distribution function

T = 5 # Time horizon
S0 = 2
K = 2
r = 0.05
sigma = 0.1

def black_scholes_price(T,S0,K,r,sigma):
    d1 = 1/(sigma*np.sqrt(T))*( np.log(S0/K)+(r+(sigma**2)/2)*T )
    d2 = d1-sigma*np.sqrt(T)
    return S0*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)

# Function in terms of T
values = np.zeros(100) # T from 1 to 100
for i in range(100):
    values[i] = black_scholes_price(i,S0,K,r,sigma)
x = np.linspace(1,100,100)
plt.plot(x,values, label="Black-Scholes-price")
plt.title("Call price for variable T")
plt.legend(loc="best")
plt.xlabel("T")
plt.show()

# Function in terms of S0
values = np.zeros(51) # S0 from 0 to 5
for i in range(50+1):
    values[i] = black_scholes_price(T,i/10,K,r,sigma)
x = np.linspace(0,5,51)
plt.plot(x,values, label="Black-Scholes-price")
plt.title("Call price for variable "+  r'$S_0$')
plt.legend(loc="best")
plt.xlabel(r'$S_0$')
plt.show()

# Function in terms of K
values = np.zeros(50) # K from 0.1 to 5
for i in range(50):
    values[i] = black_scholes_price(T,S0,(i+1)/10,r,sigma)
x = np.linspace(0.1,5,50)
plt.plot(x,values, label="Black-Scholes-price")
plt.title("Call price for variable K")
plt.legend(loc="best")
plt.xlabel("K")
plt.show()

# Function in terms of r
values = np.zeros(101) # r from 0 to 1
for i in range(100+1):
    values[i] = black_scholes_price(T,S0,K,i/100,sigma)
x = np.linspace(0,1,101)
plt.plot(x,values, label="Black-Scholes-price")
plt.title("Call price for variable r")
plt.legend(loc="best")
plt.xlabel("r")
plt.show()

# Function in terms of sigma
values = np.zeros(101) # sigma from 0 to 1
for i in range(100+1):
    values[i] = black_scholes_price(T,S0,K,r,i/100)
x = np.linspace(0,1,101)
plt.plot(x,values, label="Black-Scholes-price")
plt.title("Call price for variable "+  r'$\sigma$')
plt.legend(loc="best")
plt.xlabel(r'$\sigma$')
plt.show()