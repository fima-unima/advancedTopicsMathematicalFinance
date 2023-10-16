from unicodedata import unidata_version
import numpy as np  # Import packages numpy and matplotlib
import matplotlib.pyplot as plt 
from scipy.stats import norm  # Import norm to get access to the normal distribution function

T = 5 # parameters
S0 = 1
r = 0.03
mu = 0.06
sigma = 0.5
p = 0.5
N = 1
N_MC = 100000
V0 = 1
epsilon = 0.1


def brownian_mot(N,T):  # Function to simulate a Brownian motion
    mu, sigma = 0, np.sqrt(1/N) # Mean and standard deviation 
    s = np.random.normal(mu, sigma, N*T) # N*T normal distributed random numbers 
    # Create realization of Brownian Motion 
    B = np.zeros(N*T+1)  # Vector creation
    for i in range(N*T):
        B[i+1]=B[i]+s[i] # Increments of BM on each time step
    return B   # Return Brownian motion

def w(t, x): # value function w
    return np.exp(((mu-r)**2/(2*sigma**2)*p/(1-p) + r*p)*(T-t)) * x**p/p

#Exercise Part i)
time = np.linspace(0.0, 10, 101)
value = np.linspace(0, 100, 101)

Time, Value = np.meshgrid(time, value)
W = w(Time,Value)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(W, Time, Value, 50, cmap='binary')
ax.set_xlabel('Portfolio Value x')
ax.set_ylabel('Time t')
plt.suptitle('Plot of ' + r'$w(t,x).$')
plt.title('for '+r'$U(x)=2\sqrt{x}.$', fontsize=6)
plt.show()


#Part ii)
pi_star = (mu-r)/(sigma**2 * (1-p))  # Strategies
pi_star2 = (mu-r)/(sigma**2 * (1-p)) + epsilon
pi_star3 = (mu-r)/(sigma**2 * (1-p))-epsilon
u = np.zeros(T)  # variables for the simulated utility functions
u2 = np.zeros(T)
u3 = np.zeros(T)
for t in range(T):
    u_V = np.zeros(N_MC)
    u_V2 = np.zeros(N_MC)
    u_V3 = np.zeros(N_MC)
    for i in range(N_MC): #Momte Carlo experiment: simulate N_MC stock movements and calulate the respective terminal utilities
        B = brownian_mot(N,t)
        V = V0 * np.exp((pi_star*mu+(1-pi_star)*r -(pi_star*sigma)**2/2)*t + pi_star*sigma*B[N*t]) # value of pi*+epsilon at the end
        V2 = V0 * np.exp((pi_star2*mu+(1-pi_star2)*r -(pi_star2*sigma)**2/2)*t + pi_star2*sigma*B[N*t]) # value of pi*+epsilon at the end
        V3 = V0 * np.exp((pi_star3*mu+(1-pi_star3)*r -(pi_star3*sigma)**2/2)*t + pi_star3*sigma*B[N*t]) # value of pi*-epsilon at the end
        u_V[i] = 1/p *V**p
        u_V2[i] = 1/p *V2**p
        u_V3[i] = 1/p *V3**p
    u[t] = np.mean(u_V)
    u2[t] = np.mean(u_V2)
    u3[t] = np.mean(u_V3)

#optimal value function 
w2 = np.zeros(T)
for t in range(T):
    w2[t] = w(T-t,V0) # we want to plot the remaining time on the x axis, that's why I apply w to T-t
t = np.linspace(0,T-1,T)
plt.xlabel("Remaining time T-t")
plt.title("Comparison of (close to) optimal trading strategies")
plt.plot(t,w2, label="Optimal value function "+r'$v(T-t,V_0)$')
plt.plot(t,u, label=r'$\pi^*$')
plt.plot(t,u2, label=r'$\pi^*+\epsilon$')
plt.plot(t,u3, label=r'$\pi^*-\epsilon$')
plt.legend(loc="best")
plt.show()


