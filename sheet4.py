import numpy as np  # Import packages numpy and matplotlib
import matplotlib.pyplot as plt 
from scipy.stats import norm  # Import norm to get access to the normal distribution function
from mpl_toolkits import mplot3d

# Model parameters
r = 0.01
sigma = 0.05
K = 60
S0 = 60


# Calculate the up-and-out-call-price dependend on the barrier B and the time to maturity T
def barrier_price(b, t):
    l1 = np.log(S0/K)  # helpful variables for the pricing formula
    l2 = np.log((b**2)/(K*S0))
    l3 = np.log(S0/B)
    l4 = np.log(B/S0)
    r1 = (r+(sigma**2)/2)*t
    r2 = (r-(sigma**2)/2)*t
    help1 = norm.cdf( (l1+r1)/(sigma*np.sqrt(t)) )-norm.cdf( (l3+r1)/(sigma*np.sqrt(t)) ) - (b/S0)**(1+2*r/(sigma**2))* ( norm.cdf((l2+r1)/(sigma*np.sqrt(t))) - norm.cdf((l4+r1)/(sigma*np.sqrt(t)))  )
    help2 = norm.cdf( (l1+r2)/(sigma*np.sqrt(t)) )-norm.cdf( (l3+r2)/(sigma*np.sqrt(t)) ) - (b/S0)**(2*r/(sigma**2)-1)* ( norm.cdf((l2+r2)/(sigma*np.sqrt(t))) - norm.cdf((l4+r2)/(sigma*np.sqrt(t))) )
    return S0*help1 - np.exp(-r*t)*K*help2

b = np.linspace(70, 90, 201)
t = np.linspace(0.01, 5, 500)

B, T = np.meshgrid(b, t)
P = barrier_price(B, T)



fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(B, T, P, 50, cmap='binary')
ax.set_xlabel('Barrier')
ax.set_xlim(70, 90)
ax.set_ylabel('Time to maturity in years')
ax.set_ylim(0, 5)
plt.suptitle("Value of an up-and-out call option,")
plt.title('with ' + r'$r=$' + str(r) +', ' + r'$\sigma=$' + str(sigma)+', '+ r'$K=$' + str(K)+', '+ r'$S_0=$' + str(S0)+'.', fontsize=8)
plt.show()
