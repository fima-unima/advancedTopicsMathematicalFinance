import numpy as np  # Import packages numpy and matplotlib
import matplotlib.pyplot as plt 
import pandas as pd
import pandas_datareader as pdr
import yfinance as yf      # Note: Yahoo-Finance changed their API in late 2022, see https://pypi.org/project/yfinance/. Using yfinance should fix arising problems
yf.pdr_override()
# Request data via Yahoo public API
data_CocaCola = pdr.get_data_yahoo('KO') # access to the data
data_Nvidia = pdr.get_data_yahoo('NVDA')
data_Apple = pdr.get_data_yahoo('AAPL')

days = 180 # time period that we consider

#Technical stuff that I just need to choose tha data from the correct time horizon--------------------
data_CocaCola["Number"] = range(data_Apple.shape[0])
data_Nvidia["Number"] = range(data_Apple.shape[0])
data_Apple["Number"] = range(data_Apple.shape[0])
start = data_CocaCola.loc[["2021-06-28"]]["Number"]

data_CocaCola_short = data_CocaCola[(data_CocaCola["Number"]>=int(start)) & (data_CocaCola["Number"]<int(start)+days)] #the data in the time period that we consider
data_Nvidia_short = data_Nvidia[data_Nvidia["Number"]>=int(start) & (data_Nvidia["Number"]<int(start)+days)]
data_Apple_short = data_Apple[data_Apple["Number"]>=int(start) & (data_Apple["Number"]<int(start)+days)]
#-----------------------------------------------------------------------------------------------------

np_CocaCola = data_CocaCola_short.to_numpy() # convert to numpy arrays
np_Nvidia = data_Nvidia_short.to_numpy()
np_Apple = data_Apple_short.to_numpy()

prices = np.array([np_CocaCola[-days:,2], np_Nvidia[-days:,2], np_Apple[-days:,2]]) # use the opening prices

log_increments = np.zeros([prices.shape[0], prices.shape[1]-1]) # number of increments
for i in range(prices.shape[0]):
    for j in range(prices.shape[1]-1):
        log_increments[i,j] = np.log(prices[i,j+1]/prices[i,j])

n = log_increments.shape[1] # number of increments
R_mean = [np.sum(log_increments[0,:])/n, np.sum(log_increments[1,:])/n, np.sum(log_increments[2,:])/n]  # mean log-increments

sample_variance = np.zeros(prices.shape[0])
for i in range(prices.shape[0]): # calculation of the sample variances
    for j in range(prices.shape[1]-1):
        sample_variance[i] = sample_variance[i] + (log_increments[i,j]-R_mean[i])**2
sample_variance = sample_variance/(n-1)

vola_estimator = np.sqrt(253*sample_variance) # estimated volatility
print("The "+str(days)+"d volatilities are: Coca Cola " + str(vola_estimator[0]) + ", Nvidia " + str(vola_estimator[1]) + ", Apple " + str(vola_estimator[2]) +".")