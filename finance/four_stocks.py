from numpy.core.fromnumeric import mean
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np

path = 'four_stocks.csv'
stock_data = pd.read_csv(path)
#print(stock_data.head())
selected = list(stock_data.columns[1:])

plt.figure(figsize=(14,7))
for c in stock_data.columns[1:].values:
        plt.plot(stock_data.Date, stock_data[c], lw=3, alpha=0.8, label=c)
plt.legend(loc='upper left', fontsize=12)
plt.xticks(stock_data.Date, rotation='vertical')
plt.ylabel('price in $')
plt.tight_layout()
plt.show()

# data sets
returns_weekly = stock_data[selected].pct_change()
mean_returns = returns_weekly.mean()
cov_matrix = returns_weekly.cov()
num_portfolios=25000
risk_free_rate = 0.0163/52

np.random.seed(42)
num_portfolios = 6000
all_weights = np.zeros((num_portfolios, len(stock_data.columns)-1))
ret_arr = np.zeros(num_portfolios)
vol_arr = np.zeros(num_portfolios)
sharpe_arr = np.zeros(num_portfolios)

for x in range(num_portfolios):
        #weights
        weights = np.array(np.random.random(4))
        weights = weights/np.sum(weights)
        all_weights[x,:] = weights

        #return
        ret_arr[x] = np.sum((returns_weekly.mean() * weights * 54))

        #volatility
        vol_arr[x] = np.sqrt(np.dot(weights.T, np.dot(returns_weekly.cov()*54, weights)))

        #sharpe
        sharpe_arr[x] = ret_arr[x]/vol_arr[x]

print('Max Sharpe ratio in the array: {}'.format(sharpe_arr.max()))
print('Its location in the array: Row {}'.format(sharpe_arr.argmax()))

max_sr_ret= ret_arr[sharpe_arr.argmax()]
max_sr_vol= vol_arr[sharpe_arr.argmax()]

plt.figure(figsize=(12,8))
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.scatter(max_sr_vol, max_sr_ret,c='red', s=50) # red dot
plt.show()
