import numpy as np
import matplotlib.pyplot as plt
import os

csv_path = os.path.join(os.path.dirname(__file__), "goldstock.csv")

# # broadcasting
# a = np.ones((3, 4))
# b = np.ones(3).reshape(3, 1)
# c = a + b # error can't broadcast
# print(c)

# # axis operation
# d = np.sum(a, axis=1)
# print("======",d)
# e = np.sum(a, axis=0)
# print("------",e)

import csv

dates = []
prices = []

with open(csv_path, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        dates.append(row[1])
        price = float(row[2].replace('$',''))
        prices.append(price)

dates = np.array(dates)
prices = np.array(prices)

print("prices length:", len(prices))

# plt.figure(figsize=(12,6))
# plt.plot(dates, prices)
# plt.title('Stock Prices Over Time')
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.xticks(rotation=45)
# plt.grid(True)
# plt.show()

def simple_moving_average(prices, window):
    sma = []
    for i in range(len(prices)):
        if i < window:
            sma.append(None)
        else:
            sma.append(np.mean(prices[i-window:i]))

    return np.array(sma)

# sma_10 = simple_moving_average(prices, 10)

# # plot sma along with original prices
# plt.figure(figsize=(12,6))
# plt.plot(dates, prices, label='Original Prices')
# plt.plot(dates, sma_10, label='10-Day SMA', color='orange')
# plt.title('Stock Prices with 10-Day SMA')
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.xticks(rotation=45)
# plt.legend()
# plt.grid(True)
# plt.show()

window_size = 5

X = []
y = []


for i in range(len(prices) - window_size):
    X.append(prices[i:i+window_size])
    y.append(prices[i+window_size])


X = np.array(X)
y = np.array(y)


y_pred_sma = np.mean(X, axis=1)
# # Plotting
# plt.figure(figsize=(14,7))
# plt.plot(y, label='Actual Price', color='blue')
# plt.plot(y_pred_sma, label='Predicted Price', color='red')
# plt.legend()
# plt.show()
alpha = 0.1  # Smoothing factor for EMA
y_pred_ema = np.zeros(len(y))

y_pred_ema[0] = X[0, -1]  # Initialize the first value
for i in range(1, len(y_pred_ema)):
    y_pred_ema[i] = alpha * X[i, -1] + (1 - alpha) * y_pred_ema[i - 1]

plt.figure(figsize=(14,7))
plt.plot(y, label='Actual Price', color='blue')
plt.plot(y_pred_sma, label='Predicted Price (SMA)', color='red')
plt.plot(y_pred_ema, label='Predicted Price (EMA)', color='green')
plt.legend()
plt.show()