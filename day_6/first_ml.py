import numpy as np
# [size (sqft), rooms, age (years)]
X = np.array([
    [1500, 3, 10],
    [2000, 4, 5],
    [1700, 3, 15],
    [2500, 5, 2]
])

# Prices in $1000s
y = np.array([300, 400, 330, 500])
weights = np.random.rand(3)
bias = np.random.rand(1)

learning_rate = 0.00000001

for epoch in range(1000):
    predictions = X.dot(weights) + bias
    loss = np.mean((predictions - y) ** 2)
    
    # Calculate gradients manually
    gradient_w = 2 * X.T.dot(predictions - y) / X.shape[0]
    gradient_b = 2 * np.mean(predictions - y)
    
    # Update weights and bias
    weights -= learning_rate * gradient_w
    bias -= learning_rate * gradient_b
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss}")

print("Trained weights:", weights)
print("Trained bias:", bias)
predictions = X.dot(weights) + bias
print("Predictions:", predictions)
print("Actual prices:", y)
