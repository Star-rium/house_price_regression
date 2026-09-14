import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')

X_train = train_df.drop(columns=['price']).values.astype(np.float64)
Y_train = train_df['price'].values.astype(np.float64)
X_test = test_df.drop(columns=['price']).values.astype(np.float64)
Y_test = test_df['price'].values.astype(np.float64)

mean_X = np.mean(X_train, axis=0)
std_X = np.std(X_train, axis=0)
std_X[std_X == 0] = 1.0

# standardize the features
X_train = (X_train - mean_X) / std_X
X_test = (X_test - mean_X) / std_X

m_train = X_train.shape[0] # bias term
X_train = np.c_[np.ones((m_train, 1)), X_train]

m_test = X_test.shape[0]
X_test = np.c_[np.ones((m_test, 1)), X_test]

class LinearRegression:
    def __init__(self, alpha, epochs):
        self.learning_rate = alpha
        self.epochs = epochs
        self.theta = None
        self.cost_history = []

    def compute_cost(self, X, Y):
        m = len(Y)
        predictions = X.dot(self.theta)
        cost = (1 / (2 * m)) * np.sum((predictions - Y) ** 2) # MSE
        return cost
    
    def fit(self, X, Y):
        m, n = X.shape
        self.theta = np.zeros(n)
        
        for epoch in range(self.epochs):
            predictions = X.dot(self.theta)
            errors = predictions - Y
            gradient = (1 / m) * X.T.dot(errors)
            self.theta -= self.learning_rate * gradient
            
            cost = self.compute_cost(X, Y)
            self.cost_history.append(cost)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Cost {np.sqrt(cost)}")

    def predict(self, X):
        return X.dot(self.theta)

    def lasso_fit(self, X, Y, Lambda):
        m, n = X.shape
        self.theta = np.zeros(n)
        
        for epoch in range(self.epochs):
            predictions = X.dot(self.theta)
            errors = predictions - Y
            gradient = (1 / m) * X.T.dot(errors) + (Lambda / m) * np.sign(self.theta)
            self.theta -= self.learning_rate * gradient
            
            cost = self.compute_cost(X, Y) + (Lambda / (2 * m)) * np.sum(np.abs(self.theta))
            self.cost_history.append(cost)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Cost {np.sqrt(cost)}")

    def get_r_squared(self, X, Y):
        predictions = self.predict(X)
        ss_total = np.sum((Y - np.mean(Y)) ** 2)
        ss_residual = np.sum((Y - predictions) ** 2)
        r_squared = 1 - (ss_residual / ss_total)
        return r_squared

model = LinearRegression(alpha=0.1, epochs=200)
# model.fit(X_train, Y_train)
# model.lasso_fit(X_train, Y_train, Lambda=20000000)

y_pred = model.predict(X_test)

# metrics
MAE = np.mean(np.abs(y_pred - Y_test))
MSE = np.mean((y_pred - Y_test) ** 2)
RMSE = np.sqrt(MSE)

ss_total = np.sum((Y_test - np.mean(Y_test)) ** 2)
ss_residual = np.sum((Y_test - y_pred) ** 2)
R_squared = 1 - (ss_residual / ss_total)

# testing different lambda values for Lasso regression
values = [0.1, 1, 10, 100, 1000, 10000, 100000, 200000, 500000, 1000000, 2000000, 3000000, 4000000]
r_squared_history_train = []
r_squared_history_test = []
for Lambda in values:
    model.lasso_fit(X_train, Y_train, Lambda=Lambda)
    r_squared_train = model.get_r_squared(X_train, Y_train)
    r_squared_history_train.append(r_squared_train)
    r_squared_history_test.append(R_squared)
    print(f"Lambda: {Lambda}, R-squared Train: {r_squared_train:.4f}, R-squared Test: {R_squared:.4f}")

# visualization of R-squared values for different lambda values
plt.figure(figsize=(10, 6))
plt.plot(values, r_squared_history_train, marker='o', label='Train R-squared')
plt.plot(values, r_squared_history_test, marker='o', label='Test R-squared')
plt.xscale('log')
plt.xlabel('Lambda (Regularization Parameter)')
plt.ylabel('R-squared')
plt.title('R-squared vs Lambda for Lasso Regression')
plt.legend()
plt.grid()
plt.show()

# print(f"Mean Absolute Error (MAE): {MAE:.2f}")
print(f"Mean Squared Error (MSE): {MSE:.2f}")
print(f"Root Mean Squared Error (RMSE): {RMSE:.2f}")
print(f"R-squared Train: {model.get_r_squared(X_train, Y_train):.4f}")
print(f"R-squared Test: {R_squared:.4f}")

# visualization parts

# # plotting the cost history
# plt.plot(model.cost_history)
# plt.title('Cost History over Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Cost')
# plt.grid()
# plt.show()

# # plotting predicted vs actual prices
# plt.scatter(Y_test, y_pred, alpha=0.5)
# plt.title('Predicted vs Actual Prices')
# plt.xlabel('Actual Prices')
# plt.ylabel('Predicted Prices')
# plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r')
# plt.grid()
# plt.show()