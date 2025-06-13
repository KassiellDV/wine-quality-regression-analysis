import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

# connect to the database
engine = create_engine('sqlite:///wine_data.db')

# load the table into a DataFrame
train = pd.read_sql('SELECT * From train_data', con=engine)
test = pd.read_sql('SELECT * From test_data', con=engine)


# separate features and target
X_train = train.drop(columns=['quality'])
Y_train = train['quality']

X_test = test.drop(columns=['quality'])
Y_test = test['quality']

# calculate means from training set

mean_X_train = X_train.mean(axis=0)

# center subtracting the means
X_train_centered = X_train - mean_X_train
X_test_centered = X_test - mean_X_train

# Solve Normal Equation
XtX = X_train_centered.T @ X_train_centered #X^T * X
Xty = X_train_centered.T @ Y_train #X^T * y

v_NE = np.linalg.solve(XtX, Xty)  # v_NE = (X^T * X)^-1 * (X^T * y)


# predict on test set
Y_test_pred = X_test_centered @ v_NE

# calculate the mean squared error and Mean Absolute Deviation error
mse = np.mean((Y_test - Y_test_pred) ** 2)
mad = np.mean(np.abs(Y_test - Y_test_pred))


# Qr decomposition
Q, R = np.linalg.qr(X_train_centered.values)

# Solve using Qr decomposition
v_QR = np.linalg.solve(R, Q.T @ Y_train)


# predict on test set using Qr decomposition
Y_test_pred_QR = X_test_centered @ v_QR

# calculate the mean squared error and Mean Absolute Deviation error for Qr decomposition
mse_QR = np.mean((Y_test - Y_test_pred_QR) ** 2)
mad_QR = np.mean(np.abs(Y_test - Y_test_pred_QR))

print("\n🔵 Comparison of Methods:")
print(f"mad (Normal Equations) : {mad:.4f}")
print(f"mad (QR Decomposition) : {mad_QR:.4f}")
print(f"mse (Normal Equations): {mse:.4f}")
print(f"mse (QR Decomposition): {mse_QR:.4f}")

# Plot setting
plt.figure(figsize=(10, 6))

# plot predicted vs actual values
plt.scatter(Y_test, Y_test_pred, color='blue', label='Normal Equations', marker='o', alpha=0.5)
plt.scatter(Y_test, Y_test_pred_QR, color='red', label='QR Decomposition', marker='*', alpha=0.5)

# predict line
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=2)

# labels and title
plt.xlabel('Actual Quality')
plt.ylabel('Predicted Quality')
plt.title('Predicted vs Actual Wine Quality')
plt.legend()
plt.grid(True)

# Show the plot
plt.savefig('prediction_plot.png')
print("✅ Plot saved as 'prediction_plot.png' successfully!")
