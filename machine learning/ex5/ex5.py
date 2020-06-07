import random

import numpy as np
from scipy.io import loadmat
import my_linear_regression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# loading data from mathlab file
data = loadmat('ex5data1.mat')
X = data['X']
y = data['y']
X_cv = np.vstack((X, data['Xval']))
y_cv = np.vstack((y, data['yval']))

plt.figure()
ax = plt.axes()
ax.scatter(X_cv, y_cv, marker='+', edgecolors='black', c='grey')

# LR = my_linear_regression.RegressionModel(my_linear_regression.RIDGE_MODEL, True, 8, regularization)

polynomial_degree = 8
alphas=np.array([0.001, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 1, 100])

LR = my_linear_regression.RegressionModel(model=my_linear_regression.RIDGECV_MODEL, normalize=True, polynomial_degree=polynomial_degree, cv=5, alphas=alphas)

LR.fit(X_cv, y_cv)

regularization = LR.linear_regression.alpha_
print(f'best regularization: {regularization}')

_, y_p = LR.predict_mesh((np.linspace(X_cv.min(), X_cv.max(), 100)))
ax.plot(np.linspace(X_cv.min(), X_cv.max(), 100), y_p, color='red')

LR = my_linear_regression.RegressionModel(my_linear_regression.RIDGE_MODEL, True, polynomial_degree, regularization)

train_errors = np.empty((0, 2))
validation_errors = np.empty((0, 2))

for m in range(X_cv.shape[0]):

    current_mse_train = np.empty(50)
    current_mse_test = np.empty(50)
    for j in range(50):
        indices = np.random.choice(X_cv.shape[0], size=m + 1, replace=False)
        X_train = X_cv[indices, :]
        y_train = y_cv[indices, :]
        LR.fit(X_train, y_train)
        current_mse_train[j-1] = mean_squared_error(LR.predict(X_train), y_train)
        current_mse_test[j-1] = mean_squared_error(LR.predict(data['Xtest']), data['ytest'])

    train_errors = np.vstack((train_errors, np.array([[np.mean(current_mse_train), m]])))
    validation_errors = np.vstack((validation_errors, np.array([[np.mean(current_mse_test), m]])))

print(f'min test error: {min(validation_errors[:, 0])}')

plt.figure()

# plt.plot(np.linspace(X.min(), X.max(), 100), y_p, color='black')

plt.plot(train_errors[:, 1], train_errors[:, 0], label='train')
plt.plot(validation_errors[:, 1], validation_errors[:, 0], label='test')
# plt.plot(validation_errors, label='test')
leg = plt.legend()

plt.ylim(0, 100)

plt.show()