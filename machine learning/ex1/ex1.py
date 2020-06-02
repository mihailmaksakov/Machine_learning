import numpy as np
import my_linear_regression
import matplotlib.pyplot as plt
from matplotlib import cm

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

fig1 = plt.figure(1, figsize=(12,5))
plt1 = fig1.add_subplot(1, 2, 1)
plt2 = fig1.add_subplot(1, 2, 2, projection='3d')

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

data = np.loadtxt('ex1data1.txt', delimiter=',')

X = data[:, 0]
y = data[:, 1]

LR = my_linear_regression.RegressionModel(my_linear_regression.RIDGE_MODEL, True, 6, 10**-8)
LR.fit(X, y)

_, y_p = LR.predict_mesh((np.linspace(X.min(), X.max(), 100)))

plt1.scatter(X, y, marker='+', edgecolors='black', c='grey')

plt1.plot(np.linspace(X.min(), X.max(), 100), y_p, color='black')

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

data = np.loadtxt('ex1data2.txt', delimiter=',')

X = data[:, 0:2]
y = data[:, 2]

plt2.scatter(X[:, 0], X[:, 1], y)

LR = my_linear_regression.RegressionModel(my_linear_regression.LINEAR_REGRESSION_MODEL, True)
LR.fit(X, y)

x_p, y_p = LR.predict_mesh(np.linspace(X.min(0)[0], X.max(0)[0], 10), np.linspace(X.min(0)[1], X.max(0)[1], 10))

surf1 = plt2.plot_surface(x_p[0], x_p[1], y_p, cmap=cm.coolwarm,
                          linewidth=1, antialiased=True, alpha=0.7)

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

plt2.set_zlim(100000, 1000000)

plt.show()