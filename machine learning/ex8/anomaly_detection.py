import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

data = loadmat('ex8data1.mat')

X = data['X']

plt.figure()

estimator = EllipticEnvelope(contamination=.015)

labels = estimator.fit_predict(X)

e1 = IsolationForest()
labels1 = e1.fit_predict(X)

xx, yy = np.meshgrid(np.linspace(min(X[:, 0]), max(X[:, 0]), 150),
                     np.linspace(min(X[:, 1]), max(X[:, 1]), 150))

# Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black', alpha=0.5)
#
# Z1 = e1.predict(np.c_[xx.ravel(), yy.ravel()])
# Z1 = Z1.reshape(xx.shape)
# plt.contour(xx, yy, Z1, levels=[0], linewidths=2, colors='red', alpha=0.2)

e2 = LocalOutlierFactor()
labels2 = e2.fit_predict(X)

# lof = np.reshape(e2.negative_outlier_factor_, X.shape)

plt.scatter(X[:, 0], X[:, 1], marker='o', c=labels2, s=np.power(e2.negative_outlier_factor_, -2)*100)
# plt.scatter(X[:, 0], X[:, 1], marker='x', color='black', alpha=0.4, s=10)


# plt.scatter(lof[:, 0], lof[:, 1])

plt.show()