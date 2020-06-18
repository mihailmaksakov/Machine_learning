import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from sklearn.decomposition import PCA

data = loadmat('ex8data2.mat')

X = data['X']

e1 = EllipticEnvelope()
labels1 = e1.fit_predict(X)

e2 = LocalOutlierFactor()
labels2 = e2.fit_predict(X)

n_components=3

pca1 = PCA(n_components=n_components)
Xproj = pca1.fit_transform(X)

plt.figure()
plt.clf()
ax = plt.axes(projection='3d')

# ax.scatter(image_array[:, 0], image_array[:, 1], image_array[:, 2], c=labels, cmap='coolwarm', marker=',')

ax.scatter(Xproj[:, 0], Xproj[:, 1], Xproj[:, 2], marker='o', c=labels1)

plt.show()