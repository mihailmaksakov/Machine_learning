import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# loading data from mathlab file
data = loadmat('ex7data1.mat')
X = data['X']

X = (X - X.mean(axis=0)) / X.std(axis=0)

plt.scatter(X[:, 0], X[:, 1], marker='o', alpha=0.5)

pca = PCA(n_components=2)
X1 = pca.fit_transform(X)

X2 = np.dot(X - pca.mean_, pca.components_[0])
X2 = np.c_[(X2.T, np.zeros_like(X2).T)]

X_rec = pca.inverse_transform(X2)

plt.scatter(X_rec[:, 0], X_rec[:, 1], marker='*')

for i in range(X.shape[0]):
    plt.plot([X[i, 0], X_rec[i, 0]], [X[i, 1], X_rec[i, 1]], c='gray', linestyle='--', linewidth=1)

plt.arrow(*pca.mean_, *pca.components_[0], head_width=0.2, head_length=0.3, fc='black', ec='black')
plt.arrow(*pca.mean_, *pca.components_[1], head_width=0.2, head_length=0.3, fc='black', ec='black')

plt.axis('equal')
plt.grid()
plt.show()