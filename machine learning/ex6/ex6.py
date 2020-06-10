import numpy as np
from scipy.io import loadmat
from sklearn.svm import SVC
import matplotlib.pyplot as plt

fig1 = plt.figure(1, figsize=(17,5))
plt1 = fig1.add_subplot(1, 3, 1)
plt2 = fig1.add_subplot(1, 3, 2)
plt3 = fig1.add_subplot(1, 3, 3)

data = loadmat('ex6data1.mat')
X = data['X']
y = data['y']

# plot the samples
plt1.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')

# 'poly', 'rbf', 'sigmoid'
clf = SVC(kernel='linear', C=10)
clf.fit(X, y.ravel())

# ax = plt1.gca()
xlim = plt1.get_xlim()
ylim = plt1.get_ylim()
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T

# get the separating hyperplane
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
plt1.contour(XX, YY, Z, colors='k', levels=[0], alpha=1, linestyles=['--'])

# ======================================================================================

data = loadmat('ex6data2.mat')
X = data['X']
y = data['y']

# plot the samples
plt2.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')

# 'poly', 'rbf', 'sigmoid'
clf = SVC(kernel='rbf', C=1e6)
clf.fit(X, y.ravel())

# ax = plt1.gca()
xlim = plt2.get_xlim()
ylim = plt2.get_ylim()
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T

# get the separating hyperplane
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
plt2.contour(XX, YY, Z, colors='k', levels=[0], alpha=1, linestyles=['--'])

# ======================================================================================

data = loadmat('ex6data3.mat')
X = data['X']
y = data['y']
X_test, y_test = data['Xval'], data['yval']

# plot the samples
plt3.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')

best_score = 0
best_C = 0
best_g = 0
for C in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 50, 100, 1000, 10000]:
    for g in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 50, 100, 1000, 10000]:

        clf = SVC(kernel='rbf', C=C, gamma=g)
        clf.fit(X, y.ravel())
        score = clf.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_C = C
            best_g = g

print(best_score, best_C, best_g)
# 'poly', 'rbf', 'sigmoid'
clf = SVC(kernel='rbf', C=best_C, gamma=best_g)
clf.fit(X, y.ravel())

# ax = plt1.gca()
xlim = plt3.get_xlim()
ylim = plt3.get_ylim()
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T

# get the separating hyperplane
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
plt3.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.bone)
# plt3.contour(XX, YY, Z, colors='k', levels=[0], alpha=1, linestyles=['--'])


plt.show()