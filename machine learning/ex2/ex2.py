import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


# loading dataset
data = np.loadtxt('ex2data1.txt', delimiter=',')

X = data[:, 0:2]
y = data[:, 2]

# fitting logistic model
logreg = LogisticRegression(C=1e5)

clf = logreg.fit(X, y)

# contour decision boundary (with colored probability map) and test data
x_min, x_max = X[:, 0].min()-3, X[:, 0].max()+3
y_min, y_max = X[:, 1].min()-3, X[:, 1].max()+3

h = 1  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

grid = np.c_[xx.ravel(), yy.ravel()]
probabilities = clf.predict_proba(grid)[:, 1].reshape(xx.shape)

f, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(xx, yy, probabilities, 50, cmap="coolwarm",
                      vmin=0, vmax=1)
ax_c = f.colorbar(contour)
ax_c.set_label("pass probability")
ax_c.set_ticks([0, .25, .5, .75, 1])

ax.contour(xx, yy, probabilities, levels=[0.5], cmap="Greys", vmin=-0.6, vmax=0.6)

ax.scatter(X[:, 0], X[:, 1], c=y[:], s=50,
           cmap="RdBu", vmin=-.2, vmax=1.2,
           edgecolor="white", linewidth=1)


# test prediction
X_pred = np.array([[50, 48], [78, 50], [95, 80]])

y_pred = logreg.predict(X_pred)

ax.scatter(X_pred[:, 0], X_pred[:, 1], c=y_pred[:], s=50,
           cmap="Oranges_r", vmin=-.2, vmax=1.2,
           edgecolor="black", linewidth=1)

# accuracy
p = logreg.predict(X)

print(f'model accuracy is {np.mean(p == y)*100}%')

# other
plt.setp(ax, xlabel='Exam #1 score', ylabel='Exam #2 score')

plt.show()
