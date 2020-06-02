import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

# loading dataset
data = np.loadtxt('ex2data2.txt', delimiter=',')

X = data[:, 0:2]
y = data[:, 2]

# fitting logistic model
poly_features = PolynomialFeatures(degree=4)
poly_x = poly_features.fit_transform(X)

logreg = LogisticRegression(C=1e10, max_iter=10000)

clf = logreg.fit(poly_x, y)

# contour decision boundary (with colored probability map) and test data
x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1

h = .01  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

grid = np.c_[xx.ravel(), yy.ravel()]
probabilities = clf.predict_proba(poly_features.fit_transform(grid))[:, 1].reshape(xx.shape)

f, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(xx, yy, probabilities, 50, cmap="coolwarm")
                      # ,
                      # vmin=0, vmax=1)
ax_c = f.colorbar(contour)
ax_c.set_label("pass probability")
ax_c.set_ticks([0, .25, .5, .75, 1])

ax.contour(xx, yy, probabilities, levels=[0.5], cmap="Greys", vmin=-0.6, vmax=0.6)

ax.scatter(X[:, 0], X[:, 1], c=y[:], s=50,
           cmap="RdBu", vmin=-.2, vmax=1.2,
           edgecolor="white", linewidth=1)

# accuracy
p = logreg.predict(poly_features.fit_transform(X))

print(f'model accuracy is {np.mean(p == y)*100:.2f}%')

# other
plt.setp(ax, xlabel='Exam #1 score', ylabel='Exam #2 score')

plt.show()


