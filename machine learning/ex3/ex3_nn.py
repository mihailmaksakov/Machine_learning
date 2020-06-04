from scipy.io import loadmat
from PIL import Image, ImageOps
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from joblib import dump, load


def prepare_image(filename):
    image = Image.open(filename)
    inverted_image = ImageOps.invert(ImageOps.grayscale(image))
    inverted_image.save(f'i{filename}')
    return np.ravel(np.flipud(np.rot90(np.asarray(inverted_image, dtype="uint8"), k=3)))


def prepare_images(file_list):
    return np.array([prepare_image(f) for f in file_list])


# loading data from mathlab file
data = loadmat('ex3data1.mat')
X = data['X']
y = data['y']

X_train, X_test, y_train, y_test = train_test_split(X, np.ravel(y), stratify=y)

mlp = load('filename.joblib')
mlp1 = MLPClassifier(alpha=0.9, hidden_layer_sizes=(25, ), max_iter=1000, activation='logistic').fit(X_train, np.ravel(y_train))
mlp2 = MLPClassifier(alpha=0.9, hidden_layer_sizes=(25, ), max_iter=1000, activation='tanh').fit(X_train, np.ravel(y_train))
mlp3 = MLPClassifier(alpha=0.9, hidden_layer_sizes=(25, ), max_iter=1000, activation='relu').fit(X_train, np.ravel(y_train))
mlp4 = MLPClassifier(alpha=0.9, hidden_layer_sizes=(25, ), max_iter=1000, activation='identity').fit(X_train, np.ravel(y_train))
# dump(clf, 'filename.joblib')

print(mlp1.predict(prepare_images(['IMG_0002.png', '2.png', '3.png'])))
print(mlp1.score(X_test, np.ravel(y_test)))

plt.figure()

# fig.plot(mlp.loss_curve_)
plt.plot(mlp.loss_curve_, label='saved relu')
plt.plot(mlp1.loss_curve_, label='logistic')
plt.plot(mlp2.loss_curve_, label='tanh')
plt.plot(mlp3.loss_curve_, label='relu')
plt.plot(mlp4.loss_curve_, label='identity')
leg = plt.legend()
# plt.show()

fig, axes = plt.subplots(5, 5)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(20, 20), cmap=plt.cm.gray, vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()
