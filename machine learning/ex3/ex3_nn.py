from scipy.io import loadmat
from PIL import Image, ImageOps
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

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

clf = load('filename.joblib')
# clf = MLPClassifier(alpha=0.7, hidden_layer_sizes=(25, ), max_iter=1000).fit(X_train, np.ravel(y_train))
# dump(clf, 'filename.joblib')

print(clf.predict(prepare_images(['IMG_0002.png', '2.png', '3.png'])))
print(clf.score(X_test, np.ravel(y_test)))

