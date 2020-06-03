from random import randint
from scipy.io import loadmat
from PIL import Image, ImageOps
import numpy as np
from sklearn.linear_model import LogisticRegression


def prepare_image(filename):
    image = Image.open(filename)
    inverted_image = ImageOps.invert(ImageOps.grayscale(image))
    inverted_image.save(f'i{filename}')
    return np.ravel(np.flipud(np.rot90(np.asarray(inverted_image, dtype="uint8"), k=3)))


# loading data from mathlab file
data = loadmat('ex3data1.mat')
X = data['X']
y = data['y']

# converting pixel brightness to 0-255 value format
X_ravel = X.ravel()
min_x = min(X_ravel)
max_x = max(X_ravel)
x_variance = max_x - min_x

if min_x < 0:
    X_pic = X + abs(min_x)
else:
    X_pic = X

X_pic_p = (X_pic*255/x_variance).astype(np.uint8)

# rebuilding pixel data for visualization
X_pic = np.vstack(
            tuple(
                np.hstack(
                    tuple(
                        np.reshape(X_pic_p[randint(0, X_pic_p.shape[0]-1)], (20, 20), order='F')
                        for a in range(20)
                    )
                ) for b in range(20)
            )
        )

img = Image.fromarray(np.reshape(X_pic, (400, 400)), 'L')
# img.save('my.png')
# img.show()

# learning logistic regression model
logreg = LogisticRegression(C=1e6, max_iter=10000)

logreg.fit(X_pic_p, np.ravel(y))

# testing model
p = logreg.predict(X_pic_p)

print(np.mean(np.ravel(y) == p) * 100)

ddd1 = prepare_image('IMG_0002.png')

print(logreg.predict([ddd1]))