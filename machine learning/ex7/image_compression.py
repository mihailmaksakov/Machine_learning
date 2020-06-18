from sklearn.cluster import KMeans

from PIL import Image
from numpy import asarray
import numpy as np

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

n_colors = 64


def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image


# load the image
image = Image.open('_DSC0675_1.jpg')

data = asarray(image, dtype=np.float64) / 255

plt.figure(1)
plt.clf()
plt.axis('off')
plt.imshow(data)

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(data.shape)
assert d == 3
image_array = np.reshape(data, (w * h, d))

kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array)

labels = kmeans.predict(image_array)

plt.figure(2)
plt.clf()
plt.axis('off')
recreated_image = recreate_image(kmeans.cluster_centers_, labels, w, h)
plt.imshow(recreated_image)

recreated_image_r = recreated_image.ravel()
min_x = min(recreated_image_r)
max_x = max(recreated_image_r)
x_variance = max_x - min_x

if min_x < 0:
    X_pic = recreated_image + abs(min_x)
else:
    X_pic = recreated_image

X_pic_p = (X_pic*255/x_variance).astype(np.uint8)

im = Image.fromarray(X_pic_p, mode='RGB')
im.save("_DSC0675_1_c.jpeg")

plt.figure(3)
plt.clf()
ax = plt.axes(projection='3d')
ax.scatter(image_array[:, 0], image_array[:, 1], image_array[:, 2], c=labels, cmap='coolwarm', marker=',')

pca = PCA(n_components=2)
image_array_2d = pca.fit_transform(image_array)

plt.figure(4)
plt.clf()
plt.scatter(image_array_2d[:, 0], image_array_2d[:, 1], c=labels, cmap='coolwarm', marker=',')

plt.show()