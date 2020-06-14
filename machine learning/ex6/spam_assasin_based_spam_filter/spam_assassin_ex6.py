import pickle

import numpy as np

from main_package import create_vocabulary
from main_package import read_files

# import pymorphy2

vocabulary = create_vocabulary(False)

X_spam, y_spam = read_files('spam_assasin_based_spam_filter/spam', 1, vocabulary)
X_nonspam, y_nonspam = read_files('spam_assasin_based_spam_filter/nonspam', 0, vocabulary)

X = np.vstack((X_spam, X_nonspam))
y = np.vstack((y_spam, y_nonspam))

print(np.argwhere(np.isnan(X)))
print(np.argwhere(np.isnan(y)))

np.save('spam_assasin_based_spam_filter/X.npy', X)
np.save('spam_assasin_based_spam_filter/y.npy', y)

pickle.dump(vocabulary, open("spam_assasin_based_spam_filter/vocabulary.pkl", "wb"))
