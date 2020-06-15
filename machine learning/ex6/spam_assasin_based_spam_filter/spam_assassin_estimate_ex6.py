from joblib import dump, load

import numpy as np

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

X, y = np.load('spam_assasin_based_spam_filter/X.npy'), np.load('spam_assasin_based_spam_filter/y.npy')

min_max_scaler = preprocessing.MinMaxScaler()
X_minmax = min_max_scaler.fit_transform(X)

print(np.argwhere(np.isnan(X_minmax)))

X_train, X_test, y_train, y_test = train_test_split(X_minmax, y, test_size=0.2)

param_grid = {'C': [0.01, 0.1, 0.8, 1, 10],
              'gamma': [1], }

estimator = GridSearchCV(
    SVC(kernel='linear'), param_grid
)

# best_score = 0
# best_C = 0
# #0.8
# best_g = 0
# #1
# for C in [0.01, 0.1, 0.8, 1, 10]:
#     for g in [1]:
#
#         clf = SVC(kernel='linear', C=C, gamma=g)
#         clf.fit(X_train, y_train)
#         scores = cross_val_score(clf, X_train, y_train, cv=5)
#         score = scores.mean()
#         # print(score)
#         if score > best_score:
#             best_score = score
#             best_C = C
#             best_g = g
#
# print(best_score, best_C, best_g)

# estimator = SVC(kernel='linear', C=best_C, gamma=best_g, verbose=True)
grid_search_res = estimator.fit(X_train, y_train)
print(grid_search_res.best_estimator_)

best_estimator = grid_search_res.best_estimator_
dump(best_estimator, 'spam_assasin_based_spam_filter/spam_assasin.linear.clf')
# clf = load('spam_assasin_based_spam_filter/spam_assasin.linear.clf')

# verify model
train_errors = np.empty((0, 2))
validation_errors = np.empty((0, 2))

for m in range(100, X_train.shape[0], 500):
    current_mse_train = np.empty(2)
    current_mse_test = np.empty(2)
    for j in range(2):
        indices = np.random.choice(X_train.shape[0], size=m + 1, replace=False)
        X_tmp = X_train[indices, :]
        y_tmp = y_train[indices, :]
        clf = best_estimator.fit(X_tmp, y_tmp.ravel())
        current_mse_train[j-1] = mean_squared_error(clf.predict(X_tmp), y_tmp)
        current_mse_test[j-1] = mean_squared_error(clf.predict(X_test), y_test)

    train_errors = np.vstack((train_errors, np.array([[np.mean(current_mse_train), m]])))
    validation_errors = np.vstack((validation_errors, np.array([[np.mean(current_mse_test), m]])))

print(f'min test error: {min(validation_errors[:, 0])}')

plt.figure()

plt.plot(train_errors[:, 1], train_errors[:, 0], label='train')
plt.plot(validation_errors[:, 1], validation_errors[:, 0], label='test')
leg = plt.legend()

plt.show()
