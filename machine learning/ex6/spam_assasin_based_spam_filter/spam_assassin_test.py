from joblib import dump, load
from main_package import create_vocabulary, read_files
from sklearn.metrics import mean_squared_error

import pickle

clf = load('spam_assasin_based_spam_filter/spam_assasin.linear.clf')

# vocabulary = create_vocabulary(False)
# pickle.dump(vocabulary, open("spam_assasin_based_spam_filter/vocabulary.pkl", "wb"))
vocabulary = pickle.load(open("spam_assasin_based_spam_filter/vocabulary.pkl", "rb"))

X_test, y_test = read_files('spam_assasin_based_spam_filter/unknown_emails/nonspam', 0, vocabulary)

P = clf.predict(X_test)

print(P, mean_squared_error(P, y_test))