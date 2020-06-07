import numpy as np
# from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import sklearn.linear_model as lm
# from sklearn.linear_model import Lasso

LINEAR_REGRESSION_MODEL = lm.LinearRegression
RIDGE_MODEL = lm.Ridge
LASSO_MODEL = lm.Lasso
RIDGECV_MODEL = lm.RidgeCV


class RegressionModel():

    def __init__(self, model, normalize=False, polynomial_degree=0, alpha=0, cv=None, alphas=None):

        if model == LINEAR_REGRESSION_MODEL:
            self.linear_regression = model(normalize=normalize)
        elif model == RIDGE_MODEL:
            self.linear_regression = model(alpha=alpha, normalize=normalize)
        elif model == LASSO_MODEL:
            self.linear_regression = model(alpha=alpha, normalize=normalize)
        elif model == RIDGECV_MODEL:
            self.linear_regression = model(alphas=alphas, normalize=normalize, cv=cv)
        else:
            raise NotImplementedError(f'model {model} not implemented!')

        self.polynomial_degree = polynomial_degree

    def prepare_x(self, x):

        if len(x.shape) == 1:
            poly_x = np.reshape(x, (x.shape[0], 1))
        else:
            poly_x = x

        if self.polynomial_degree!=0:
            poly_features = PolynomialFeatures(degree=self.polynomial_degree)
            poly_x = poly_features.fit_transform(poly_x)

        return poly_x

    def fit(self, x, y):
        self.linear_regression.fit(self.prepare_x(x), y)

    def predict(self, x):
        return self.linear_regression.predict(self.prepare_x(x))

    def predict_mesh(self, *x):
        """
        Return prediction y vector for input N range arrays
        """
        x_mesh = np.meshgrid(*x)
        x_p = x_mesh[0].ravel()
        for axis in range(1, len(x_mesh)):
            x_p = np.c_[x_p, x_mesh[axis].ravel()]
        x_p = x_p.reshape((x_p.shape[0], len(x)))
        y_p = self.linear_regression.predict(self.prepare_x(x_p))
        return x_mesh, y_p.reshape(tuple(len(x) for x in x))
