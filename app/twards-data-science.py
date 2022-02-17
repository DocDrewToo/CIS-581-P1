from re import X
import numpy
from scipy.linalg import lstsq
from sklearn.preprocessing import PolynomialFeatures


def x_data_in_polynomial_matrix(x_data, _degree):
    x_reshaped = x_data.reshape(-1,1)
    polynomial_features = PolynomialFeatures(degree=_degree)
    X = polynomial_features.fit_transform(x_reshaped)
    return X


def transform(x_data):
    


def fit(X, y, solver):
    # X : array-like of shape (n_samples, n_features)
    # y : array-like of shape (n_samples, 1)
    if solver == 'lstsq':
        coefecients, residues, rank, singular = lstsq(X,y)
    return coefecients, residues, rank, singular 


def predict(coefecients, X):
    x_predictions = X.dot(coefecients)
    return x_predictions


def seperate_data_to_x_y_arrays(input_data_file_name):    
    x_axis = []
    y_axis = []
    with open(input_data_file_name) as input_data_object:
        for line in input_data_object:
            row = line.strip().split(" ")
            x_axis.append(row[0])
            y_axis.append(row[1])

    x_data_numpy = numpy.array(x_axis, dtype="int")
    y_data_numpy = numpy.array(y_axis, dtype="float")
    return x_data_numpy, y_data_numpy


data = "data/test.txt"
# data = "data/deficit_train.dat"
x_data, y_data = seperate_data_to_x_y_arrays(data)
degree = 2
transform(x_data)
x_matrix = x_data_in_polynomial_matrix(x_data, degree)
coeficients, _, _, _ = fit(x_matrix, y_data, "lstsq")
hypothesis = predict(coeficients, x_matrix)
print(hypothesis)