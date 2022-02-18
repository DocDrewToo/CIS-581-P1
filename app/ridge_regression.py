import numpy
from sklearn.preprocessing import PolynomialFeatures

def read_data_to_x_y_arrays(input_data_file_name):    
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


def x_data_in_polynomial_matrix(x_numpy, _degree):
    x_reshaped = x_numpy.reshape(-1,1)
    polynomial_features = PolynomialFeatures(degree=_degree)
    X = polynomial_features.fit_transform(x_reshaped)
    return X

def fit(X, y, _lambda = 0):
    iota = numpy.identity(X.shape[1])
    left_matrix = numpy.linalg.inv(X.T @ X + _lambda * iota)
    right_matrix = X.T @ y
    coefficients = left_matrix @ right_matrix
    return coefficients

data = "data/test.txt"
x_numpy, y_numpy = read_data_to_x_y_arrays(data)
degree = 1
x_features = x_data_in_polynomial_matrix(x_numpy, degree)
coefficients = (x_features, y_numpy)
