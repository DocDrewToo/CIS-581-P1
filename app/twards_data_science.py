from re import X
import numpy
from scipy.linalg import lstsq
from sklearn.preprocessing import PolynomialFeatures
from scipy.special import factorial
from itertools import combinations_with_replacement


def seperate_data_to_x_y_arrays(input_data_file_name):    
    x_axis = []
    y_axis = []
    with open(input_data_file_name) as input_data_object:
        for line in input_data_object:
            row = line.strip().split(" ")
            x_axis.append(row[0])
            y_axis.append(row[1])

    x_numpy = numpy.array(x_axis, dtype="int")
    y_numpy = numpy.array(y_axis, dtype="float")
    return x_numpy, y_numpy

def x_data_in_polynomial_matrix(x_numpy, _degree):
    x_reshaped = x_numpy.reshape(-1,1)
    polynomial_features = PolynomialFeatures(degree=_degree)
    X = polynomial_features.fit_transform(x_reshaped)
    return X

# def features_output(x_features, _degree):
#     n_intput_features = x_features.shape[1]
#     numerator = factorial(n_intput_features + _degree)
#     denominator = factorial(_degree) * factorial(n_intput_features)
#     n_output_features = int(numerator / denominator) - 1
#     return n_output_features


# def transform(output_features, x_features, _degree):
#     #Transform the data into a polynomial feature matrix
#     input_features = x_features.shape[1]
#     # combinations of feature indices stored in tuples
#     combos = [combinations_with_replacement(range(input_features), index) 
#         for index in range(1, _degree + 1)]
    
#     combinations = [item for sublist in combos for item in sublist]
#     x_new = numpy.empty((x_features.shape[0], output_features))
#     for index, index_combos in enumerate(combinations):
#         x_new[:, index] = numpy.prod(x_features[:, index_combos], axis=1)
    
#     return x_new

def regularization_term(my_lambda, weights):
    my_lambda * 0.5 * numpy.linalg.norm(weights, 2)

def gradient_descent_penalty(weights, _lambda):
    penalty = numpy.asarray(_lambda) * weights
    # ?? Add 0 for a bias term ??
    bias0_penalty = numpy.insert(penalty, 0, 0, axis=0)
    return bias0_penalty


def fit(X, y, solver):
    # X : array-like of shape (n_samples, n_features)
    # y : array-like of shape (n_samples, 1)
    if solver == 'lstsq':
        coefecients, residues, rank, singular = lstsq(X,y)
    # if solver == 'ridge':
        
    return coefecients, residues, rank, singular 


def predict(coefecients, X):
    x_predictions = X.dot(coefecients)
    return x_predictions

data = "data/test.txt"
# data = "data/deficit_train.dat"
x_numpy, y_numpy = seperate_data_to_x_y_arrays(data)
degree = 2
x_features = x_data_in_polynomial_matrix(x_numpy, degree)
# output_features = features_output(x_features, degree)
fit(x_features, y_numpy, 'ridge')
"""
    Minimizes the cost fuction:
        J(w) = MSE(w) + alpha * 1/2 * ||w||^2
"""
my_lambda = [0, 1e-25, 1e-20, 1e-14, 1e-7, 1e-3, 1, 1e3, 1e7]
n_iterations = 1000
lr = 1e-1 # Learning rate determining the size of steps in batch gradient descent
regularization_term(my_lambda)
gradient_descent_penalty(weights, my_lambda)
# coeficients, _, _, _ = fit(x_matrix, y_data, "ridge")
# hypothesis = predict(coeficients, x_matrix)
# print(hypothesis)