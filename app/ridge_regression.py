import numpy
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plot
from sklearn.metrics import mean_squared_error
import math

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
    #https://www.kaggle.com/residentmario/ridge-regression-proof-and-implementation/notebook
    iota = numpy.identity(X.shape[1])
    left_matrix = numpy.linalg.inv(X.T @ X + _lambda * iota)
    right_matrix = X.T @ y
    weights = left_matrix @ right_matrix
    return weights


def predict(X, weights):
    y_predicted = X @ weights
    return y_predicted


def root_mean_square_error(y_orig, y_predicted):
    mse = mean_squared_error(y_orig, y_predicted)
    rmse = math.sqrt(mse)
    return rmse


def plot_this(x, y, y_plot, _degree):
    # Raw Data
    plot.style.use('fivethirtyeight')
    plot.scatter(x, y, color='black')
    # plot.title("Stuff")
    # Polynomial from predictions
    curve = numpy.polyfit(x, y_plot, _degree) 
    poly = numpy.poly1d(curve)   
    plot.plot(x, y_plot,)
    plot.show()

data = "data/test.txt"
x_numpy, y_numpy = read_data_to_x_y_arrays(data)
# degree = 1
for degree in range(13):
    print("For degree:", degree)
    x_features = x_data_in_polynomial_matrix(x_numpy, degree)
    weights = fit(x_features, y_numpy)
    y_predicted = predict(x_features, weights)
    error = root_mean_square_error(y_numpy, y_predicted)
    print("Error (rmse)", error)
# plot_this(x_numpy, y_numpy, y_predicted, degree)
