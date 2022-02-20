from operator import index
import numpy
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
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


def seperate_to_folds(x, y, num_sections):
    x_array_of_arrays = numpy.array_split(x, num_sections)
    y_array_of_arrays = numpy.array_split(y, num_sections)
    return x_array_of_arrays, y_array_of_arrays


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


def plot_this(x, y, weights, _degree):
    # Raw Data
    plot.style.use('fivethirtyeight')
    plot.scatter(x, y, color='black')
    # plot.title("Stuff")
    min_x = numpy.amin(x)
    max_x = numpy.amax(x)
    num_of_x = x.size
    lots_of_xs = numpy.linspace(min_x, max_x)
    # lots_of_xs_numpy = numpy.array(lots_of_xs, dtype="int")
    lots_of_xs_matrix = x_data_in_polynomial_matrix(lots_of_xs, _degree)
    y_predicted = predict(lots_of_xs_matrix, weights)
    plot.plot(lots_of_xs, y_predicted)
    plot.show()


def normalize(_data, type="int"):
    scaler = StandardScaler()
    scaler.fit(_data.reshape(-1, 1))
    normalized_data = scaler.transform(_data.reshape(-1, 1))

    normal_array = []
    for value in normalized_data:
        normal_array.append(value[0])
    single_numpy_array = numpy.array(normal_array)

    average = numpy.average(single_numpy_array)
    standard_deviation = numpy.std(single_numpy_array)

    return scaler, single_numpy_array

def un_normalize(_data, scalar):
    un_normalized_data = scalar.inverse_transform(_data.reshape(-1, 1))

    normal_array = []
    for value in un_normalized_data:
        normal_array.append(value[0])
    single_numpy_array = numpy.array(normal_array)

    return single_numpy_array

def indexes_of_data(data, start_index):
    indexes = []
    for items in range(data.size):
        indexes.append(items + start_index)
    return indexes

training_dataset = "data/deficit_train.dat"
validation_dataset = "data/deficit_test.dat"

training_x , training_y = read_data_to_x_y_arrays(training_dataset)
validation_x , validation_y = read_data_to_x_y_arrays(validation_dataset)

training_x_scalar, training_x_normalized = normalize(training_x)
training_y_scalar, training_y_normalized = normalize(training_y)
validation_x_scalar, validation_x_normalized = normalize(validation_x)
validation_y_scalar, validation_y_normalized = normalize(validation_y)

folds = 6 # Split the training data into folds to use as mini testing data
my_lambda = [0, math.exp(-25), math.exp(-20), math.exp(-14),
   math.exp(-7), math.exp(-3), 1, math.exp(3), math.exp(7)]

totals = []
for degree in range(13):
    breakPoint = []
    # print("For degree:", degree)
    for _lambda in my_lambda:

        average_error_list = []
        hold_outs_x , hold_outs_y = seperate_to_folds(training_x_normalized, training_y_normalized, folds)
        for fold_n, hold_out_test_x  in enumerate(hold_outs_x):
            # Remove the hold_out[index] test data from the overall training data set
            # training_data_x = numpy.setdiff1d(training_x, hold_out_test_x)

            indexes_to_del = indexes_of_data(hold_out_test_x, fold_n * folds)
            
            training_data_x = numpy.delete(training_x_normalized, indexes_to_del)
            training_data_y = numpy.delete(training_y_normalized, indexes_to_del)
            training_x_features = x_data_in_polynomial_matrix(training_x_normalized, degree)
            weights = fit(training_x_features, training_y_normalized, _lambda)

            # cv_test_data_x , cv_test_data_y =
            un_normalize_hold_out_x = un_normalize(hold_out_test_x, training_x_scalar)
            un_normalize_hold_out_y = un_normalize(hold_outs_y[fold_n], training_y_scalar)
            hold_out_data_x_features =  x_data_in_polynomial_matrix(un_normalize_hold_out_x, degree)
            un_normalize_weights = un_normalize(weights, training_x_scalar)
            y_predicted = predict(hold_out_data_x_features, un_normalize_weights)
            error = root_mean_square_error(un_normalize_hold_out_y, y_predicted)
            average_error_list.append(error)
            # print("For lambda:", _lambda, "Error (rmse)", error)

        
        average_error = sum(average_error_list) / len(average_error_list)
        totals.append([degree, _lambda, average_error])

    # plot_this(un_normalize_hold_out_x, un_normalize_hold_out_y, un_normalize_weights, degree)

numpy_array_of_totals = numpy.array(totals)
min_error = numpy.amin(numpy_array_of_totals, axis=0)
what_be_here = numpy_array_of_totals[:,2]
indexes_of_minimums = numpy.where(numpy_array_of_totals[:,2] == min_error)

print("Min Error:", min_error[2])
