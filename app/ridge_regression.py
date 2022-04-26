from operator import index
import numpy
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import matplotlib.pyplot as plot
from sklearn.metrics import mean_squared_error
import math
import pprint

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

def data_for_hypothesis_curve(data):
    min_x = numpy.amin(data)
    max_x = numpy.amax(data)
    num_of_xs = data.size
    lots_of_xs = numpy.linspace(min_x, max_x)
    return lots_of_xs

def x_data_in_polynomial_matrix(x_numpy, _degree):
    x_reshaped = x_numpy.reshape(-1,1)
    polynomial_features = PolynomialFeatures(degree=_degree)
    X = polynomial_features.fit_transform(x_reshaped)
    return X


def seperate_to_folds(x, y, num_sections):
    x_array_of_arrays = numpy.array_split(x, num_sections)
    y_array_of_arrays = numpy.array_split(y, num_sections)
    return x_array_of_arrays, y_array_of_arrays


def fit(x_1_dimension, y, _degree, _lambda = 0):
    #https://www.kaggle.com/residentmario/ridge-regression-proof-and-implementation/notebook

    X = x_data_in_polynomial_matrix(x_1_dimension, _degree)          

    iota = numpy.identity(X.shape[1])
    left_matrix = numpy.linalg.inv(X.T @ X + _lambda * iota)
    right_matrix = X.T @ y
    hypothesis = left_matrix @ right_matrix
    return hypothesis


def predict(x_1_dimension, hypothesis, _degree):
    X = x_data_in_polynomial_matrix(x_1_dimension, _degree)          

    y_predicted = X @ hypothesis
    return y_predicted


def root_mean_square_error(y_orig, y_predicted):
    mse = mean_squared_error(y_orig, y_predicted)
    rmse = math.sqrt(mse)
    return rmse

def normalize_w_scalar(_data, _scaler):
    normalized_data = _scaler.transform(_data.reshape(-1, 1))

    normal_array = []
    for value in normalized_data:
        normal_array.append(value[0])
    single_numpy_array = numpy.array(normal_array)

    average = numpy.average(single_numpy_array)
    standard_deviation = numpy.std(single_numpy_array)

    return single_numpy_array

def normalize(_data):
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


def lowest_error(_totals):
    numpy_array_of_totals = numpy.array(_totals, dtype=object)
    error_column = numpy_array_of_totals[:,2]
    min_error = numpy.amin(error_column, axis=0)

    indexes_of_minimums = numpy.where(error_column == min_error)
    return numpy_array_of_totals[indexes_of_minimums]


def lowest_error_per_degree(_totals, _degree):
    numpy_array_of_totals = numpy.array(_totals, dtype=object)
    degree_column = numpy_array_of_totals[:,0]
    indexes_of_degree = numpy.where(degree_column == _degree)
    degree_n_matrix = numpy_array_of_totals[indexes_of_degree]
    values_for_lowest_error = lowest_error(degree_n_matrix)
    return values_for_lowest_error


def error_on_test_data(x, y, x_scalar, y_scalar, best_algorithm):
    _degree = best_algorithm[0][0]
    _lambda = best_algorithm[0][1]
    hypothesis = best_algorithm[0][3]
    y_predicted = predict(x, hypothesis, _degree)

    x_un_normalized = un_normalize(x, x_scalar)
    y_predicted_un_normalized = un_normalize(y_predicted, y_scalar)
    y_un_normalized = un_normalize(y, y_scalar)
    error = root_mean_square_error(y_un_normalized, y_predicted_un_normalized)

    plot_data_y_predicted = predict(plot_data_x_normalized, hypothesis, _degree)
    plot_data_y_predicted_un_normalized = un_normalize(plot_data_y_predicted, y_scalar)
    plot_test_data(x_un_normalized, y_un_normalized, 
          plot_data_x, plot_data_y_predicted_un_normalized, _degree, _lambda)
    return error


def plot_test_data(actual_x, actual_y, curve_x, predeticted_curve_y, _degree, _lambda):
    plot.title("National Deficit - Test Results with optimized model")
    plot.scatter(actual_x, actual_y, color='black')

    my_label = "D:" + str(_degree)
    plot.plot(curve_x, predeticted_curve_y, label=my_label)
    plot.legend(loc="upper left")
    plot.xlabel("Year")
    plot.ylabel("Deficit (Billions)")
    plot.show
    plot.savefig("Validation_w_Optimized_model.png", dpi=480)
    plot.clf()


def plot_all_itterations(_totals):
    plot.title("National Deficit - All Degrees")
    for _degree in range(13):
        lowest_error = lowest_error_per_degree(_totals, _degree)
        hypotheses_for_this_lambda = lowest_error[0][3]
        best_lambda = lowest_error[0][1]
        plot_data_y_predicted = predict(plot_data_x_normalized, hypotheses_for_this_lambda, _degree)
        plot_data_y_predicted_un_normalized = un_normalize(plot_data_y_predicted, training_y_scalar)
       
        my_label = "D:" + str(_degree)
        plot.scatter(training_x, training_y, color='black')
        plot.plot(plot_data_x, plot_data_y_predicted_un_normalized, label=my_label)
    plot.xlabel("Year")
    plot.ylabel("Deficit (Billions)")
    plot.legend(loc="lower left")
    plot.savefig("Training_All_Degrees.png", dpi=480)
    plot.show
    plot.clf()

def plot_all_lambda_for_d_12(_totals):
    plot.title("National Deficit - D12 - All Lambdas")

    numpy_array_of_totals = numpy.array(_totals, dtype=object)
    degree_column = numpy_array_of_totals[:,0]
    indexes_of_degree = numpy.where(degree_column == 12)
    results_degree_12_matrix = numpy_array_of_totals[indexes_of_degree]

    for results in results_degree_12_matrix:
        degree = results[0]
        _lambda = results[1] 
        _hypotheses = results[3]
        plot_data_y_predicted = predict(plot_data_x_normalized, _hypotheses, degree)
        plot_data_y_predicted_un_normalized = un_normalize(plot_data_y_predicted, training_y_scalar)
       
        my_label = "D: " + str(degree) + " Lambda: " + str(_lambda)
        plot.scatter(training_x, training_y, color='black')
        plot.plot(plot_data_x, plot_data_y_predicted_un_normalized, label=my_label)

    plot.xlabel("Year")
    plot.ylabel("Deficit (Billions)")
    plot.legend(loc="lower left")
    plot.savefig("Lambda_For_D12.png", dpi=480)
    plot.show
    plot.clf()


training_dataset = "data/deficit_train.dat"
validation_dataset = "data/deficit_test.dat"

training_x , training_y = read_data_to_x_y_arrays(training_dataset)
validation_x , validation_y = read_data_to_x_y_arrays(validation_dataset)

validation_x_scalar, validation_x_normalized = normalize(validation_x)
validation_y_scalar, validation_y_normalized = normalize(validation_y)

plot_data_x = data_for_hypothesis_curve(training_x)
plot_data_x_scalar, plot_data_x_normalized = normalize(plot_data_x)


folds = 6 # Split the training data into folds to use as mini testing data

totals = []
for degree in range(13):
    my_lambda = [0]
    if degree == 12:
        my_lambda = [0, math.exp(-25), math.exp(-20), math.exp(-14),
                     math.exp(-7), math.exp(-3), 1, math.exp(3), math.exp(7)]
    for _lambda in my_lambda:
        average_error_list = []
        hold_outs_x , hold_outs_y = seperate_to_folds(training_x, training_y, folds)
        for fold_n, hold_out_test_x  in enumerate(hold_outs_x):
            # Remove the hold_out[index] test data from the overall training data set
            # training_data_x = numpy.setdiff1d(training_x, hold_out_test_x)

            indexes_to_del = indexes_of_data(hold_out_test_x, fold_n * folds)
            training_data_x = numpy.delete(training_x, indexes_to_del)
            training_data_y = numpy.delete(training_y, indexes_to_del)

            training_x_scalar, training_x_normalized = normalize(training_data_x)
            training_y_scalar, training_y_normalized = normalize(training_data_y)

            hypothesis = fit(training_x_normalized, training_y_normalized, degree, _lambda)

            hold_out_test_x_normalized = normalize_w_scalar(hold_out_test_x, training_x_scalar)
            y_predicted_normalized = predict(hold_out_test_x_normalized, hypothesis, degree)

            un_normalize_y_predicted = un_normalize(y_predicted_normalized, training_y_scalar)
            error = root_mean_square_error(hold_outs_y[fold_n], un_normalize_y_predicted)
            average_error_list.append(error)
   #matplot lib
        hypothesis = fit(training_x_normalized, training_y_normalized, degree, _lambda)
        average_error = sum(average_error_list) / len(average_error_list)
        totals.append([degree, _lambda, average_error, hypothesis])


plot_all_itterations(totals)

plot_all_lambda_for_d_12(totals)

print("__ALL TOTALS__")
pp = pprint.PrettyPrinter(indent=4)
print(pp.pprint(totals))
print("______________")
lowest_overall_error = lowest_error(totals)
print("The Winner!", lowest_overall_error)

testing_error = error_on_test_data(validation_x_normalized, validation_y_normalized, 
                   validation_x_scalar, validation_y_scalar, lowest_overall_error)

print("Error on test data:", testing_error)