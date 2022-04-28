from operator import index
import numpy
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plot
from sklearn.metrics import mean_squared_error
import math
from data_processing import the_data
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

# def data_for_hypothesis_curve(data):
#     min_x = numpy.amin(data)
#     max_x = numpy.amax(data)
#     num_of_xs = data.size
#     lots_of_xs = numpy.linspace(min_x, max_x)
#     return lots_of_xs

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


# def normalize(_data, type="int"):
#     scaler = StandardScaler()
#     scaler.fit(_data.reshape(-1, 1))
#     normalized_data = scaler.transform(_data.reshape(-1, 1))

#     normal_array = []
#     for value in normalized_data:
#         normal_array.append(value[0])
#     single_numpy_array = numpy.array(normal_array)

#     average = numpy.average(single_numpy_array)
#     standard_deviation = numpy.std(single_numpy_array)

#     return scaler, single_numpy_array

# def un_normalize(_data, scaler):
#     un_normalized_data = scaler.inverse_transform(_data.reshape(-1, 1))

#     normal_array = []
#     for value in un_normalized_data:
#         normal_array.append(value[0])
#     single_numpy_array = numpy.array(normal_array)

#     return single_numpy_array

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


def plot_all_itterations(_totals, x, y):
    plot.title("National Deficit - All Degrees")
    for _degree in range(13):
        lowest_error = lowest_error_per_degree(_totals, _degree)
        hypotheses = lowest_error[0][3]
        x_scatter_data = the_data(x.scatter_plot())
        x_scatter_data.scaler = x.scaler

        y_plot_curve_predicted = predict(x_scatter_data.normalize(), hypotheses, _degree)

        my_label = "D:" + str(_degree)
        plot.scatter(x.data, y.data, color='black')
        y_plot_curve_predicted_data = the_data(y_plot_curve_predicted)
        y_plot_curve_predicted_data.scaler = y.scaler
        plot.plot(x.scatter_plot(), y_plot_curve_predicted_data.un_normalize(), label=my_label)
    plot.xlabel("Year")
    plot.ylabel("Deficit (Billions)")
    plot.legend(loc="lower left")
    plot.savefig("Training_All_Degrees.png", dpi=480)
    plot.show
    plot.clf()

def plot_all_lambda_for_d_12(_totals, x, y):
    plot.title("National Deficit - D12 - All Lambdas")

    numpy_array_of_totals = numpy.array(_totals, dtype=object)
    degree_column = numpy_array_of_totals[:,0]
    indexes_of_degree = numpy.where(degree_column == 12)
    results_degree_12_matrix = numpy_array_of_totals[indexes_of_degree]

    for results in results_degree_12_matrix:
        degree = results[0]
        _lambda = results[1] 
        hypotheses = results[3]
        x_scatter_data = the_data(x.scatter_plot())
        x_scatter_data.scaler = x.scaler

        y_plot_curve_predicted = predict(x_scatter_data.normalize(), hypotheses, 12)

        my_label = "D: " + str(degree) + " Lambda: " + str(_lambda)
        plot.scatter(x.data, y.data, color='black')
        y_plot_curve_predicted_data = the_data(y_plot_curve_predicted)
        y_plot_curve_predicted_data.scaler = y.scaler
        plot.plot(x.scatter_plot(), y_plot_curve_predicted_data.un_normalize(), label=my_label)

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

x_training_data_full = the_data(training_x)
y_training_data_full = the_data(training_y)

x_validation_data = the_data(validation_x)
y_validation_data = the_data(validation_y)

folds = 6 # Split the training data into folds to use as mini testing data

totals = []
hold_outs_x , y_test = seperate_to_folds(x_training_data_full.data, y_training_data_full.data, folds)

for degree in range(13):
    my_lambda = [0]
    if degree == 12:
        my_lambda = [0, math.exp(-25), math.exp(-20), math.exp(-14),
                     math.exp(-7), math.exp(-3), 1, math.exp(3), math.exp(7)]
    for _lambda in my_lambda:
        average_error_list = []
        for fold_n, x_test  in enumerate(hold_outs_x):
            # Remove the hold_out[index] test data from the overall training data set

            indexes_to_del = indexes_of_data(x_test, fold_n * folds)
            x_training_fold = numpy.delete(x_training_data_full.data, indexes_to_del)
            y_training_fold = numpy.delete(y_training_data_full.data, indexes_to_del)
            
            x_training_fold_data = the_data(x_training_fold)
            y_training_fold_data = the_data(y_training_fold)

            hypothesis = fit(x_training_fold_data.normalize(), 
                             y_training_fold_data.normalize(), 
                             degree, 
                             _lambda)
            
            x_test_data = the_data(x_test)
            x_test_data.scaler = x_training_fold_data.scaler
            y_predicted = predict(x_test_data.normalize(), 
                                  hypothesis, 
                                  degree)

            # y_test_data = the_data(y_test[fold_n])
            y_predicted_data = the_data(y_predicted)
            y_predicted_data.scaler = y_training_fold_data.scaler
            error = root_mean_square_error(y_test[fold_n],
                                           y_predicted_data.un_normalize())
            average_error_list.append(error)
   
        average_error = sum(average_error_list) / len(average_error_list)
        hypothesis = fit(x_training_data_full.normalize(), 
                         y_training_data_full.normalize(), 
                         degree, 
                         _lambda)
        totals.append([degree, _lambda, average_error, hypothesis])


plot_all_itterations(totals, x_training_data_full, y_training_data_full)

# plot_all_lambda_for_d_12(totals, x_training_data_full, y_training_data_full)

print("__ALL TOTALS__")
pp = pprint.PrettyPrinter(indent=4)
print(pp.pprint(totals))
print("______________")
lowest_overall_error = lowest_error(totals)
print("The Winner!", lowest_overall_error)

best_degree = lowest_overall_error[0][0]
best_hypothesis = lowest_overall_error[0][3]

x_validation_data.scaler = x_training_data_full.scaler
y_predicted = predict(x_validation_data.normalize(), 
                      best_hypothesis, 
                      best_degree)

y_validation_data.scaler = y_training_data_full.scaler
validation_error = root_mean_square_error(y_validation_data.un_normalize(),
                              y_predicted)

print("Error on validation data:", validation_error)