import numpy
import os
from sklearn.metrics import mean_squared_error

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


def fit_numpy(x_data, y_data):
    optimal_values = numpy.polyfit(x_data, y_data, 2)

    optimal_intercept = round(optimal_values[0], 4)
    optimal_slope = round(optimal_values[1], 4)

    print("y = ",optimal_slope,"x +",optimal_intercept)
    return(optimal_values)


def fit_manual(x_data, y_data):
    #Pad the 1st column with 1's, 2nd column are just the x values
    X = numpy.c_[numpy.ones((x_data.shape[0], 1)), x_data]

    #Calculate optimal values for variables
    # X_transpose = X.transpose()
    # product_of_X_transpose_times_X = numpy.matmul(X_transpose, X)
    # inverse_matrix = numpy.linalg.inv(product_of_X_transpose_times_X)
    # product_of_inverse_times_X_transpose = numpy.matmul(inverse_matrix, X_transpose)
    # optimal_values = numpy.matmul(product_of_inverse_times_X_transpose, y_data)

    optimal_values = numpy.linalg.inv((X.T @ X)) @ ( X.T @ y_data)

    optimal_intercept = round(optimal_values[0], 4)
    optimal_slope = round(optimal_values[1], 4)

    print("y = ",optimal_slope,"x +",optimal_intercept)

    return(optimal_values)

def calculate_loss(loss_function, x, y, hypothesis_constants):
    slope_m = float(hypothesis_constants[1])
    intercept_b = float(round(hypothesis_constants[0]))

    sum_total = 0
    for index, x_value in enumerate(x):
        y_fit = slope_m * x_value + intercept_b
        part1 = y_fit - y[index]
        part2 = part1 * part1
        sum_total = sum_total + part2

    total_average_loss = sum_total / x.size
    return total_average_loss

def calculate_loss_scikit(loss_function, x, y, hypothesis_constants):
    slope_m = float(hypothesis_constants[1])
    intercept_b = float(round(hypothesis_constants[0]))

    predicted_y_array = [] 
    for index, x_value in enumerate(x):
        y_predicted = slope_m * x_value + intercept_b
        predicted_y_array.append(y_predicted)
    y_as_array = list(y)
    total_average_loss = mean_squared_error(y_as_array, predicted_y_array)
    return total_average_loss


data = "data/test.txt"
# data = "data/deficit_train.dat"
x_data, y_data = seperate_data_to_x_y_arrays(data)

optimal_values = fit_manual(x_data, y_data)
# optimal_values = fit_numpy(x_data,y_data)

loss = calculate_loss("MSE", x_data ,y_data, optimal_values)
print("Average loss (manual) using MSE",loss)
loss = calculate_loss_scikit("MSE", x_data ,y_data, optimal_values)
print("Average loss (scikit) using MSE",loss)