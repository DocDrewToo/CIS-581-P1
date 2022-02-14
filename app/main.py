import numpy
import os
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import pandas
import matplotlib.pyplot as pyplot
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

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


def fit_manual(x_data, y_data, _degree=1):

    #Pad the 1st column with 1's, the rest depending on degree
    # d=1 | 1 | x | 
    # d=2 | 1 | x | x^2 |
    x_reshaped = x_data.reshape(-1,1)
    print("x re-shaped:", x_reshaped.shape)

    polynomial_features = PolynomialFeatures(degree=_degree)
    X = polynomial_features.fit_transform(x_reshaped)

    #Calculate optimal values for variables
    optimal_values = numpy.linalg.inv((X.T @ X)) @ ( X.T @ y_data)
    print("Optimal Values for degree:",_degree, optimal_values)
    # optimal_intercept = round(optimal_values[0], 4)
    # optimal_slope = round(optimal_values[1], 4)

    # print("y = ",optimal_slope,"x +",optimal_intercept)

    return(optimal_values)

def matrix_x_with_degree_d(x, _degree=1):
    x_reshaped = x.reshape(-1,1)

    polynomial_features = PolynomialFeatures(degree=_degree)
    X = polynomial_features.fit_transform(x_reshaped)

    return X

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
x_test = []
y_test =[]
ridge_regression = Ridge(solver='lsqr')
params = {'alpha':[1e-25,1e-20,1e-14,1e-7,1e-3,1e3,1e7]}

# for degree in range(2,3):
#     pipe = Pipeline(steps=[
#         ('scaler', StandardScaler()),
#         ('preprocessor', PolynomialFeatures(degree=degree, include_bias=False)),
#         ('estimator', GridSearchCV(ridge_regression,params,scoring='neg_root_mean_squared_error',cv=2)),
#         ('scoring', "neg_root_mean_squared_error")
#     ])
#     pipe.fit(x_data, y_data)
#     y_pred_train_pipe = pipe.predict(x_data)
#     print(y_pred_train_pipe[:5])

Regressor = GridSearchCV(ridge_regression, params, scoring="neg_mean_squared_error", cv=2)
x_matrix_degree_d = matrix_x_with_degree_d(x_data, 2)
fit = Regressor.fit(x_matrix_degree_d,y_data)
predictions = Regressor.predict(x_data.reshape(-1, 1))
print(predictions)

    # print('best parameter: ', Regressor.best_params_)
    # print('best score: ', -Regressor.best_score_)

# optimal_values = fit_manual(x_data, y_data, 2)
# optimal_values = fit_numpy(x_data,y_data)


# loss = calculate_loss("MSE", x_data ,y_data, optimal_values)
# print("Average loss (manual) using MSE",loss)
# loss = calculate_loss_scikit("MSE", x_data ,y_data, optimal_values)
# print("Average loss (scikit) using MSE",loss)