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


def normalize(_data):
    scaler = StandardScaler()
    scaler.fit(_data.reshape(-1, 1))
    normalized_data = scaler.transform(_data.reshape(-1, 1))

    average = numpy.average(normalized_data)
    standard_deviation = numpy.std(normalized_data)

    return scaler, normalized_data

def un_normalize(_data, scalar):
    un_normalized_data = scalar.inverse_transform(_data)
    return un_normalized_data

training_dataset = "data/deficit_train.dat"
validation_dataset = "data/deficit_test.dat"

training_x , training_y = read_data_to_x_y_arrays(training_dataset)
validation_x , validation_y = read_data_to_x_y_arrays(validation_dataset)

normalized_training_x = normalize(training_x)

# is_this_an_array = numpy.asanyarray(normalized_training_x_y)


# Split the training data into folds to use as mini testing data
folds = 6
my_lambda = [0, math.exp(-25), math.exp(-20), math.exp(-14),
 math.exp(-7), math.exp(-3), 1, math.exp(3), math.exp(7)]

for degree in range(13):
    print("For degree:", degree)
    for _lambda in my_lambda:

        hold_outs_x , hold_outs_y = seperate_to_folds(normalized_training_x, training_y, folds)
        for fold_n, hold_out_test_x  in enumerate(hold_outs_x):
            # Remove the hold_out[index] test data from the overall training data set
            # training_data_x = numpy.setdiff1d(training_x, hold_out_test_x)

            indexes_to_del = []
            for items in range(hold_out_test_x.size):
                start_index = fold_n * folds
                indexes_to_del.append(items + start_index)
            
            training_data_x = numpy.delete(training_x, indexes_to_del)
            training_data_y = numpy.delete(training_y, indexes_to_del)
            training_x_features = x_data_in_polynomial_matrix(training_data_x, degree)
            weights = fit(training_x_features, training_data_y, _lambda)

            # cv_test_data_x , cv_test_data_y =
            hold_out_data_x_features =  x_data_in_polynomial_matrix(hold_out_test_x, degree)
            y_predicted = predict(hold_out_data_x_features, weights)
            error = root_mean_square_error(hold_outs_y[fold_n], y_predicted)
        
            print("For lambda:", _lambda, "Error (rmse)", error)
            plot_this(validation_x, validation_y, weights, degree)
