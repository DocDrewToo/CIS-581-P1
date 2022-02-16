import numpy
from scipy.linalg import lstsq
from sklearn.preprocessing import PolynomialFeatures

def fit(x, y, solver, _degree):
    # X : array-like of shape (n_samples, n_features)
    # y : array-like of shape (n_samples, 1)
    if solver == 'lstsq':
        x_reshaped = x_data.reshape(-1,1)
        polynomial_features = PolynomialFeatures(degree=_degree)
        X = polynomial_features.fit_transform(x_reshaped)
        X = np.c_[np.ones((X.shape[0], 1)), X]



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
fit(x_data, y_data, "lstsq", degree=2)