from sklearn.preprocessing import StandardScaler
import numpy

class the_data():
    def __init__(self, data):
        self.data = data

    def __call__(self):
        return None

    def normalized(self):
        try:
            normalized_data = self.normalized_data
        except AttributeError:
            scaler = StandardScaler()
            scaler.fit(self.data.reshape(-1, 1))
            normalized_data = scaler.transform(self.data.reshape(-1, 1))

            normal_array = []
            for value in normalized_data:
                normal_array.append(value[0])
            normalized_data = numpy.array(normal_array)

            self.normalized_data = normalized_data
            self.scaler = scaler
        # average = numpy.average(single_numpy_array)
        # standard_deviation = numpy.std(single_numpy_array)
        return normalized_data

    def un_normalize(self):
        return None

    def read_data(self):
        return None

    def split_into_test_folds(self):
        return None