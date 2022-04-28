from sklearn.preprocessing import StandardScaler
import numpy

class the_data():
    def __init__(self, data, folds=0):
        self.data = data
        self.folds = folds
        self.normalize()

    def __call__(self):
        return True


    def normalize(self):
        try:
            normalized_data = self.normalized_data
        except AttributeError:
            scaler = StandardScaler()
            scaler.fit(self.data.reshape(-1, 1))
            transform = scaler.transform(self.data.reshape(-1, 1))

            output_array = []
            for value in transform:
                output_array.append(value[0])
            normalized_data = numpy.array(output_array)

            self.normalized_data = normalized_data
            self.scaler = scaler
            # average = numpy.average(normalized_data)
            # standard_deviation = numpy.std(normalized_data)
        return normalized_data

    def un_normalize(self):
        self.normalize()
        try:
            un_normalized_data = self.un_normalized_data
        except AttributeError:
            inverse_transform = self.scaler.inverse_transform(self.data.reshape(-1, 1))
            output_array = []
            for value in inverse_transform:
                output_array.append(value[0])
            un_normalized_data = numpy.array(output_array)

        return un_normalized_data

    def scatter_plot(self):
        try:
            scatter_plot_x = self.scatter_plot_x
        except AttributeError:
            min_x = numpy.amin(self.data)
            max_x = numpy.amax(self.data)
            num_of_xs = self.data.size * 4
            scatter_plot_x = numpy.linspace(min_x, max_x, num=num_of_xs)
            self.scatter_plot_x = scatter_plot_x
       
        return scatter_plot_x


    def scatter_plot_normalized(self):
        try:
            scatter_plot_x = self.scatter_plot_x
        except AttributeError:
            self.scatter_plot()

        try:
            plot_normalized_data = self.plot_normalized_data
        except AttributeError:
            self.scaler.fit(self.scatter_plot_x.reshape(-1, 1))
            transform = self.scaler.transform(self.scatter_plot_x.reshape(-1, 1))

            output_array = []
            for value in transform:
                output_array.append(value[0])
            plot_normalized_data = numpy.array(output_array)

            self.plot_normalized_data = plot_normalized_data

        # average = numpy.average(plot_normalized_data)
        # standard_deviation = numpy.std(plot_normalized_data)
        return plot_normalized_data

    def split_into_test_folds(self):
        return None