import numpy
import os

def seperate_data_to_x_y_arrays(input_data_file_name):
    # input_data_file = open(input_data_file_name)
    data_as_mulit_array = numpy.loadtxt(input_data_file_name,dtype="i,f", delimiter=" ")
    print(data_as_mulit_array)
    print("--------------------------")
    split_arrays = numpy.hsplit(data_as_mulit_array,1)
    print(split_arrays)
    return "", ""

x_data, y_data = seperate_data_to_x_y_arrays("data/deficit_train.dat")
print("the end")