import numpy

def normalize_array(array):
    normalized_array = array/array.max()
    return normalized_array


array = numpy.array(([70, 50, 45], [40, 30, 10], [80, 40, 30]), numpy.float32)
normalized_array = normalize_array(array)