import numpy as np

def difference_value(u_interpolated_array, v_interpolated_array):
    max_difference_u = int((u_interpolated_array.max() - u_interpolated_array.min())/2)
    max_difference_v = int((v_interpolated_array.max() - v_interpolated_array.min())/2)
    return max_difference_u, max_difference_v


def find_closest_in_array(u_interpolated_array, v_interpolated_array, value, max_difference_u, max_difference_v):
    '''
    Finds the most accurate vector in u, v vector field.
    :param u_interpolated_array: interpolated u field/array of vectors
    :param v_interpolated_array: interpolated v field/array of vectors
    :param value: (x, y) value
    :param max_difference_u: from function difference_value
    :param max_difference_v: from function difference_value
    :return: result_numbers, result_x, result_y, result_diff
    '''

    result_numbers = 0
    result_x = 0
    result_y = 0
    result_diff = 0

    if u_interpolated_array.shape == v_interpolated_array.shape:
        last_diff_u = 10
        u_best_numbers = []
        u_best_number_x = []
        u_best_number_y = []
        u_best_number_diff = []

        for i in range(0, u_interpolated_array.shape[0]):
            for y in range(0, u_interpolated_array.shape[1]):
                number = u_interpolated_array[i, y]
                diff = number - value[0]
                if 0 <= diff < last_diff_u:
                    last_diff_u = diff
                    u_best_numbers.append(number)
                    u_best_number_x.append(i)
                    u_best_number_y.append(y)
                    u_best_number_diff.append(diff)

        last_diff_v = 10
        v_best_numbers = []
        v_best_number_x = []
        v_best_number_y = []
        v_best_number_diff = []

        for i in range(0, v_interpolated_array.shape[0]):
            for y in range(0, v_interpolated_array.shape[1]):
                diff = v_interpolated_array[i, y] - value[1]
                if 0 <= diff < last_diff_v:
                    last_diff_v = diff
                    v_best_numbers.append(v_interpolated_array[i, y] )
                    v_best_number_x.append(i)
                    v_best_number_y.append(y)
                    v_best_number_diff.append(diff)

        u2 = np.zeros(u_interpolated_array.shape, np.uint8)
        v2 = np.zeros(v_interpolated_array.shape, np.uint8)

        for i in range(0, len(u_best_numbers)):
            u2[u_best_number_x[i], u_best_number_y[i]] = u_best_numbers[i]
        for i in range(0, len(v_best_numbers)):
            v2[v_best_number_x[i], v_best_number_y[i]] = v_best_numbers[i]

        for i in range(0, u2.shape[0]):
            for y in range(0, u2.shape[1]):
                if u2[i, y] == 0 and v2[i, y] != 0:
                    u2[i, y] = u_interpolated_array[i, y]
                elif u2[i, y] != 0 and v2[i, y] == 0:
                    v2[i, y] = v_interpolated_array[i, y]

        last_diff_result = 100
        for i in range(0, u2.shape[0]):
            for y in range(0, u2.shape[1]):
                if u2[i, y] != 0 and v2[i, y] != 0:
                    diff = np.abs(np.abs(u2[i, y] - value[0]) + np.abs(v2[i, y] - value[1]))
                    if 0 <= diff < last_diff_result:
                        last_diff_result = diff
                        result_numbers = (u2[i, y], v2[i, y])
                        result_x = i
                        result_y = y
                        result_diff = last_diff_result
    return result_numbers, result_x, result_y, result_diff


value = (5, 4)
u_interpolated_array = np.array([(10, 5, 0), (2, 3, 0), (0, 0, 0)], dtype=np.uint8)
v_interpolated_array = np.array([(6, 8, 1), (4, 1, 1), (0, 0, 0)], dtype=np.uint8)

#u_interpolated_array = np.array([(10, 5, 0), (2, 3, 0)], dtype=np.uint8)
#v_interpolated_array = np.array([(6, 8, 1), (4, 1, 1)], dtype=np.uint8)

max_difference_u, max_difference_v = difference_value(u_interpolated_array, v_interpolated_array)
result_numbers, result_x, result_y, result_diff = find_closest_in_array(u_interpolated_array, v_interpolated_array, value, max_difference_u, max_difference_v)
print("result_numbers", result_numbers)
print("result_x", result_x)
print("result_y", result_y)
print("result_diff", result_diff)