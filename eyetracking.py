import cv2
import numpy as np


def normalize_array(array, value):
    """
    Normalize array to have values 0-1.
    :param array: array you want to normalize
    :param value: value you want to normalize
    :return: normalized_array, normlaized_value
    """
    normlaized_value = value / array.max()
    normalized_array = array / array.max()
    return normalized_array, normlaized_value


#def difference_value(u_interpolated_array, v_interpolated_array):
  #  """
  #  Get maximum difference value for function find_closest_in_array
  #  :param u_interpolated_array: u vector parameter (magnitude)
 #   :param v_interpolated_array: v vector parameter (direction)
 #   :return: max_difference_u, max_difference_v
  #  """
  #  # max_difference_u = (u_interpolated_array.max() - u_interpolated_array.min())/30
  #  # max_difference_v = (v_interpolated_array.max() - v_interpolated_array.min())/20
 #   max_difference_u = u_interpolated_array.max()/2
 #   max_difference_v = v_interpolated_array.max()/2
#    print("difference", max_difference_u, max_difference_v)
 #   return max_difference_u, max_difference_v


def find_closest_in_array(u_interpolated_array, v_interpolated_array, value, max_difference_u, max_difference_v):
    """
    Finds the most accurate vector in u, v vector field.
    :param u_interpolated_array: interpolated u field/array of vectors
    :param v_interpolated_array: interpolated v field/array of vectors
    :param value: (x, y) value
    :param max_difference_u: from function difference_value
    :param max_difference_v: from function difference_value
    :return: result_numbers, result_x, result_y, result_diff, nothing found
    """

    result_numbers = 0
    result_x = 0
    result_y = 0
    result_diff = 0

    if u_interpolated_array.shape == v_interpolated_array.shape:
        u_best_numbers = []
        u_best_number_x = []
        u_best_number_y = []
        u_best_number_diff = []
        nothing_found = 0

        for i in range(0, u_interpolated_array.shape[0]):
            for y in range(0, u_interpolated_array.shape[1]):
                number = u_interpolated_array[i, y]
                diff = np.abs(number - value[0])
                if 0 <= diff < max_difference_u:  # last_diff_u
                    u_best_numbers.append(number)
                    u_best_number_x.append(y)
                    u_best_number_y.append(i)
                    u_best_number_diff.append(diff)

        v_best_numbers = []
        v_best_number_x = []
        v_best_number_y = []
        v_best_number_diff = []

        for i in range(0, v_interpolated_array.shape[0]):
            for y in range(0, v_interpolated_array.shape[1]):
                number = v_interpolated_array[i, y]
                diff = np.abs(number - value[1])
                if 0 <= diff < max_difference_v:
                    v_best_numbers.append(number)
                    v_best_number_x.append(y)
                    v_best_number_y.append(i)
                    v_best_number_diff.append(diff)

        if v_best_numbers == [] and v_best_number_x == [] and v_best_number_y == [] and v_best_number_diff == []:
            nothing_found = 1
        else:
            u2 = np.zeros(u_interpolated_array.shape, np.float32)
            v2 = np.zeros(v_interpolated_array.shape, np.float32)

            for i in range(0, len(u_best_number_x)):
                u2[u_best_number_y[i], u_best_number_x[i]] = u_best_numbers[i]
            for i in range(0, len(v_best_number_x)):
                v2[v_best_number_y[i], v_best_number_x[i]] = v_best_numbers[i]

            for i in range(0, u2.shape[0]):
                for y in range(0, u2.shape[1]):
                    if u2[i, y] == 0 and v2[i, y] != 0:
                        u2[i, y] = u_interpolated_array[i, y]
                    elif u2[i, y] != 0 and v2[i, y] == 0:
                        v2[i, y] = v_interpolated_array[i, y]

            last_diff_result = 5
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

    else:
        print("ERROR...u and v interpolated vectors should have the same size.")
    return result_numbers, result_x, result_y, result_diff, nothing_found


def show_eyetracking(coordinate_x, coordinate_y, interpolation_size, mask_bgr,
                     coordinates_of_center, hit_target, hit_target_value):
    """
    Visualize eyetracking
    :param coordinate_x: coordinate x
    :param coordinate_y: coordinate y
    :param window_name: window name in string
    :param interpolation_size: interpolation size (x,y)
    :param mask_bgr: output in rgb
    :return:
    """

    if mask_bgr[np.abs(coordinate_x - (interpolation_size[1] - 1))][coordinate_y][0] >= 5 and \
            mask_bgr[np.abs(coordinate_x - (interpolation_size[1] - 1))][coordinate_y][1] >= 5 and \
            mask_bgr[np.abs(coordinate_x - (interpolation_size[1] - 1))][coordinate_y][2] == 255:

        mask_bgr[np.abs(coordinate_x - (interpolation_size[1] - 1))][coordinate_y][0] =\
            mask_bgr[np.abs(coordinate_x - (interpolation_size[1] - 1))][coordinate_y][0] - 5
        mask_bgr[np.abs(coordinate_x - (interpolation_size[1] - 1))][coordinate_y][1] =\
            mask_bgr[np.abs(coordinate_x - (interpolation_size[1] - 1))][coordinate_y][1] - 5

    elif mask_bgr[np.abs(coordinate_x - (interpolation_size[1] - 1))][coordinate_y][0] <= 5 and \
            mask_bgr[np.abs(coordinate_x - (interpolation_size[1] - 1))][coordinate_y][1] <= 5 and \
            mask_bgr[np.abs(coordinate_x - (interpolation_size[1] - 1))][coordinate_y][2] >= 120:

        mask_bgr[np.abs(coordinate_x - (interpolation_size[1] - 1))][coordinate_y][2] = \
            mask_bgr[np.abs(coordinate_x - (interpolation_size[1] - 1))][coordinate_y][2] - 5

    elif mask_bgr[np.abs(coordinate_x - (interpolation_size[1] - 1))][coordinate_y][0] == 0 and \
            mask_bgr[np.abs(coordinate_x - (interpolation_size[1] - 1))][coordinate_y][1] == 0 and \
            mask_bgr[np.abs(coordinate_x - (interpolation_size[1] - 1))][coordinate_y][2] <= 115:

        print("Value of red is lower than 150.")

    elif coordinates_of_center[1] == np.abs(coordinate_x - (interpolation_size[1] - 1)) and \
            coordinates_of_center[0] == coordinate_y:
        hit_target = True
        hit_target_value.append(hit_target)

    #else:
     #   mask_bgr[np.abs(coordinate_x - (interpolation_size[1] - 1))][coordinate_y][0] = 0  # red color (0, 0, 255)
      #  mask_bgr[np.abs(coordinate_x - (interpolation_size[1] - 1))][coordinate_y][1] = 0

    coordinate_x = np.abs(coordinate_x - (interpolation_size[1] - 1))  # coordinate x

    return coordinate_x, coordinate_y, mask_bgr, hit_target, hit_target_value


def dimension(img_before, scale_percent):
    '''
    Get dimension to reshape picture
    :param img_before: image in bgr
    :param scale_percent: percenteage
    :return:
    '''
    mask_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
    width = int(mask_gray.shape[1] * scale_percent / 100)
    height = int(mask_gray.shape[0] * scale_percent / 100)
    reshape_dimension = (width, height)
    return reshape_dimension


def empty_mask_for_eyetracking(size_of_output_screen):
    """
    Makes empty mask for eyetracking.
    :param size_of_output_screen: Screen size for example (16, 9)
    :return: empty array for saving eyetraking
    """
    mask_for_eyetracking = np.zeros((size_of_output_screen[1], size_of_output_screen[0]), np.uint8) + 255
    mask_for_eyetracking_bgr = cv2.cvtColor(mask_for_eyetracking, cv2.COLOR_GRAY2BGR)
    return mask_for_eyetracking_bgr


def make_array_from_vectors(target_coordinate_x, target_coordinate_y, measured_vector_true_u_normalized,
                            measured_vector_true_v_normalized, measured_vector_true_u, measured_vector_true_v):
    """
    Make array from measured and found vectors to save results easily
    :param target_coordinate_x: coordinate x of target
    :param target_coordinate_y: coordinate y of target
    :param measured_vector_true_u_normalized: normalized value of measured vector u
    :param measured_vector_true_v_normalized: normalized value of measured vector v
    :param measured_vector_true_u: value of measured vector u
    :param measured_vector_true_v: value of measured vector v
    :return: array of results containing all numbers above
    """

    target_and_measured_vector_array = np.array([[target_coordinate_x], [target_coordinate_y],
                                                 [measured_vector_true_u_normalized],
                                                 [measured_vector_true_v_normalized],
                                                 [measured_vector_true_u], [measured_vector_true_v]])

    return target_and_measured_vector_array

