import cv2
import numpy as np
import time
import random
import threading

def normalize_array(array, value):
    '''
    Normalize array to have values 0-1.
    :param array: array you want to normalize
    :param value: value you want to normalize
    :return: normalized_array, normlaized_value
    '''
    normlaized_value = value / array.max()
    normalized_array = array / array.max()
    return normalized_array, normlaized_value


def difference_value(u_interpolated_array, v_interpolated_array):
    '''
    Get maximum difference value for function find_closest_in_array
    :param u_interpolated_array: u vector parameter (magnitude)
    :param v_interpolated_array: v vector parameter (direction)
    :return: max_difference_u, max_difference_v
    '''
    # max_difference_u = (u_interpolated_array.max() - u_interpolated_array.min())/30
    # max_difference_v = (v_interpolated_array.max() - v_interpolated_array.min())/20
    max_difference_u = u_interpolated_array.max()
    max_difference_v = v_interpolated_array.max()
    print("difference", max_difference_u, max_difference_v)
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

    # img[np.where(img == (0.4, 0.4, 0.4))] = (0.54, 0.27, 0.27)

    result_numbers = 0
    result_x = 0
    result_y = 0
    result_diff = 0

    if u_interpolated_array.shape == v_interpolated_array.shape:
        u_best_numbers = []
        u_best_number_x = []
        u_best_number_y = []
        u_best_number_diff = []

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
            print("Difference is to small to find some position. Change the difference.")
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
    return result_numbers, result_x, result_y, result_diff


#def make_bgr_mask(bf, gf, rf, size):
#    mask_for_eyetracking = np.zeros((size[0], size[1]), np.uint8)
#    mask_for_eyetracking = cv2.cvtColor(mask_for_eyetracking, cv2.COLOR_GRAY2BGR)
#    b, g, r = cv2.split(mask_for_eyetracking)
#    b[:, :] = bf
#    g[:, :] = gf
#    r[:, :] = rf
#    mask_for_eyetracking_bgr = cv2.merge([b, g, r])
#    return mask_for_eyetracking_bgr


def show_eyetracking(coordinate_x, coordinate_y, window_name, vector_end_coordinates,
                     interpolation_size, mask_bgr, coordinates_of_center, hit_target, hit_target_value):
    """
    Visualize eyetracking
    :param coordinate_x: coordinate x
    :param coordinate_y: coordinate y
    :param window_name: window name in string
    :param vector_end_coordinates: x and y coordinate from vector function
    :param interpolation_size: interpolation size (x,y)
    :param mask_bgr: output in rgb
    :return:
    """

    # get start points (point where is pupil when looking into the middle of the screen)
    start_point = (int(interpolation_size[0] / 2), int(interpolation_size[1] / 2))
    # get end point (point where the middle of pupil is * 10)
    end_point = (int(vector_end_coordinates[0] * 10 + start_point[0]),
                 int(vector_end_coordinates[1] * 10 + start_point[1]))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # make new window with window name
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # set window to full screen

    if mask_bgr[np.abs(coordinate_x - (interpolation_size[1] - 1))][coordinate_y][0] == 0 and \
            mask_bgr[np.abs(coordinate_x - (interpolation_size[1] - 1))][coordinate_y][1] == 0 and \
            mask_bgr[np.abs(coordinate_x - (interpolation_size[1] - 1))][coordinate_y][2] >= 153:

        mask_bgr[np.abs(coordinate_x - (interpolation_size[1] - 1))][coordinate_y][2] = \
            mask_bgr[np.abs(coordinate_x - (interpolation_size[1] - 1))][coordinate_y][2] - 2

    elif mask_bgr[np.abs(coordinate_x - (interpolation_size[1] - 1))][coordinate_y][0] == 0 and \
            mask_bgr[np.abs(coordinate_x - (interpolation_size[1] - 1))][coordinate_y][1] == 0 and \
            mask_bgr[np.abs(coordinate_x - (interpolation_size[1] - 1))][coordinate_y][2] << 150:

        print("menší, x, y, hodnota", np.abs(coordinate_x - (interpolation_size[1] - 1)), coordinate_y,
              mask_bgr[np.abs(coordinate_x - (interpolation_size[1] - 1))][coordinate_y][2])

    elif coordinates_of_center[1] == np.abs(coordinate_x - (interpolation_size[1] - 1)) and \
            coordinates_of_center[0] == coordinate_y:
        hit_target = True
        hit_target_value.append(hit_target)

    else:
        mask_bgr[np.abs(coordinate_x - (interpolation_size[1] - 1))][coordinate_y][0] = 0  # red color (0, 0, 255)
        mask_bgr[np.abs(coordinate_x - (interpolation_size[1] - 1))][coordinate_y][1] = 0

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


def change_coordinates_of_target(size_of_output_screen):
    num_rows = random.randint(0, size_of_output_screen[0]-1)
    num_coll = random.randint(0, size_of_output_screen[1]-1)
    coordinates_of_center_dot = (num_rows, num_coll)
    return coordinates_of_center_dot


def empty_mask_for_eyetracking(size_of_output_screen):
    mask_for_eyetracking = np.zeros((size_of_output_screen[1], size_of_output_screen[0]), np.uint8) + 255
    mask_for_eyetracking_bgr = cv2.cvtColor(mask_for_eyetracking, cv2.COLOR_GRAY2BGR)
    return mask_for_eyetracking_bgr





def add_red_pixels(mask_for_eyetracking_bgr, x_of_red_pixel_before, y_of_red_pixel_before, value_red):
    for i in range(0, len(x_of_red_pixel_before)):
        mask_for_eyetracking_bgr[x_of_red_pixel_before[i]][y_of_red_pixel_before[i]][2] = value_red[i]
    return mask_for_eyetracking_bgr


def save_red_pixels(mask_for_eyetracking_bgr, coordinates_of_center_dot, size_of_output_screen, step):
    x1_of_red_pixel_before = []
    y1_of_red_pixel_before = []
    value_red_1 = []
    x2_of_red_pixel_before = []
    y2_of_red_pixel_before = []
    value_red_2 = []

    for i in range(0, (int((3 * size_of_output_screen[1])/100) * 2) - 1):
        b_pixel = mask_for_eyetracking_bgr[coordinates_of_center_dot[1] - step + i][coordinates_of_center_dot[0]][0]
        g_pixel = mask_for_eyetracking_bgr[coordinates_of_center_dot[1] - step + i][coordinates_of_center_dot[0]][1]
        r_pixel = mask_for_eyetracking_bgr[coordinates_of_center_dot[1] - step + i][coordinates_of_center_dot[0]][2]
        if b_pixel == 0 and g_pixel == 0:
            x1_of_red_pixel_before.append(coordinates_of_center_dot[1] - step + i)
            y1_of_red_pixel_before.append(coordinates_of_center_dot[0])
            value_red_1.append(r_pixel)

    for y in range(0, (int((3 * size_of_output_screen[1])/100) * 2) - 1):
        b_pixel = mask_for_eyetracking_bgr[coordinates_of_center_dot[1]][coordinates_of_center_dot[0] - step + y][0]
        g_pixel = mask_for_eyetracking_bgr[coordinates_of_center_dot[1]][coordinates_of_center_dot[0] - step + y][1]
        r_pixel = mask_for_eyetracking_bgr[coordinates_of_center_dot[1]][coordinates_of_center_dot[0] - step + y][2]
        if b_pixel == 0 and g_pixel == 0:
            x2_of_red_pixel_before.append(coordinates_of_center_dot[1])
            y2_of_red_pixel_before.append(coordinates_of_center_dot[0] - step + y)
            value_red_2.append(r_pixel)

    return x1_of_red_pixel_before, y1_of_red_pixel_before, value_red_1, \
           x2_of_red_pixel_before, y2_of_red_pixel_before, value_red_2


def draw_line(mask_for_eyetracking_bgr, coordinates_of_center_dot, step, color):
    cv2.line(mask_for_eyetracking_bgr, (coordinates_of_center_dot[0] - step, coordinates_of_center_dot[1]),
             (coordinates_of_center_dot[0] + step, coordinates_of_center_dot[1]), color, 1)
    cv2.line(mask_for_eyetracking_bgr, (coordinates_of_center_dot[0], coordinates_of_center_dot[1] - step),
             (coordinates_of_center_dot[0], coordinates_of_center_dot[1] + step), color, 1)


def make_array_from_vectors(target_coordinate_x, target_coordinate_y, measured_vector_true_u, measured_vector_true_v):

    target_and_measured_vector_array = np.array([[target_coordinate_x], [target_coordinate_y],
                                                 [measured_vector_true_u], [measured_vector_true_v]])

    return target_and_measured_vector_array
