import cv2
import ctypes
import numpy as np
from ctypes import wintypes


def normalize_array(array, value):
    '''
    Normalize array to have values 0-1.
    :param array: array you want to normalize
    :param value: value you want to normalize
    :return: normalized_array, normlaized_value
    '''
    normlaized_value = value/array.max()
    normalized_array = array/array.max()
    return normalized_array, normlaized_value


def difference_value(u_interpolated_array, v_interpolated_array):
    '''
    Get maximum difference value for function find_closest_in_array
    :param u_interpolated_array: u vector parameter (magnitude)
    :param v_interpolated_array: v vector parameter (direction)
    :return: max_difference_u, max_difference_v
    '''
    max_difference_u = (u_interpolated_array.max() - u_interpolated_array.min())/50
    max_difference_v = (v_interpolated_array.max() - v_interpolated_array.min())/20
    #print("difference", max_difference_u, max_difference_v)
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
                diff = np.abs(number - value[0])
                if 0 <= diff < max_difference_u:  # last_diff_u
                    last_diff_u = diff
                    u_best_numbers.append(number)
                    u_best_number_x.append(y)
                    u_best_number_y.append(i)
                    u_best_number_diff.append(diff)
        #print("1")
        last_diff_v = 10
        v_best_numbers = []
        v_best_number_x = []
        v_best_number_y = []
        v_best_number_diff = []

        for i in range(0, v_interpolated_array.shape[0]):
            for y in range(0, v_interpolated_array.shape[1]):
                number = v_interpolated_array[i, y]
                diff = np.abs(number - value[1])
                if 0 <= diff < max_difference_v:
                    last_diff_v = diff
                    v_best_numbers.append(number)
                    v_best_number_x.append(y)
                    v_best_number_y.append(i)
                    v_best_number_diff.append(diff)
        #print("2")

        if v_best_numbers == [] and v_best_number_x == [] and v_best_number_y == [] and v_best_number_diff == []:
            print("Difference is to small to find some position. Change the difference.")
        else:
            u2 = np.zeros(u_interpolated_array.shape, np.float32)
            v2 = np.zeros(v_interpolated_array.shape, np.float32)

            #print("v_best_number_x", len(v_best_number_x))

            for i in range(0, len(u_best_number_x)):
                u2[u_best_number_y[i], u_best_number_x[i]] = u_best_numbers[i]
            for i in range(0, len(v_best_number_x)):
                v2[v_best_number_y[i], v_best_number_x[i]] = v_best_numbers[i]

            #print("3")

            for i in range(0, u2.shape[0]):
                for y in range(0, u2.shape[1]):
                    if u2[i, y] == 0 and v2[i, y] != 0:
                        u2[i, y] = u_interpolated_array[i, y]
                    elif u2[i, y] != 0 and v2[i, y] == 0:
                        v2[i, y] = v_interpolated_array[i, y]

            #print("4")

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

            #print("5")

    else:
        print("ERROR...u and v interpolated vectors should have the same size.")
    return result_numbers, result_x, result_y, result_diff


def hide_taskbar():
    '''
    Function for hiding taskbar.
    :return: Hides screen taskbar
    '''
    user32 = ctypes.WinDLL("user32")
    SW_HIDE = 0
    user32.FindWindowW.restype = wintypes.HWND
    user32.FindWindowW.argtypes = (
        wintypes.LPCWSTR,  # lpClassName
        wintypes.LPCWSTR)  # lpWindowName
    user32.ShowWindow.argtypes = (
        wintypes.HWND,  # hWnd
        ctypes.c_int)  # nCmdShow
    hWnd = user32.FindWindowW(u"Shell_traywnd", None)
    user32.ShowWindow(hWnd, SW_HIDE)


def unhide_taskbar():
    '''
    Function for showing taskbar.
    :return: Show screen taskbar
    '''
    user32 = ctypes.WinDLL("user32")
    SW_SHOW = 5
    user32.FindWindowW.restype = wintypes.HWND
    user32.FindWindowW.argtypes = (
        wintypes.LPCWSTR,  # lpClassName
        wintypes.LPCWSTR)  # lpWindowName
    user32.ShowWindow.argtypes = (
        wintypes.HWND,  # hWnd
        ctypes.c_int)  # nCmdShow
    hWnd = user32.FindWindowW(u"Shell_traywnd", None)
    user32.ShowWindow(hWnd, SW_SHOW)


def show_eyetracking(coordinate_x, coordinate_y, window_name, screensize, vector_end_coordinates, interpolation_size):
    '''
    Visualize eyetracking
    :param coordinate_x: coordinate x
    :param coordinate_y: coordinate y
    :param window_name: window name in string
    :param screensize: screensize
    :param vector_end_coordinates: x and y coordinate from vector function
    :return:
    '''
   # mask = np.zeros((screensize[1], screensize[0]), np.uint8) + 255  # mask with size of screen and value 255
    #start_point = (int(screensize[0]/2), int(screensize[1]/2))  # nebo opacne?
    #end_point = (int(vector_end_coordinates[0]*10 + start_point[0]), int(vector_end_coordinates[1]*10 + start_point[1]))
   # circle_size = 4

    mask = np.zeros((interpolation_size[1], interpolation_size[0]), np.uint8) + 255  # mask with size of screen and value 255
    start_point = (int(interpolation_size[0]/2), int(interpolation_size[1]/2))  # nebo opacne?
    end_point = (int(vector_end_coordinates[0] * 10 + start_point[0]), int(vector_end_coordinates[1] * 10 + start_point[1]))
    circle_size = 1

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # set window to full screen
    cv2.circle(mask, (coordinate_x, coordinate_y), circle_size, (0, 0, 255), -1)  # lower left
    cv2.arrowedLine(mask, start_point, end_point, color=(0, 255, 0), thickness=1)
    cv2.imshow(window_name, mask)

unhide_taskbar()