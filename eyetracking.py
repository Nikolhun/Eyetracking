import cv2
import ctypes
import numpy as np
from ctypes import wintypes
import math


def uv_from_vector(vector):
    u = vector[2]
    v = vector[3]
    uv = math.sqrt(u * u + v * v)
    return uv


def find_closest_in_array(array, value, closest_from_every_row=None):
    '''
    Finds closest number and its coordinates in array
    :param array: numpy array with at least two rows
    :param value: value, that is rearched
    :return: [number, row, column]
    '''
    ro = -1  # not 0 because len array returns from 1 not from 0
    closest_from_every_row = []
    closest_from_every_row_row = []
    closest_from_every_row_column = []
    for i in range(0, len(array)):
        ro = ro + 1
        array_line = np.asarray(array[i])
        idx = (np.abs(array_line - value)).argmin()  # get position
        closest_from_every_row.append(array_line[idx])  # closest numbers from row
        closest_from_every_row_row.append(ro)  # coordinate row
        closest_from_every_row_column.append(idx)  # coordinate column

    closest_from_every_row = np.asarray(closest_from_every_row)
    idx_from_every_row = (np.abs(closest_from_every_row - value)).argmin()  # get position prom closest numbers in row
    return closest_from_every_row[idx_from_every_row], closest_from_every_row_row[idx_from_every_row], \
           closest_from_every_row_column[idx_from_every_row]

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


def show_eyetracking(coordinate_x, coordinate_y, window_name, screensize, vector_end_coordinates):
    '''
    Visualize eyetracking
    :param coordinate_x: coordinate x
    :param coordinate_y: coordinate y
    :param window_name: window name in string
    :param screensize: screensize
    :return:
    '''
    mask = np.zeros((screensize[1], screensize[0]), np.uint8) + 255  # mask with size of screen and value 255
    start_point = (int(screensize[0]/2), int(screensize[1]/2))  # nebo opacne?
    end_point = (vector_end_coordinates[0]*10 + start_point[0], vector_end_coordinates[1]*10 + start_point[1])
    circle_size = 4
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # set window to full screen
    cv2.circle(mask, (coordinate_x, coordinate_y), circle_size, (0, 0, 255), -1)  # lower left
    cv2.arrowedLine(mask, start_point, end_point, color=(0, 255, 0), thickness=1)
    cv2.imshow(window_name, mask)
    cv2.waitKey(1)
