import cv2
import ctypes
import numpy as np
from ctypes import wintypes

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


def show_eyetracking(coordinate_x, coordinate_y, window_name, screensize):
    '''
    Visualize eyetracking
    :param coordinate_x: coordinate x
    :param coordinate_y: coordinate y
    :param window_name: window name in string
    :param screensize: screensize
    :return:
    '''
    mask = np.zeros((screensize[1], screensize[0]), np.uint8) + 255  # mask with size of screen and value 255
    circle_size = 4
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # set window to full screen
    cv2.circle(mask, (coordinate_x, coordinate_y), circle_size, (0, 0, 255), -1)  # lower left
    cv2.imshow(window_name, mask)
    cv2.waitKey(1)
