import cv2
import ctypes
import numpy as np
from ctypes import wintypes

def hide_taskbar():
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
    mask = np.zeros((screensize[1], screensize[0]), np.uint8) + 255
    circle_size = 4
    #cv2.startWindowThread(window_name)
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.circle(mask, (coordinate_x, coordinate_y), circle_size, (0, 0, 255), -1)  # lower left
    cv2.imshow(window_name, mask)
    cv2.waitKey(100)
