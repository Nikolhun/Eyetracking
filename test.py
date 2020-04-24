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

unhide_taskbar()
def show_eyetracking(coordinate_x, coordinate_y):
    user32 = ctypes.windll.user32
    screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    mask = np.zeros((screensize[1], screensize[0]), np.uint8) + 255
    circle_size = 4
    cv2.namedWindow("calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.circle(mask, (coordinate_y, coordinate_x), circle_size, (0, 0, 255), -1)  # lower left
    cv2.imshow("calibration", mask)
    hide_taskbar()
    cv2.waitKey(10000)
    unhide_taskbar()
    cv2.destroyWindow("calibration")
