import cv2
import numpy as np
import ctypes
import keyboard
from ctypes import wintypes


#######################################################################################################################
# ---------------------------------- Functions ---------------------------------------------- #
#######################################################################################################################
def upper_left(vector):
    #cv2.circle(mask, (circle_size, circle_size), circle_size, (0, 0, 255), -1)  # uper left
    print('Look into upper left corner and press ENTER.')
    upper_left_state = False
    upper_left_corner = [0, 0, 0]
    while not upper_left_state:
        if keyboard.is_pressed("1"):
            upper_left_corner = vector
            upper_left_state = True
            # cv2.circle(mask, (circle_size, circle_size), circle_size, (255, 255, 255), -1)  # hide circle
    return upper_left_corner


def upper_right(vector):
    #cv2.circle(mask, (screensize[1] - circle_size, circle_size), circle_size, (0, 0, 255), -1)  # uper right
    print('Look into upper right corner and press ENTER.')
    upper_right_state = False
    upper_right_corner = [0, 0, 0]
    while not upper_right_state:
        if keyboard.is_pressed("2"):
            upper_right_corner = vector
            upper_right_state = True
            # cv2.circle(mask, (screensize[1] - circle_size, circle_size), circle_size, (255, 255, 255), -1)  # hide circle
    return upper_right_corner


def lower_right(vector):
    #cv2.circle(mask, (screensize[1] - circle_size, screensize[0] - circle_size), circle_size, (0, 0, 255), -1)  # lower right
    print('Look into lower right corner and press ENTER.')
    lower_right_state = False
    lower_right_corner = [0, 0, 0]
    while not lower_right_state:
        if keyboard.is_pressed("3"):
            lower_right_corner = vector
            lower_right_state = True
            # cv2.circle(mask, (screensize[1] - circle_size, screensize[0] - circle_size), circle_size, (255, 255, 255), -1)  # hide circle
    return lower_right_corner


def lower_left(vector):
    #cv2.circle(mask, (circle_size, screensize[0] - circle_size), circle_size, (0, 0, 255), -1)  # lower left
    print('Look into lower left corner and press ENTER.')
    lower_left_state = False
    lower_left_corner = [0, 0, 0]
    while not lower_left_state:
        if keyboard.is_pressed("4"):
            lower_left_corner = vector
            lower_left_state = True
            #cv2.circle(mask, (circle_size, screensize[0] - circle_size), circle_size, (255, 255, 255), -1)  # hide circle
    return lower_left_corner

#######################################################################################################################
# ---------------------------------- Get screen size and hide task bar ---------------------------------------------- #
#######################################################################################################################
#user32 = ctypes.WinDLL("user32")

#SW_HIDE = 0
#SW_SHOW = 5

#user32.FindWindowW.restype = ctypes.wintypes.HWND
#user32.FindWindowW.argtypes = (
   # wintypes.LPCWSTR,  # lpClassName
  #  wintypes.LPCWSTR)  # lpWindowName

#user32.ShowWindow.argtypes = (
   # wintypes.HWND,  # hWnd
  #  ctypes.c_int)  # nCmdShow

#def hide_taskbar():
 #   hWnd = user32.FindWindowW(u"Shell_traywnd", None)
 #   user32.ShowWindow(hWnd, SW_HIDE)

#def unhide_taskbar():
 #   hWnd = user32.FindWindowW(u"Shell_traywnd", None)
#    user32.ShowWindow(hWnd, SW_SHOW)

# Get screen size
#user32 = ctypes.windll.user32
#screensize = (user32.GetSystemMetrics(79)-50, user32.GetSystemMetrics(78))

#cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)  # Dlib landmark left eye
#cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # set window to full screen

# making mask for calibrating fantom
#mask = 255 - np.zeros((screensize[0], screensize[1]), np.uint8)
#mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

#upper_left()
#upper_right()
#lower_right()
#lower_left()

#hide_taskbar()
#cv2.imshow("Calibration", mask)
#mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

#print("print enter to continue")
#input()
#cv2.waitKey(10000)
#unhide_taskbar()
#cv2.destroyAllWindows()


