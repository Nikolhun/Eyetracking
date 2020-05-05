import numpy as np
import cv2


def prepare_mask_for_calibration(screensize, press, output_vector_in_eye_frame):
    '''
    Shows calibration points and prints text to navigate.
    :param screensize: screensize in (rows, collumns)
    :param press: number that is going to be pressed
    :param output_vector_in_eye_frame: vector rom find vector
    :return: shows window
    '''
    mask = np.zeros((screensize[1], screensize[0]), np.uint8) + 255  # mask with size of screen and value 255
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.namedWindow('calibration', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # set window to full screen
    circle_size = 40

    if press == 1:
        coordinates = (circle_size, screensize[1] - circle_size)
        text0 = ""
        text1 = "Look into lower left and press 1"
    elif press == 2:
        coordinates = (circle_size, int(screensize[1]/2))
        text0 = "Lower left saved"
        text1 = "Look into middle left and press 2"
    elif press == 3:
        coordinates = (circle_size, circle_size)
        text0 = "Middle left saved"
        text1 = "Look into upper left and press 3"
    elif press == 4:
        coordinates = (int(screensize[0]/2), screensize[1] - circle_size)
        text0 = "Upper left saved"
        text1 = "Look into middle bottom and press 4"
    elif press == 5:
        coordinates = (int(screensize[0]/2), int(screensize[1]/2))
        text0 = "Middle bottom saved"
        text1 = "Look into middle and press 5"
    elif press == 6:
        coordinates = (int(screensize[0]/2), circle_size)
        text0 = "Middle saved"
        text1 = "Look into middle top and press 6"
    elif press == 7:
        coordinates = (screensize[0] - circle_size, screensize[1] - circle_size)
        text0 = "Middle top saved"
        text1 = "Look into lower right and press 7"
    elif press == 8:
        coordinates = (screensize[0] - circle_size, int(screensize[1]/2))
        text0 = "Lower right saved"
        text1 = "Look into middle right and press 8"
    elif press == 9:
        coordinates = (screensize[0] - circle_size, circle_size)
        text0 = "Middle right saved"
        text1 = "Look into upper right and press 9"
    elif press == 0:
        coordinates = []
        text0 = "Upper right saved"
        text1 = "Pres enter to save measured data"
    else:
        coordinates = []
        text0 = "Calibration works only with numbers 1-9"
        text1 = ""

    if coordinates:
        cv2.circle(mask, coordinates, 10, (0, 0, 255), -1)

    # get size of the text to write it in the middle
    text0_size, _ = cv2.getTextSize(text0, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
    text1_size, _ = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)

    start_point0 = (int(screensize[0] / 2) - int(text0_size[0]/2),
                    int(screensize[1]/2) - int(text0_size[1]/2) - int(screensize[1] / 100))
    start_point1 = (int(screensize[0] / 2) - int(text1_size[0]/2),
                    int(screensize[1]/2) + int(text1_size[1]/2) + int(screensize[1] / 100))

    cv2.putText(mask, text0, start_point0, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(mask, text1, start_point1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow('calibration', mask)


def lower_left(vector):
    '''
    Save coordinates x, y, u, v from lower left corner.
    :param vector: x, y, u, v from vector
    :return: coordinates x, y, u, v
    '''
    lower_left_state = False
    lower_left_corner = [0, 0, 0, 0]
    while not lower_left_state:
        lower_left_corner = vector
        lower_left_state = True
    return lower_left_corner


def middle_left(vector):
    '''
    Save coordinates x, y, u, v from middle left.
    :param vector: x, y, u, v from vector
    :return: coordinates x, y, u, v
    '''
    middle_left_state = False
    middle_left_corner = [0, 0, 0, 0]
    while not middle_left_state:
        middle_left_corner = vector
        middle_left_state = True
    return middle_left_corner


def upper_left(vector):
    '''
        Save coordinates x, y, u, v from upper left corner.
        :param vector: x, y, u, v from vector
        :return: coordinates x, y, u, v
        '''
    upper_left_state = False
    upper_left_corner = [0, 0, 0, 0]
    while not upper_left_state:
        upper_left_corner = vector
        upper_left_state = True
    return upper_left_corner


def middle_bottom(vector):
    '''
    Save coordinates x, y, u, v from middle bottom.
    :param vector: x, y, u, v from vector
    :return: coordinates x, y, u, v
    '''
    middle_bottom_state = False
    middle_bottom_corner = [0, 0, 0, 0]
    while not middle_bottom_state:
        middle_bottom_corner = vector
        middle_bottom_state = True
    return middle_bottom_corner


def middle_screen(vector):
    '''
    Save coordinates x, y, u, v from middle.
    :param vector: x, y, u, v from vector
    :return: coordinates x, y, u, v
    '''
    middle_state = False
    middle = [0, 0, 0, 0]
    while not middle_state:
        middle = vector
        middle_state = True
    return middle


def middle_up(vector):
    '''
    Save coordinates x, y, u, v from middle up.
    :param vector: x, y, u, v from vector
    :return: coordinates x, y, u, v
    '''
    middle_up_state = False
    middle_up_corner = [0, 0, 0, 0]
    while not middle_up_state:
        middle_up_corner = vector
        middle_up_state = True
    return middle_up_corner


def lower_right(vector):
    '''
        Save coordinates x, y, u, v from lower right corner.
        :param vector: x, y, u, v from vector
        :return: coordinates x, y, u, v
        '''
    lower_right_state = False
    lower_right_corner = [0, 0, 0, 0]
    while not lower_right_state:
        lower_right_corner = vector
        lower_right_state = True
    return lower_right_corner


def middle_right(vector):
    '''
    Save coordinates x, y, u, v from middle right.
    :param vector: x, y, u, v from vector
    :return: coordinates x, y, u, v
    '''
    middle_right_state = False
    middle_right_corner = [0, 0, 0, 0]
    while not middle_right_state:
        middle_right_corner = vector
        middle_right_state = True
    return middle_right_corner


def upper_right(vector):
    '''
    Save coordinates x, y, u, v from upper right corner.
    :param vector: x, y, u, v from vector
    :return: coordinates x, y, u, v
    '''
    upper_right_state = False
    upper_right_corner = [0, 0, 0, 0]
    while not upper_right_state:
        upper_right_corner = vector
        upper_right_state = True
    return upper_right_corner
