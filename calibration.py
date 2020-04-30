import numpy as np
import cv2


def prepare_mask_for_calibration(screensize, press, output_vector_in_eye_frame):
    mask = np.zeros((screensize[0], screensize[1]), np.uint8) + 255  # mask with size of screen and value 255
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.namedWindow('calibration', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # set window to full screen
    circle_size = 40

    if press == 1:
        coordinates = (circle_size, screensize[0] - circle_size)
    elif press == 2:
        coordinates = (circle_size, int(screensize[0]/2))
    elif press == 3:
        coordinates = (circle_size, circle_size)
    elif press == 4:
        coordinates = (int(screensize[1]/2), screensize[0] - circle_size)
    elif press == 5:
        coordinates = (int(screensize[1]/2), int(screensize[0]/2))
    elif press == 6:
        coordinates = (int(screensize[1]/2), circle_size)
    elif press == 7:
        coordinates = (screensize[1] - circle_size, screensize[0] - circle_size)
    elif press == 8:
        coordinates = (screensize[1] - circle_size, int(screensize[0]/2))
    elif press == 9:
        coordinates = (screensize[1] - circle_size, circle_size)
    else:
        coordinates = []
        print("Calibration works only with number 0-9.")

    cv2.circle(mask, coordinates, 10, (0, 0, 255), -1)  # lower left
    start_point = (int(screensize[1] / 2), int(screensize[0] / 2))
    end_point = (int(output_vector_in_eye_frame[1] * 10 + start_point[1]),
                 int(output_vector_in_eye_frame[0] * 10 + start_point[0]))
    cv2.arrowedLine(mask, start_point, end_point, color=(0, 255, 0), thickness=1)
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
