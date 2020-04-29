import keyboard


def lower_left(vector):
    '''
    Save coordinates x, y, u, v from lower left corner.
    :param vector: x, y, u, v from vector
    :return: coordinates x, y, u, v
    '''
    #cv2.circle(mask, (circle_size, screensize[0] - circle_size), circle_size, (0, 0, 255), -1)  # lower left
    lower_left_state = False
    lower_left_corner = [0, 0, 0, 0]
    while not lower_left_state:
        if keyboard.is_pressed("1"):
            lower_left_corner = vector
            lower_left_state = True
            #cv2.circle(mask, (circle_size, screensize[0] - circle_size), circle_size, (255, 255, 255), -1)  # hide circle
    return lower_left_corner

def middle_left(vector):
    '''
    Save coordinates x, y, u, v from middle left.
    :param vector: x, y, u, v from vector
    :return: coordinates x, y, u, v
    '''
    #cv2.circle(mask, (circle_size, screensize[0] - circle_size), circle_size, (0, 0, 255), -1)  # lower left
    middle_left_state = False
    middle_left_corner = [0, 0, 0, 0]
    while not middle_left_state:
        if keyboard.is_pressed("2"):
            middle_left_corner = vector
            middle_left_state = True
            print("middle_left", vector)
            #cv2.circle(mask, (circle_size, screensize[0] - circle_size), circle_size, (255, 255, 255), -1)  # hide circle
    return middle_left_corner

def upper_left(vector):
    '''
        Save coordinates x, y, u, v from upper left corner.
        :param vector: x, y, u, v from vector
        :return: coordinates x, y, u, v
        '''
    #cv2.circle(mask, (circle_size, circle_size), circle_size, (0, 0, 255), -1)  # uper left
    upper_left_state = False
    upper_left_corner = [0, 0, 0, 0]
    while not upper_left_state:
        if keyboard.is_pressed("3"): # 2
            upper_left_corner = vector
            upper_left_state = True
            # cv2.circle(mask, (circle_size, circle_size), circle_size, (255, 255, 255), -1)  # hide circle
    return upper_left_corner


def middle_bottom(vector):
    '''
    Save coordinates x, y, u, v from middle bottom.
    :param vector: x, y, u, v from vector
    :return: coordinates x, y, u, v
    '''
    #cv2.circle(mask, (circle_size, screensize[0] - circle_size), circle_size, (0, 0, 255), -1)  # lower left
    middle_bottom_state = False
    middle_bottom_corner = [0, 0, 0, 0]
    while not middle_bottom_state:
        if keyboard.is_pressed("4"):
            middle_bottom_corner = vector
            middle_bottom_state = True
            print("middle_bottom", vector)
            #cv2.circle(mask, (circle_size, screensize[0] - circle_size), circle_size, (255, 255, 255), -1)  # hide circle
    return middle_bottom_corner


def middle_screen(vector):
    '''
        Save coordinates x, y, u, v from middle.
        :param vector: x, y, u, v from vector
        :return: coordinates x, y, u, v
        '''
    # cv2.circle(mask, (screensize[1] - circle_size, screensize[0] - circle_size), circle_size, (0, 0, 255), -1)  # lower right
    middle_state = False
    middle = [0, 0, 0, 0]
    while not middle_state:
        if keyboard.is_pressed("5"): #3
            middle = vector
            middle_state = True
            # cv2.circle(mask, (screensize[1] - circle_size, screensize[0] - circle_size), circle_size, (255, 255, 255), -1)  # hide circle
    return middle


def middle_up(vector):
    '''
    Save coordinates x, y, u, v from middle up.
    :param vector: x, y, u, v from vector
    :return: coordinates x, y, u, v
    '''
    #cv2.circle(mask, (circle_size, screensize[0] - circle_size), circle_size, (0, 0, 255), -1)  # lower left
    middle_up_state = False
    middle_up_corner = [0, 0, 0, 0]
    while not middle_up_state:
        if keyboard.is_pressed("6"):
            middle_up_corner = vector
            middle_up_state = True
            print("middle_up", vector)
            #cv2.circle(mask, (circle_size, screensize[0] - circle_size), circle_size, (255, 255, 255), -1)  # hide circle
    return middle_up_corner


def lower_right(vector):
    '''
        Save coordinates x, y, u, v from lower right corner.
        :param vector: x, y, u, v from vector
        :return: coordinates x, y, u, v
        '''
    #cv2.circle(mask, (screensize[1] - circle_size, screensize[0] - circle_size), circle_size, (0, 0, 255), -1)  # lower right
    lower_right_state = False
    lower_right_corner = [0, 0, 0, 0]
    while not lower_right_state:
        if keyboard.is_pressed("7"): #4
            lower_right_corner = vector
            lower_right_state = True
            # cv2.circle(mask, (screensize[1] - circle_size, screensize[0] - circle_size), circle_size, (255, 255, 255), -1)  # hide circle
    return lower_right_corner


def middle_right(vector):
    '''
        Save coordinates x, y, u, v from middle right.
        :param vector: x, y, u, v from vector
        :return: coordinates x, y, u, v
        '''
    #cv2.circle(mask, (screensize[1] - circle_size, screensize[0] - circle_size), circle_size, (0, 0, 255), -1)  # lower right
    middle_right_state = False
    middle_right_corner = [0, 0, 0, 0]
    while not middle_right_state:
        if keyboard.is_pressed("8"):
            middle_right_corner = vector
            middle_right_state = True
            print("middle_right", vector)
            # cv2.circle(mask, (screensize[1] - circle_size, screensize[0] - circle_size), circle_size, (255, 255, 255), -1)  # hide circle
    return middle_right_corner


def upper_right(vector):
    '''
        Save coordinates x, y, u, v from upper right corner.
        :param vector: x, y, u, v from vector
        :return: coordinates x, y, u, v
        '''
    #cv2.circle(mask, (screensize[1] - circle_size, circle_size), circle_size, (0, 0, 255), -1)  # uper right
    upper_right_state = False
    upper_right_corner = [0, 0, 0, 0]
    while not upper_right_state:
        if keyboard.is_pressed("9"): #5
            upper_right_corner = vector
            upper_right_state = True
            # cv2.circle(mask, (screensize[1] - circle_size, circle_size), circle_size, (255, 255, 255), -1)  # hide circle
    return upper_right_corner
