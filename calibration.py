import keyboard


def lower_left(vector):
    #cv2.circle(mask, (circle_size, screensize[0] - circle_size), circle_size, (0, 0, 255), -1)  # lower left
    #print('Look into lower left corner and press ENTER.')
    lower_left_state = False
    lower_left_corner = [0, 0, 0]
    while not lower_left_state:
        if keyboard.is_pressed("1"):
            lower_left_corner = vector
            lower_left_state = True
            #cv2.circle(mask, (circle_size, screensize[0] - circle_size), circle_size, (255, 255, 255), -1)  # hide circle
    return lower_left_corner


def upper_left(vector):
    #cv2.circle(mask, (circle_size, circle_size), circle_size, (0, 0, 255), -1)  # uper left
    #print('Look into upper left corner and press ENTER.')
    upper_left_state = False
    upper_left_corner = [0, 0, 0]
    while not upper_left_state:
        if keyboard.is_pressed("2"):
            upper_left_corner = vector
            upper_left_state = True
            # cv2.circle(mask, (circle_size, circle_size), circle_size, (255, 255, 255), -1)  # hide circle
    return upper_left_corner


def middle_screen(vector):
    # cv2.circle(mask, (screensize[1] - circle_size, screensize[0] - circle_size), circle_size, (0, 0, 255), -1)  # lower right
    # print('Look into lower right corner and press ENTER.')
    middle_state = False
    middle = [0, 0, 0]
    while not middle_state:
        if keyboard.is_pressed("3"):
            middle = vector
            middle_state = True
            # cv2.circle(mask, (screensize[1] - circle_size, screensize[0] - circle_size), circle_size, (255, 255, 255), -1)  # hide circle
    return middle


def lower_right(vector):
    #cv2.circle(mask, (screensize[1] - circle_size, screensize[0] - circle_size), circle_size, (0, 0, 255), -1)  # lower right
    #print('Look into lower right corner and press ENTER.')
    lower_right_state = False
    lower_right_corner = [0, 0, 0]
    while not lower_right_state:
        if keyboard.is_pressed("4"):
            lower_right_corner = vector
            lower_right_state = True
            # cv2.circle(mask, (screensize[1] - circle_size, screensize[0] - circle_size), circle_size, (255, 255, 255), -1)  # hide circle
    return lower_right_corner


def upper_right(vector):
    #cv2.circle(mask, (screensize[1] - circle_size, circle_size), circle_size, (0, 0, 255), -1)  # uper right
    #print('Look into upper right corner and press ENTER.')
    upper_right_state = False
    upper_right_corner = [0, 0, 0]
    while not upper_right_state:
        if keyboard.is_pressed("5"):
            upper_right_corner = vector
            upper_right_state = True
            # cv2.circle(mask, (screensize[1] - circle_size, circle_size), circle_size, (255, 255, 255), -1)  # hide circle
    return upper_right_corner
