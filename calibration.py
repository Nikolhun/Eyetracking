import keyboard
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata



#######################################################################################################################
# ---------------------------------- Functions ---------------------------------------------- #
#######################################################################################################################
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


def lower_right(vector):
    #cv2.circle(mask, (screensize[1] - circle_size, screensize[0] - circle_size), circle_size, (0, 0, 255), -1)  # lower right
    #print('Look into lower right corner and press ENTER.')
    lower_right_state = False
    lower_right_corner = [0, 0, 0]
    while not lower_right_state:
        if keyboard.is_pressed("3"):
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
        if keyboard.is_pressed("4"):
            upper_right_corner = vector
            upper_right_state = True
            # cv2.circle(mask, (screensize[1] - circle_size, circle_size), circle_size, (255, 255, 255), -1)  # hide circle
    return upper_right_corner


#######################################################################################################################
# ---------------------------------- Calibration -------------------------------------------------------------------- #
#######################################################################################################################
def interpolate(lower_left_corner, upper_left_corner, lower_right_corner, upper_right_corner):
    '''
    Interpolates vectors from measured corners. You need x, y, u and v of a vector.
    :param lower_left_corner: [x, y, u, v] of lower left corner
    :param upper_left_corner: [x, y, u, v] of upper left corner
    :param lower_right_corner: [x, y, u, v] of lower right corner
    :param upper_right_corner: [x, y, u, v] of upper right corner
    :return:
    '''

    x = [0, 0, 100, 100]
    y = [0, 100, 0, 100]
    u = [lower_left_corner[2], upper_left_corner[2], lower_right_corner[2], upper_right_corner[2]]
    v = [lower_left_corner[3], upper_left_corner[3], lower_right_corner[3], upper_right_corner[3]]

    plt.figure(1)
    plt.quiver(x, y, u, v)  # show measured vectors

    xx = np.linspace(min(x), max(x), 20)  # new x ax for interpolated data
    yy = np.linspace(min(y), max(y), 20)  # new y ax for interpolated data
    xx, yy = np.meshgrid(xx, yy)

    points = np.transpose(np.vstack((x, y)))
    u_interp = griddata(points, u, (xx, yy), method='cubic')  # interpolate u
    v_interp = griddata(points, v, (xx, yy), method='cubic')  # interpolate v

    plt.figure(2)
    plt.quiver(xx, yy, u_interp, v_interp)  # show interpolated vectors
    plt.show()

    return u_interp, v_interp

#ll = [-14, -1, 15, 85]
#ul = [-15, -4, 16, 74]
#lr = [5, 1, 6, -24]
#ur = [6, -6, 9, -45]
