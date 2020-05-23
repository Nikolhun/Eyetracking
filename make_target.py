import cv2
import random
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np

def check_target_spot_before(draw_point_after_next_target, hit_target, mask_for_eyetracking_bgr, coordinates_of_center_dot,
                      value_of_point, hit_target_value):
    """
    Saves spots that are red before target deletes them.
    :param draw_point_after_next_target: If it should draw hit after deleting target = true
    :param hit_target: was there a target hit = true
    :param mask_for_eyetracking_bgr: array where is eyetracker result (white canvas, red pixels)
    :param coordinates_of_center_dot: coordinates of target (x, y)
    :param value_of_point: value of red pixel before target
    :param hit_target_value: saved hits, length of this is number of hits
    :return: array where is eyetracker result; if there was a hit; value of hit
    """
    if draw_point_after_next_target or hit_target:
        mask_for_eyetracking_bgr[coordinates_of_center_dot[1]][coordinates_of_center_dot[0]][0] = 0
        mask_for_eyetracking_bgr[coordinates_of_center_dot[1]][coordinates_of_center_dot[0]][1] = 0
        if draw_point_after_next_target and hit_target:

            mask_for_eyetracking_bgr[coordinates_of_center_dot[1]][coordinates_of_center_dot[0]][2] = value_of_point
            draw_point_after_next_target = False

        elif not draw_point_after_next_target and hit_target:

            if 5 * len(hit_target_value) >= 104:
                mask_for_eyetracking_bgr[coordinates_of_center_dot[1]][coordinates_of_center_dot[0]][2] = 151
            else:
                mask_for_eyetracking_bgr[coordinates_of_center_dot[1]][coordinates_of_center_dot[0]][2] = \
                    255 - 5 * len(hit_target_value)

        elif draw_point_after_next_target and hit_target:

            draw_point_after_next_target = False

            if value_of_point - 5 * len(hit_target_value) << 151:
                mask_for_eyetracking_bgr[coordinates_of_center_dot[1]][coordinates_of_center_dot[0]][2] = 151
            else:
                mask_for_eyetracking_bgr[coordinates_of_center_dot[1]][coordinates_of_center_dot[0]][2] = \
                    value_of_point - 5 * len(hit_target_value)

    mask_for_eyetracking_bgr_out = mask_for_eyetracking_bgr
    return mask_for_eyetracking_bgr_out, draw_point_after_next_target, value_of_point


def check_target_spot_after(mask_for_eyetracking_bgr, coordinates_of_center_dot):
    """
    Check and save hited pixels after deleting target
    :param mask_for_eyetracking_bgr: array where is eyetracker result (white canvas, red pixels)
    :param coordinates_of_center_dot: coordinates of target (x, y)
    :return: array where is eyetracker result; if there was a hit; value of hit
    """
    if mask_for_eyetracking_bgr[coordinates_of_center_dot[1]][coordinates_of_center_dot[0]][0] == 0 and \
            mask_for_eyetracking_bgr[coordinates_of_center_dot[1]][coordinates_of_center_dot[0]][1] == 0 and \
            mask_for_eyetracking_bgr[coordinates_of_center_dot[1]][coordinates_of_center_dot[0]][2] >> 1:
        draw_point_after_next_target = True
        value_of_point = mask_for_eyetracking_bgr[coordinates_of_center_dot[1]][coordinates_of_center_dot[0]][2]
    else:
        draw_point_after_next_target = False
        value_of_point = 0
    mask_for_eyetracking_bgr_out = mask_for_eyetracking_bgr
    return mask_for_eyetracking_bgr_out, draw_point_after_next_target, value_of_point



def draw_line(mask_for_eyetracking_bgr, coordinates_of_center_dot, step, color):
    """
    draw target into scene
    :param mask_for_eyetracking_bgr: scene, array where is eyetracker result (white canvas, red pixels)
    :param coordinates_of_center_dot: coordinates of target (x, y)
    :param step: size of a target
    :param color: color of a target
    :return: it is used for drawing and deleting targets
    """
    cv2.line(mask_for_eyetracking_bgr, (coordinates_of_center_dot[0] - step, coordinates_of_center_dot[1]),
             (coordinates_of_center_dot[0] + step, coordinates_of_center_dot[1]), color, 1)
    cv2.line(mask_for_eyetracking_bgr, (coordinates_of_center_dot[0], coordinates_of_center_dot[1] - step),
             (coordinates_of_center_dot[0], coordinates_of_center_dot[1] + step), color, 1)


def change_coordinates_of_target_random(size_of_output_screen):
    """
    function for changing targets randomly
    :param size_of_output_screen: size of screen (x, y)
    :return: coordinates of new target (x, y)
    """
    num_rows = random.randint(0, size_of_output_screen[0]-1)
    num_coll = random.randint(0, size_of_output_screen[1]-1)
    return num_rows, num_coll


def change_coordinates_of_target(coordinates_of_center_dot, size_of_output_screen, part, change_accepted,
                                 acceptation_of_change):
    """
    The journey of animated target
    :param coordinates_of_center_dot: coordinates of old target (x, y)
    :param size_of_output_screen: size of screen (x, y)
    :param part: part of the line (1-16)
    :param change_accepted: if the change should be
    :param acceptation_of_change: line of non changed rounds
    :return:  coordinates of new target (x, y); part of line, acceptation_of_change, change_accepted
    """
    left_right = coordinates_of_center_dot[0]
    top_bottom = coordinates_of_center_dot[1]
    part_out = 0

    if left_right <= int((80*size_of_output_screen[0])/100) and part == 1 and change_accepted:
        left_right = left_right + 1
        change_accepted = False
        part_out = 1
    if left_right == int((80*size_of_output_screen[0])/100) and part == 1:
        part_out = 2

    if top_bottom <= int((80 * size_of_output_screen[1]) / 100) and part == 2 and change_accepted:
        top_bottom = top_bottom + 1
        change_accepted = False
        part_out = 2
    if top_bottom == int((80 * size_of_output_screen[1]) / 100) and part == 2:
        part_out = 3

    if left_right >= int((30 * size_of_output_screen[0]) / 100) and part == 3 and change_accepted:
        left_right = left_right - 1
        change_accepted = False
        part_out = 3
    if left_right == int((30 * size_of_output_screen[0]) / 100) and part == 3:
        part_out = 4

    if top_bottom >= int((55 * size_of_output_screen[1]) / 100) and part == 4 and change_accepted:
        top_bottom = top_bottom - 1
        change_accepted = False
        part_out = 4
    if top_bottom == int((55 * size_of_output_screen[1]) / 100) and part == 4:
        part_out = 5

    if left_right <= int((65 * size_of_output_screen[0]) / 100) and part == 5 and change_accepted:
        left_right = left_right + 1
        change_accepted = False
        part_out = 5
    if left_right == int((65 * size_of_output_screen[0]) / 100) and part == 5:
        part_out = 6

    if top_bottom >= int((40 * size_of_output_screen[1]) / 100) and part == 6 and change_accepted:
        top_bottom = top_bottom - 1
        change_accepted = False
        part_out = 6
    if top_bottom == int((40 * size_of_output_screen[1]) / 100) and part == 6:
        part_out = 7

    if left_right >= int((20 * size_of_output_screen[0]) / 100) and part == 7 and change_accepted:
        left_right = left_right - 1
        change_accepted = False
        part_out = 7
    if left_right == int((20 * size_of_output_screen[0]) / 100) and part == 7:
        part_out = 8

    if top_bottom >= int((20 * size_of_output_screen[1]) / 100) and part == 8 and change_accepted:
        top_bottom = top_bottom - 1
        change_accepted = False
        part_out = 8
    if top_bottom == int((20 * size_of_output_screen[1]) / 100) and part == 8:
        part_out = 9

    if left_right >= 0 and part == 9 and change_accepted:
        left_right = left_right - 1
        change_accepted = False
        part_out = 9
    if left_right == 0 and part == 9:
        part_out = 10

    if top_bottom >= 0 and part == 10 and change_accepted:
        top_bottom = top_bottom - 1
        change_accepted = False
        part_out = 10
    if top_bottom == 0 and part == 10:
        part_out = 11

    if left_right <= size_of_output_screen[0]-1 and part == 11 and change_accepted:
        left_right = left_right + 1
        change_accepted = False
        part_out = 11
    if left_right == size_of_output_screen[0]-1 and part == 11:
        part_out = 12

    if top_bottom <= size_of_output_screen[1]-1 and part == 12 and change_accepted:
        top_bottom = top_bottom + 1
        change_accepted = False
        part_out = 12
    if top_bottom == size_of_output_screen[1]-1 and part == 12:
        part_out = 13

    if left_right >= 0 and part == 13 and change_accepted:
        left_right = left_right - 1
        change_accepted = False
        part_out = 13
    if left_right == 0 and part == 13:
        part_out = 14

    if top_bottom >= int(size_of_output_screen[1]/2) and part == 14 and change_accepted:
        top_bottom = top_bottom - 1
        change_accepted = False
        part_out = 14
    if top_bottom == int(size_of_output_screen[1]/2) and part == 14:
        part_out = 15

    if left_right <= int(size_of_output_screen[0]/2) and part == 15 and change_accepted:
        left_right = left_right + 1
        change_accepted = False
        part_out = 15
    if left_right == int(size_of_output_screen[0]/2) and part == 15:
        part_out = 16

    if part_out == 0:
        part_out = part

    coordinates_of_center_dot_output = (left_right, top_bottom)
    if coordinates_of_center_dot_output == coordinates_of_center_dot:
        acceptation_of_change.append(1)
    return coordinates_of_center_dot_output, part_out, acceptation_of_change, change_accepted


def heat_map(eyetracker_screen_bgr, screensize, name):
    """
    Makes and saves heat map.
    :param eyetracker_screen_bgr: screen scene array
    :param screensize: size of the screen (x, y)
    :param name: name of the heat map
    :return: makes and saves heat map with name
    """
    plt.rcParams['figure.figsize'] = (16, 9)
    eyetracker_screen_gray = cv2.cvtColor(cv2.resize(eyetracker_screen_bgr, screensize,
                                      interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY)
    eyetracker_screen_gray = np.abs((eyetracker_screen_gray / eyetracker_screen_gray.max()) - 1)

    heat_map_cubic = sb.heatmap(eyetracker_screen_gray,
                                center=(eyetracker_screen_gray.max() - eyetracker_screen_gray.min()) / 1.5,
                                vmin=0, vmax=1, xticklabels=False, yticklabels=False, cmap="Blues", cbar=False)
    plt.show()
    heat_map_cubic.get_figure().savefig("results/heat_map_" + name + ".png")  # dpi = 400


def show_target(mask_bgr, coordinates_of_center_dot):
    """
    Shows target in screen array.
    :param mask_bgr:  screen array
    :param coordinates_of_center_dot: coordinates of target
    :return: coordinations of targer x and y, changed screen array
    """
    mask_bgr[np.abs(coordinates_of_center_dot[1])][coordinates_of_center_dot[0]][0] = 255
    mask_bgr[np.abs(coordinates_of_center_dot[1])][coordinates_of_center_dot[0]][1] = 0
    mask_bgr[np.abs(coordinates_of_center_dot[1])][coordinates_of_center_dot[0]][2] = 0

    return coordinates_of_center_dot[1], coordinates_of_center_dot[0], mask_bgr