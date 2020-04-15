import math


def find_vector(left_center_pupil, left_center_eye, right_center_pupil, right_center_eye):
    '''
    Detects direction of view. Left and right eye are averaged to supress squint or imperfect detection.
    :param left_center_pupil: center of left pupil
    :param left_center_eye: center of left eye
    :param right_center_pupil: center of right pupil
    :param right_center_eye: center of right eye
    :return: [x, y, size], vector of position x, y and it's size
    '''
    l_x = left_center_pupil[0] - left_center_eye[0]  # left eye x
    l_y = left_center_pupil[1] - left_center_eye[1]   # left eye y
    r_x = right_center_pupil[0] - right_center_eye[0]  # right eye x
    r_y = right_center_pupil[1] - right_center_eye[1]   # right eye y

    l_size = math.sqrt(l_x * l_x + l_y * l_y)  # size of left eye
    r_size = math.sqrt(r_x * r_x + r_y * r_y)  # size of right eye

    output_vector = [int((l_x + r_x)/2)*10, int((l_y + r_y)/2)*10, int((r_size + l_size)/2)]  # vector [x, y, size]
    return output_vector