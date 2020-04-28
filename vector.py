import math

def calibrate_vector_eye_center(center_pupil_in_eye_frame):
    calibrating_vector_in_frame = [center_pupil_in_eye_frame[0], center_pupil_in_eye_frame[1], 0, 0]
    return calibrating_vector_in_frame

def vector_start_center(calibrating_vector_in_frame, corner):
    vector_center = [calibrating_vector_in_frame[0] + corner[0], calibrating_vector_in_frame[1] + corner[1]]
    return vector_center

def find_vector(left_center_pupil, left_calibrating_point, right_center_pupil, right_calibrating_point):
    '''
    Detects direction of view. Left and right eye are averaged to supress squint or imperfect detection.
    :param left_center_pupil: center of left pupil
    :param left_calibrating_point: center of left eye
    :param right_center_pupil: center of right pupil
    :param right_calibrating_point: center of right eye
    :return: [x, y, magnitude, direction], vector of position x, y it's magnitude and direction
    '''
    l_x = left_center_pupil[0] - left_calibrating_point[0]  # left eye x
    l_y = left_center_pupil[1] - left_calibrating_point[1]   # left eye y
    r_x = right_center_pupil[0] - right_calibrating_point[0]  # right eye x
    r_y = right_center_pupil[1] - right_calibrating_point[1]   # right eye y

    l_magnitude = math.sqrt(l_x * l_x + l_y * l_y)  # magnitude of left eye
    r_magnitude = math.sqrt(r_x * r_x + r_y * r_y)  # magnitude of right eye

    if l_y == 0:
        l_y = 0.000001
    if r_y == 0:
        r_y = 0.000001

    l_direction_radian = math.atan(l_x/l_y)  # direction of left eye
    r_direction_radian = math.atan(r_x/r_y)  # direction of right eye

    l_direction = (l_direction_radian * 180) / math.pi  # from radians to degrees for left eye
    r_direction = (r_direction_radian * 180) / math.pi  # from radians to degrees for right eye

    output_vector = [(l_x + r_x)/2, (l_y + r_y)/2, (r_magnitude + l_magnitude)/2,
                     (r_direction + l_direction)/2]  # vector [x, y, magnitude, direction]
    return output_vector

