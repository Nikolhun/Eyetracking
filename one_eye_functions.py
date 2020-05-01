import math

def eye_center(position_in_frame, move_frame):
    height, width = position_in_frame.shape
    eye_center_coordinates_in_frame = (int(width/2), int(height/2))
    eye_center_coordinates = (int(width/2 + move_frame[1]), int(height/2 + move_frame[0]))
    return eye_center_coordinates_in_frame, eye_center_coordinates


def find_vector(pupil_center, eye_center_coordinates):
    x = pupil_center[0] - eye_center_coordinates[0]  # vector x
    y = pupil_center[1] - eye_center_coordinates[1]  # vector y

    if y == 0:
        y = 0.000001

    magnitude = math.sqrt(x * x + y * y)  # magnitude of vector
    direction_radian = math.atan(x / y)  # direction in radians
    direction = (direction_radian * 180) / math.pi  # from radians to degrees

    return x, y, magnitude, direction


