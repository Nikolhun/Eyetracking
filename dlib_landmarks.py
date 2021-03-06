import cv2
import numpy as np


def view_face_frame(face, frame):
    """
    View face frame.
    :param face: face from faces
    :param frame: frame you want to view face to
    """
    x, y = face.left(), face.top()
    x1, y1 = face.right(), face.bottom()
    cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)


def midpoint(p1, p2):
    """
    Finds midpoint of two numbers.
    :param p1: first number
    :param p2: second number
    :return: midpoint
    """
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)


def draw_point(number, landmarks, frame):
    """
    Draws points that shows face structures in Dlib.
    :param number: number of landmark
    :param landmarks: landmarks from predictor_dlib(gray, face)
    :param frame: frame you want to draw points in
    """
    x = landmarks.part(number).x
    y = landmarks.part(number).y
    cv2.line(frame, (x, y), (x, y), (0, 0, 255), 2, 1)


def eye_center_dlib(eye, position_for_eye):
    """
    Gets center of eye in eye frame and in frame.
    :param eye: eye array in gray
    :param position_for_eye: position [0, 0] of eye frame
    :return: [center, center_in_frame] center is center in whole array, center_in_frame is center in eye frame
    """
    height, width = eye.shape
    center = [int(width/2) + position_for_eye[0], int(height/2) + position_for_eye[1]]
    center_in_frame = [int(width/2), int(height/2)]
    return center, center_in_frame


def landmarks_array(number1, number2, number3, number4, number5, number6, landmarks, frame, lines):
    """
    Saves landmarks into array. Can print lines in eye if lines = 1.
    :param number1 - number6: number of landmarks from landmark map
    :param landmarks: output from function predictor_dlib
    :param frame: image in gray
    :param lines: lines = 0 -> dont draw lines, lines = 1 -> draw lines
    :return: array of landmarks number1 - number6 in int32
    """
    l_array = np.array([(landmarks.part(number1).x, landmarks.part(number1).y),
                        (landmarks.part(number2).x, landmarks.part(number2).y),
                        (landmarks.part(number3).x, landmarks.part(number3).y),
                        (landmarks.part(number4).x, landmarks.part(number4).y),
                        (landmarks.part(number5).x, landmarks.part(number5).y),
                        (landmarks.part(number6).x, landmarks.part(number6).y)], np.int32)
    if lines == 1:  # draw horizontal and vertical lines
        cv2.line(frame, (landmarks.part(number1).x, landmarks.part(number1).y),
                        (landmarks.part(number4).x, landmarks.part(number4).y),
                        (0, 255, 0), 1)  # horizontal line
        cv2.line(frame, midpoint(landmarks.part(number2), landmarks.part(number3)),
                 midpoint(landmarks.part(number6), landmarks.part(number5)), (0, 255, 0), 1)  # vertical line
    return l_array


def fill_frame(img, left_array, right_array):
    """
    Get eye from image
    :param img: image in gray
    :param left_array: array of left eye landmarks
    :param right_array: array of right eye landmarks
    :return: image in gray
    """
    height, width = img.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_array], True, 255, 2)
    cv2.polylines(mask, [right_array], True, 255, 2)
    cv2.fillPoly(mask, [left_array], 255)
    cv2.fillPoly(mask, [right_array], 255)
    # eye_filling = cv2.bitwise_and(img, img, mask, mask)
    return img


def crop_eyes(eye_fill, eye_array):
    """
    Crop eye region from frame
    :param eye_fill: img in gray
    :param eye_array: array of eye landmarks
    :return: croped eye in gray, corner of image (coordinate of eye crop in frame)
    """
    min_x = np.min(eye_array[:, 0])
    max_x = np.max(eye_array[:, 0])
    min_y = np.min(eye_array[:, 1])
    max_y = np.max(eye_array[:, 1])
    eye_crop = eye_fill[min_y:max_y, min_x: max_x]
    corner = [min_x, min_y]
    return eye_crop, corner
