import cv2
import numpy as np

def view_face_frame(face, frame):
    x, y = face.left(), face.top()
    x1, y1 = face.right(), face.bottom()
    cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
    return


def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)


def draw_point(number, landmarks, frame):
    x = landmarks.part(number).x
    y = landmarks.part(number).y
    cv2.line(frame, (x, y), (x, y), (0, 0, 255), 2, 1)


def eye_center(eye):
    height, width = eye.shape
    x = int(height/2)
    y = int(width/2)
    center = [x, y]
    return center


def eye_center_dlib(eye, position_for_eye):
    height, width = eye.shape
    y = int(height/2)
    x = int(width/2)
    center = [x + position_for_eye[0], y + position_for_eye[1]]
    center_in_frame = [x, y]
    return center, center_in_frame

def landmarks_array(number1, number2, number3, number4, number5, number6, landmarks, frame, lines):
    '''
    Saves landmarks into array. Can print lines in eye if lines = 1.
    :param number1 - number6: number of landmarks from landmark map
    :param landmarks: output from function predictor_dlib
    :param frame: image in gray
    :param lines: lines = 0 -> dont draw lines, lines = 1 -> draw lines
    :return: array of landmarks number1 - number6 in int32
    '''
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
    # not ideal yet
    '''
    Get eye from image
    :param img: image in gray
    :param left_array: array of left eye landmarks
    :param right_array: array of right eye landmarks
    :return: image in gray
    '''
    height, width = img.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_array], True, 255, 2)
    cv2.polylines(mask, [right_array], True, 255, 2)
    cv2.fillPoly(mask, [left_array], 255)
    cv2.fillPoly(mask, [right_array], 255)
    #eye_filling = cv2.bitwise_and(frame, frame, mask_inv, mask_inv)
    eye_filling = cv2.bitwise_and(img, img, mask, mask)
    #cv2.imshow("eye_fill", eye_filling)
    return img

def fill_frame_gui(img, array):
    # not ideal yet
    '''
    Get eye from image
    :param img: image in gray
    :param _array: array of eye landmarks
    :return: image in gray
    '''
    height, width = img.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [array], True, 255, 2)
    cv2.fillPoly(mask, [array], 255)
    #eye_filling = cv2.bitwise_and(frame, frame, mask_inv, mask_inv)
    eye_filling = cv2.bitwise_and(img, img, mask, mask)
    #cv2.imshow("eye_fill", eye_filling)
    return img


def crop_eyes(eye_fill, eye_array):
    '''
    Crop eye region from frame
    :param eye_fill: img in gray
    :param eye_array: array of eye landmarks
    :return: croped eye in gray, corner of image (coordinate of eye crop in frame)
    '''
    min_x = np.min(eye_array[:, 0])
    max_x = np.max(eye_array[:, 0])
    min_y = np.min(eye_array[:, 1])
    max_y = np.max(eye_array[:, 1])
    eye_crop = eye_fill[min_y:max_y, min_x: max_x]
    corner = [min_x, min_y]
    return eye_crop, corner