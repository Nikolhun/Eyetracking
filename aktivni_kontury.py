import numpy as np
import cv2
import dlib

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def draw_point(number):
    x = landmarks.part(number).x
    y = landmarks.part(number).y
    cv2.line(frame, (x, y), (x, y), (0, 0, 255), 2, 1)

def crop_eye(number1, number2, number3, number4, number5, number6):
    eye = np.array([(landmarks.part(number1).x, landmarks.part(number1).y),
                    (landmarks.part(number2).x, landmarks.part(number2).y),
                    (landmarks.part(number3).x, landmarks.part(number3).y),
                    (landmarks.part(number4).x, landmarks.part(number4).y),
                    (landmarks.part(number5).x, landmarks.part(number5).y),
                    (landmarks.part(number6).x, landmarks.part(number6).y)], np.int32)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [eye], True, 255, 2)
    cv2.fillPoly(mask, [eye], 255)
    eye_fill = cv2.bitwise_and(frame, frame, mask, mask)
    cv2.imshow("eye_fill", eye_fill)

    min_x = np.min(eye[:, 0])
    max_x = np.max(eye[:, 0])
    min_y = np.min(eye[:, 1])
    max_y = np.max(eye[:, 1])
    eye_crop = eye_fill[min_y: max_y, min_x: max_x]

    _, threshold_eye = cv2.threshold(eye_crop, 70, 255, cv2.THRESH_BINARY)
    threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
    eye_crop = cv2.resize(eye_crop, None, fx=5, fy=5)
    cv2.imshow("threshold_eye", threshold_eye)
    cv2.imshow("eye_crop", eye_crop)
    cv2.imshow("eye_fill", eye_fill)
    cv2.polylines(frame, [eye], True, (0, 0, 255), 1)
    return eye_crop

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        # view face frame
        # x, y = face.left(), face.top()
        # x1, y1 = face.right(), face.bottom()
        # cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        # detect face structures using landmarks
        landmarks = predictor(gray, face)

        # crop eyes
        left_eye_crop = crop_eye(36, 37, 38, 39, 40, 41)
        #right_eye_crop = crop_eye(42, 43, 44, 45, 46, 47)
        # cv2.imshow("Right Eye", right_eye_crop)
        # cv2.imshow("Left Eye", left_eye_crop)

        # draw points into eyes
        #for i in range(36, 48):
           # draw_point(i)

        # draw points into face
        for i in range(0, 67):
            draw_point(i)

        # draw horizontal and vertical lines
        # left_point = (landmarks.part(36).x, landmarks.part(36).y)
        # right_point = (landmarks.part(39).x, landmarks.part(39).y)
        # center_top = midpoint(landmarks.part(37), landmarks.part(38))
        # center_bottom = midpoint(landmarks.part(41), landmarks.part(40))
        # hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
        # ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()