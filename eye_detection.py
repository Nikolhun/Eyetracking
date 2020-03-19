import numpy as np
import cv2
import dlib
from scipy import signal

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def draw_point(number):
    x = landmarks.part(number).x
    y = landmarks.part(number).y
    cv2.line(frame, (x, y), (x, y), (0, 0, 255), 2, 1)

def eye_center(eye):
    height, width, _ = eye.shape
    x = int(height/2)
    y = int(width/2)
    return x, y

def landmarks_array(number1, number2, number3, number4, number5, number6):
    l_array = np.array([(landmarks.part(number1).x, landmarks.part(number1).y),
                        (landmarks.part(number2).x, landmarks.part(number2).y),
                        (landmarks.part(number3).x, landmarks.part(number3).y),
                        (landmarks.part(number4).x, landmarks.part(number4).y),
                        (landmarks.part(number5).x, landmarks.part(number5).y),
                        (landmarks.part(number6).x, landmarks.part(number6).y)], np.int32)
    return l_array

def fill_frame (frame, left_array, right_array):
    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_array], True, 255, 2)
    cv2.polylines(mask, [right_array], True, 255, 2)
    cv2.fillPoly(mask, [left_array], 255)
    cv2.fillPoly(mask, [right_array], 255)
    eye_filling = cv2.bitwise_and(frame, frame, mask, mask)
    #cv2.imshow("maska", mask)
    return eye_filling

def crop_eyes (eye_fill, eye_array):
    min_x = np.min(eye_array[:, 0])
    max_x = np.max(eye_array[:, 0])
    min_y = np.min(eye_array[:, 1])
    max_y = np.max(eye_array[:, 1])
    eye_crop = eye_fill[min_y: max_y, min_x: max_x]
    #cv2.imshow("eye_crop", eye_crop)
    return eye_crop

def resize_eye (eye_crop,l_array):
    _, threshold_eye = cv2.threshold(eye_crop, 70, 255, cv2.THRESH_BINARY)
    threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
    resized_eye = cv2.resize(eye_crop, None, fx=5, fy=5)
    #cv2.imshow("threshold_eye", threshold_eye)
    #cv2.imshow("eye_crop", eye_crop)
    #cv2.imshow("eye_fill", eye_fill)
    cv2.polylines(frame, [l_array], True, (0, 0, 0), 1)
    return resized_eye, threshold_eye

def detect_pupil(eye_crop):
    eye_crop = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
    #thres = cv2.inRange(eye_crop, 0, 20)
    eye_crop = cv2.equalizeHist(eye_crop)
    eye_crop = cv2.bilateralFilter(eye_crop, 9, 75, 75)
    kernel = np.ones((3, 3), np.uint8)
    # /------- removing small noise inside the white image ---------/#
    dilation = cv2.dilate(eye_crop, kernel, iterations=1)
    # /------- decreasing the size of the white region -------------/#
    erosion = cv2.erode(dilation, kernel, iterations=1)

    # Canny
    edges = cv2.Canny(erosion, 20, 40)
    # cv2.imshow("edges", edges)

    # Laplacian
    laplacian = cv2.Laplacian(erosion, cv2.CV_16U, ksize=3)
    # cv2.imshow("Laplacian", laplacian)
    return laplacian, edges

def detect_hough_circles(edges):
    gray = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
    gray_shape = gray.shape
    # Process the image for circles using the Hough transform
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, gray_shape[1], 50, 30, 1, 2)
    #circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, gray_shape[1], 50, 30, 1, 40)
    shape_circles = circles.shape
    # Determine if any circles were found
    if circles is None:
        print("No circles found")
    elif len(shape_circles) == 1 or len(shape_circles) == 2:
        print("No circles found")
    else:
        # convert the (x, y) coordinates and radius
        # of the circles to integers
        circles = np.uint16(np.around(circles))
        print(circles)
        for pt in circles[0, :]:
            a, b, r, = pt[0], pt[1], pt[2]
            # Draw the circumference of the circle.
            cv2.circle(edges, (a, b), r, (0, 255, 0), 3)
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(edges, (a, b), 1, (255, 0, 0), 3)
    cv2.imshow('detected circles', edges)

    return circles



# nepotřebná fce
#def eye_crop_rectangle (number1, number2, number3, number4, number5, number6):
#    #x1 = 42x
#    x1 = landmarks.part(number1).x
#    #y1 = 46y, 47y menší
    #   if landmarks.part(number5).y < landmarks.part(number6).y:
    #       lower_46y_47y = landmarks.part(number5).y
    #else:
    #   lower_46y_47y = landmarks.part(number6).y
    #y1 = lower_46y_47y
    ###x2
    #x2 = landmarks.part(number1).x
    ##y2 = 43y nebo 44y, vetsi
    #if landmarks.part(number2).y < landmarks.part(number3).y:
    #    higher_43y_44y = landmarks.part(number3).y
    #else:
    #    higher_43y_44y = landmarks.part(number2).y
    #y2 = higher_43y_44y
    ##x3 = 45x
    #x3 = landmarks.part(number4).x
    ##y3 = 43y nebo 44y, vetsi
    #y3 = higher_43y_44y
    ##x4 = 45x
    #x4 = landmarks.part(number4).x
    ##y4 = 46y nebo 47y, mensi
    #y4 = lower_46y_47y
    #l_array_rec = np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], np.int32)
    #return l_array_rec


while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        # view face frame
        # x, y = face.left(), face.top()
        # x1, y1 = face.right(), face.bottom()
        # cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        # detect face structures using landmarks
        landmarks = predictor(gray, face)

        # crop eyes from the video
        left_landmarks_array = landmarks_array(36, 37, 38, 39, 40, 41)
        right_landmarks_array = landmarks_array(42, 43, 44, 45, 46, 47)

        # black mask with just eyes
        eye_fill = fill_frame(frame, left_landmarks_array, right_landmarks_array)
        cv2.imshow("eye_fill", eye_fill)

        # crop eyes from black rectangle
        left_eye_crop = crop_eyes(eye_fill, left_landmarks_array)
        right_eye_crop = crop_eyes(eye_fill, right_landmarks_array)

        # up size eyes
        left_eye_resize, left_threshold = resize_eye(left_eye_crop, left_landmarks_array)  # just for control view
        right_eye_resize, right_threshold = resize_eye(right_eye_crop, right_landmarks_array)  # just for control view

        # canny filter with preprocessing
        laplacian_left, edges_left = detect_pupil(left_eye_crop)
        laplacian_right, edges_right = detect_pupil(right_eye_crop)
        cv2.imshow("edges_left", edges_left)
        cv2.imshow("edges_right", edges_right)


        # finding eye center
        x_left, y_left = eye_center(left_eye_crop)  # center of left eye
        x_right, y_right = eye_center(right_eye_crop)  # center of right eye

        # View eye_center
        #left_center = cv2.circle(left_eye_resize, (y_left, x_left), 0, (0, 0, 255), 1)
        #cv2.imshow("left_center", left_center)
        #right_center = cv2.circle(right_eye_resize, (x_right, y_right), 0, (255, 0, 0), -1)
        #cv2.imshow("right_center", right_center)

        #find circles in eyes
        circles_left = detect_hough_circles(left_eye_crop)
        circles_right = detect_hough_circles(right_eye_crop)

        # draw points into eyes
        #for i in range(36, 48):
         #  draw_point(i)

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