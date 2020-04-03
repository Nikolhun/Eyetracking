import numpy as np
import cv2
import dlib
from scipy import signal

#######################################################################################################################
# ------------------------------- Choosing detection method --------------------------------------------------------- #
#######################################################################################################################
# Choose your method and detection method:
#method = 0  # Haar cascade
method = 1  # Landmark dlib
detection_method = 0  # Blob detection
#detection_method = 1  # Hough circle
#color = 0  # gray frame
color = 1  # hsv frame

if method == 0:
    print("Haar cascade was chosen.")
elif method == 1:
    print("Dlib Landmark method was chosen.")

if detection_method == 0:
    print("Blob detection method was chosen.")
elif detection_method == 1:
    print("Hough circle detection method was chosen.")

#######################################################################################################################
# ------------------------------- Initiation part ------------------------------------------------------------------- #
#######################################################################################################################

# ----------------------------- Haar features ----------------------------------------------------------------------- #
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # haar cascade classifiers for face
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')  # haar cascade classifiers for eye
#nose_cascade = cv2.CascadeClassifier('haarcascade_nose.xml')  # hhar cascade classifier for nose
#if nose_cascade.empty():
  #raise IOError('Unable to load the nose cascade classifier xml file')

#  ------------------------------Landmark dlib ---------------------------------------------------------------------- #
detector_dlib = dlib.get_frontal_face_detector()
predictor_dlib = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# ----------------------- Parametres for Blob Detection ------------------------------------------------------------- #
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True  # activates filtering by area
detector_params.filterByCircularity = 1  # activates filtering by circularity
detector_params.minCircularity = 0.5  # min circularity (0.75)
detector_params.maxCircularity = 1  # max circularity
detector_params.maxArea = 5000 # max area (1800)
detector_blob = cv2.SimpleBlobDetector_create(detector_params)  # saving parametres into detector

#######################################################################################################################
# ------------------------------- Functions ------------------------------------------------------------------------- #
#######################################################################################################################

# ------------------------------- Haar detection--------------------------------------------------------------------- #
def detect_faces(img, cascade):
    # detects face from frame
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converting from rgb to gray
    coords = cascade.detectMultiScale(gray_frame, 1.3, 5)  # getting coordinated of face
    if len(coords) > 1:  # getting the biggest face
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None
    for (x, y, w, h) in biggest:
        frame = img[y:y + h, x:x + w]  # crop the image
    # cv2.imshow('detect_faces', img)  # show face detection
    return frame

#def detect_nose(img, cascade):
 #   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  #  coords = cascade.detectMultiScale(gray, 1.3, 5)
   # for (x, y, w, h) in coords:
    #    frame = img[y:y + h, x:x + w]
     #   cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
    #cv2.imshow('detect_nose', img)
    #return frame, x, y

def detect_eyes(img, cascade):
    # detects eyes from face
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converting from rgb to gray
    #gray_frame_2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # converting from rgb to hsv
    eyes = cascade.detectMultiScale(gray_frame, 1.3, 5)  # getting coordinates of eyes
    width = np.size(img, 1)  # get face frame width
    height = np.size(img, 0)  # get face frame height
    left_eye = None
    right_eye = None
    for (x, y, w, h) in eyes:
        hh = h / 2  # getting half of the picture
        if y + hh < height / 2:  # eyes are in the upper part
            eyecenter = x + w / 2  # get the eye center
            if eyecenter < width * 0.5:  # right eye is on the left side of the face
                right_eye = img[y:y + h, x:x + w]
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:  # left eye is on the right side of the face
                left_eye = img[y:y + h, x:x + w]
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #cv2.imshow('detect_eyes', img)  # show eye detection
    return left_eye, right_eye

def cut_eyebrows(img):
    # cuts eyebrows from eyes
    height, width = img.shape[:2]  # get height and width of the picture
    eyebrow_h = int(height / 4) # get height of eyebrows
    img = img[eyebrow_h:height, 0:width]  # cut eyebrows
    return img

# ------------------------------- Dlib Landmarks -------------------------------------------------------------------- #
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
    height, width, _ = eye.shape
    x = int(height/2)
    y = int(width/2)
    return x, y

def landmarks_array(number1, number2, number3, number4, number5, number6, landmarks, frame, lines):
    # lines = 0 -> dont draw lines
    # lines = 1 -> draw lines

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

def fill_frame (frame, left_array, right_array):
    height, width, _ = frame.shape
    mask = np.ones((height, width), np.uint8)
    #mask = np.zeros((height, width), np.uint8)
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
    eye_crop = eye_fill[min_y:max_y, min_x: max_x]
    #cv2.imshow("eye_crop", eye_crop)
    return eye_crop

def resize_eye (eye_crop, l_array, frame):
    _, threshold_eye = cv2.threshold(eye_crop, 70, 255, cv2.THRESH_BINARY)
    threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
    resized_eye = cv2.resize(eye_crop, None, fx=5, fy=5)
    #cv2.imshow("threshold_eye", threshold_eye)
    #cv2.imshow("eye_crop", eye_crop)
    #cv2.imshow("eye_fill", eye_fill)
    cv2.polylines(frame, [l_array], True, (0, 0, 0), 1)
    return resized_eye, threshold_eye

def edges(eye_crop, method):
    # method = 0 -> Canny
    # method = 1 -> Laplatian
    eye_crop = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
    #thres = cv2.inRange(eye_crop, 0, 20)
    eye_crop = cv2.equalizeHist(eye_crop)
    eye_crop = cv2.bilateralFilter(eye_crop, 9, 75, 75)
    kernel = np.ones((3, 3), np.uint8)
    # /------- removing small noise inside the white image ---------/#
    dilation = cv2.dilate(eye_crop, kernel, iterations=1)
    # /------- decreasing the size of the white region -------------/#
    erosion = cv2.erode(dilation, kernel, iterations=1)
    if method == 0:
        # Canny
        edges = cv2.Canny(erosion, 20, 40)
        # cv2.imshow("edges", edges)
    elif method == 1:
        # Laplacian
        edges = cv2.Laplacian(erosion, cv2.CV_16U, ksize=3)
        # cv2.imshow("Laplacian", laplacian)
    return edges

# ------------------------------- Hough circles --------------------------------------------------------------------- #

def detect_hough_circles(edges):
    gray = cv2.cvtColor(edges, cv2.COLOR_BGR2HSV)
    gray_shape = gray.shape
    # Process the image for circles using the Hough transform
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, gray_shape[1], 50, 30, 1, 2)
    #circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2, gray_shape[1]*2, param2=150)
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
        #print(circles)
        for pt in circles[0, :]:
            a, b, r, = pt[0], pt[1], pt[2]
            # Draw the circumference of the circle.
            cv2.circle(edges, (a, b), r, (0, 255, 0), 3)
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(edges, (a, b), 1, (255, 0, 0), 3)
    cv2.imshow('detected circles', edges)
    return circles
# ------------------------------- Blob process--------------------------------------------------------------------- #
def preprocessing (img, threshold, method, color):
    '''
    color 0 -> gray frame
    color 1 -> hsv frame
    '''
    color_frame = 0
    # detects pupil from eyes
    if color == 0:
        color_frame = cv2.cvtColor(img, cv2.COLOR_GRAY2GRAY)  # converting from rgb to gray
    elif color == 1:
        color_frame = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # converting from rgb to hsv
        color_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2HSV)  # converting from rgb to hsv

    _, img = cv2.threshold(color_frame, threshold, 255, cv2.THRESH_BINARY_INV)  # gray to binary image with threshold
    if method == 0:  # Haar cascade
        img = cv2.erode(img, None, iterations=2)  # erode picture
        img = cv2.dilate(img, None, iterations=4)  # dilate picture
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, None)  # morphology opening
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, None)  # morphology closing
        img = cv2.bilateralFilter(img, 9, 75, 75)  # bilateral filtration
        img = cv2.medianBlur(img, 5)  # median filtration
        img = cv2.GaussianBlur(img, (5, 5), 0)  # Gaussian filtration
        cv2.imshow('blob_after', img)  # show after blob process
    if method == 1:  # Dlib Landmark
        # opravit ještě podle nejlepšího
        img = cv2.erode(img, None, iterations=2)  # erode picture
        img = cv2.dilate(img, None, iterations=4)  # dilate picture
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, None)  # morphology opening
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, None)  # morphology closing
        img = cv2.bilateralFilter(img, 9, 75, 75)  # bilateral filtration
        img = cv2.medianBlur(img, 5)  # median filtration
        img = cv2.GaussianBlur(img, (5, 5), 0)  # Gaussian filtration
        #cv2.imshow('blob_after', img)  # show after blob process
    return img

def blob_process(img, detector_blob):
    keypoints = detector_blob.detect(img)  # getting keypoints of eye
    return keypoints

# ------------------------------- Korneální reflexe ----------------------------------------------------------------- #
def corneal_reflex (frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, _, _ = cv2.split(lab)
    _, threshold = cv2.threshold(l, 220, 255, cv2.THRESH_BINARY)
    dilate = cv2.dilate(threshold, None, iterations=1)  # dilate picture
    open = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, None)  # morphology opening
    close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, None)  # morphology closing
    row = close.shape[0]
    coll = close.shape[1]
    output = gray_frame
    for i in range(0, row):
        for y in range(0, coll):
            if close[i, y] == 0:
                output[i, y] = output[i, y]
            elif close[i, y] == 255:
                output[i, y] = 128
    #cv2.imshow("corneal reflection is deleted", output)
    #contours, hierarchy = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(fr, contours, -1, (0, 255, 0), 3)
    #cv2.imshow("1", fr)
    return output


def nothing(x):
    '''
    creating Trackbar
    '''
    pass


#######################################################################################################################
# ---------------------------------- Main --------------------------------------------------------------------------- #
#######################################################################################################################
def main():
    cap = cv2.VideoCapture(0)  # reaching the port 0 for video capture
    if method == 0:  # Haar cascade
        #if detection_method == 0:  # Blob detection
        cv2.namedWindow('image')  # Haar cascade
        cv2.createTrackbar('Left', 'image', 0, 255, nothing)  # threshold track bar
        cv2.createTrackbar('Right', 'image', 0, 255, nothing)  # threshold track bar
    elif method == 1:  # Dlib Landmarks
        cv2.namedWindow('Dlib Landmarks')  # Dlib landmark left eye
        #if detection_method == 0:  # Blob detection
        cv2.createTrackbar('Left', 'Dlib Landmarks', 0, 255, nothing)  # threshold track bar
        cv2.createTrackbar('Right', 'Dlib Landmarks', 0, 255, nothing)  # threshold track bar

    while cap.isOpened():  # while th video capture is
        _, frame = cap.read()  # convert cap to matrix for future work

        frame = cv2.flip(frame, 1)  # flip video to not be mirrored
# ---------------------------------- Haar cascade ------------------------------------------------------------------- #
        if method == 0:  # if you choose Haar cascade
            face_frame = detect_faces(frame, face_cascade)  # function for face detection
            if face_frame is not None:  # if the face is detected
                #print("Face detected succesfully.")
                left_eye, right_eye = detect_eyes(face_frame, eye_cascade)  # function for eye detection
                if left_eye is not None:  # for left eye
                    if detection_method == 0:  # Blob
                        threshold_left = cv2.getTrackbarPos('Left', 'image')
                                                # getting position of the trackbar
                        eye_no_eyebrows_left = cut_eyebrows(left_eye)  # cutting eyebrows
                        no_reflex_left = corneal_reflex(eye_no_eyebrows_left)
                        cv2.imshow("left",no_reflex_left)
                        eye_preprocesed_left = preprocessing(no_reflex_left, threshold_left, method, color)
                        keypoints_left = blob_process(eye_preprocesed_left, detector_blob)  # detecting pupil
                        detected_eye_left = cv2.drawKeypoints(no_reflex_left, keypoints_left,
                                                         no_reflex_left, (0, 0, 255),
                                                         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                                                        # drawig red circle into image
                    elif detection_method == 1:
                        cirlces_left = detect_hough_circles(left_eye)
                        #DODĚLAT
                        #cv2.imshow('image', frame)  # visualization of detection

                if right_eye is not None:  # for right eye
                    if detection_method == 0:  # blob
                        threshold_right = cv2.getTrackbarPos('Right', 'image')
                                                # getting position of the trackbar
                        eye_no_eyebrows_right = cut_eyebrows(right_eye)  # cutting eyebrows
                        no_reflex_right = corneal_reflex(eye_no_eyebrows_right)
                        cv2.imshow("right", no_reflex_right)
                        eye_preprocesed_right = preprocessing(no_reflex_right, threshold_right, method, color)
                        keypoints_right = blob_process(eye_preprocesed_right, detector_blob)  # detecting pupil
                        detected_eye_right = cv2.drawKeypoints(no_reflex_right, keypoints_right,
                                                         no_reflex_right, (0, 0, 255),
                                                         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                                                        # drawig red circle into image
                    elif detection_method == 1:  # Hough circles
                        cirlces_right = detect_hough_circles(left_eye)
                        #DODĚLAT
                        #cv2.imshow('image', frame)  # visualization of detection
# ---------------------------------- Dlib Landmark ------------------------------------------------------------------ #
        elif method == 1:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            faces = detector_dlib(gray)
            for face in faces:
                #view_face_frame(face, frame)  # view face frame
                landmarks = predictor_dlib(gray, face)  # detect face structures using landmarks

                # crop eyes from the video
                left_landmarks_array = landmarks_array(36, 37, 38, 39, 40, 41, landmarks, frame, lines=0)
                right_landmarks_array = landmarks_array(42, 43, 44, 45, 46, 47, landmarks, frame, lines=0)

                eye_fill = fill_frame(frame, left_landmarks_array, right_landmarks_array)  # black mask with just eyes
                #cv2.imshow("eye_fill", eye_fill)

                # crop eyes from black rectangle
                left_eye_crop = crop_eyes(eye_fill, left_landmarks_array)
                right_eye_crop = crop_eyes(eye_fill, right_landmarks_array)
                #cv2.imshow("left_eye_crop", left_eye_crop)
                #cv2.imshow("right_eye_crop", right_eye_crop)

                #removing corneal reflex
                no_reflex_left = corneal_reflex(left_eye_crop)
                no_reflex_right = corneal_reflex(right_eye_crop)

                # View eye_center
                # left_center = cv2.circle(left_eye_resize, (y_left, x_left), 0, (0, 0, 255), 1)
                # cv2.imshow("left_center", left_center)
                # right_center = cv2.circle(right_eye_resize, (x_right, y_right), 0, (255, 0, 0), -1)
                # cv2.imshow("right_center", right_center)
                if detection_method == 0: #  blob detection
                    # left eye
                    threshold_left = cv2.getTrackbarPos('Left', 'Dlib Landmarks')  # getting position of the trackbar
                    no_reflex_left = preprocessing(no_reflex_left, threshold_left, method, color)
                    keypoints_left = blob_process(no_reflex_left, detector_blob)  # detecting pupil
                    eye_left = cv2.drawKeypoints(no_reflex_left, keypoints_left, no_reflex_left, (0, 0, 255),
                                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  # drawig red circle into image
                    # right eye
                    threshold_right = cv2.getTrackbarPos('Right', 'Dlib Landmarks')  # getting position of the trackbar
                    no_reflex_right = preprocessing(no_reflex_right, threshold_right, method, color)
                    keypoints_right = blob_process(no_reflex_right, detector_blob)  # detecting pupil
                    eye_right = cv2.drawKeypoints(no_reflex_right, keypoints_right, no_reflex_right, (0, 0, 255),
                                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  # drawig red circle into image
                if detection_method == 1:  # hough circles
                    # find circles in eyes
                    circles_left = detect_hough_circles(no_reflex_left)
                    circles_right = detect_hough_circles(no_reflex_right)

                # draw points into eyes
                #for i in range(36, 48):
                    #draw_point(i, landmarks, frame)

                # draw points into face
                for i in range(0, 67):
                    draw_point(i, landmarks, frame)




                # Laplacian and filter with preprocessing, 0 = canny, 1 = Laplacian
                edges_left = edges(no_reflex_left, 0)
                edges_right = edges(no_reflex_right, 0)
                # cv2.imshow("edges_left", edges_left)
                # cv2.imshow("edges_right", edges_right)

                # up size eyes
                left_eye_resize, left_threshold = resize_eye(no_reflex_left,
                                                             left_landmarks_array, frame)  # just for control view
                right_eye_resize, right_threshold = resize_eye(no_reflex_right,
                                                               right_landmarks_array, frame)  # just for control view

                # finding eye center
                x_left, y_left = eye_center(no_reflex_left)  # center of left eye
                x_right, y_right = eye_center(no_reflex_right)  # center of right eye


        if method == 0:
            cv2.imshow('image', frame)  # visualization of detection
        elif method == 1:
            cv2.imshow('Dlib Landmarks', frame)  # visualization of detection
            #cv2.imshow('right_eye', frame)  # visualization of detection

        if cv2.waitKey(1) & 0xFF == ord('q'):  # "q" means close the detection
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()




