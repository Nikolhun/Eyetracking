import numpy as np
import cv2
import dlib
import imutils
import math

#######################################################################################################################
# ------------------------------- Initiation part ------------------------------------------------------------------- #
#######################################################################################################################
detector_dlib = dlib.get_frontal_face_detector()
predictor_dlib = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# ----------------------- Parametres for Blob Detection ------------------------------------------------------------- #
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True  # activates filtering by area
detector_params.filterByCircularity = 1  # activates filtering by circularity
detector_params.minCircularity = 0.5  # min circularity (0.75)
detector_params.maxCircularity = 1  # max circularity
detector_params.maxArea = 5000  # max area (1800)
detector_blob = cv2.SimpleBlobDetector_create(detector_params)  # saving parametres into detector


#######################################################################################################################
# ------------------------------- Functions ------------------------------------------------------------------------- #
#######################################################################################################################

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


# ------------------------------- Blob process --------------------------------------------------------------------- #
def converting_gray_to_hsv(img):
    '''
    Convert gray frame into hsv frame.
    :param img: image in gray frame you want to convert
    :return: hsv frame
    '''
    color_frame_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # converting from rgb to hsv
    hsv_img = cv2.cvtColor(color_frame_rgb, cv2.COLOR_RGB2HSV)  # converting from rgb to hsv
    return hsv_img


def filtration(img):
    '''
    Bilateral filtration, median filtration, gaussian filtration.
    :param img: image in gray frame
    :return: filtrated image in gray frame
    '''
    img_bilateral = cv2.bilateralFilter(img, 9, 75, 75)  # bilateral filtration
    img_median = cv2.medianBlur(img_bilateral, 5)  # median filtration
    filtrated_img = cv2.GaussianBlur(img_median, (5, 5), 0)  # Gaussian filtration
    return filtrated_img


def gama_correction(img, gamma):
    '''
    Function makes gamma correction acording to picture 'img' according to the parameter 'gamma'.
    :param img: picture in GRAY
    :param gamma: gamma parameter, best it's 1.2
    :return: gamma corrected picture in GRAY
    '''
    # Apply gamma correction.
    gamma_corrected = np.array(255 * (img / 255) ** gamma, dtype='uint8')
    # Show edited images.
    #cv2.imshow('gamma_transformed' + str(gamma), gamma_corrected)
    return gamma_corrected


def preprocessing(img, threshold):
    '''
    Preprocessing picture: erode, dilate, morphology opening, morphology closing, bilateral filtration,
    median filtration, gaussian filtration.
    :param img: image in gray frame or hsv frame
    :param threshold: threshold for thresholding image into binary image
    :return: preprocessed_image, threshold
    '''
    _, img_binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)  # gray to binary image with threshold
    img_erode = cv2.erode(img_binary, None, iterations=2)  # erode picture
    img_dilate = cv2.dilate(img_erode, None, iterations=4)  # dilate picture
    img_open = cv2.morphologyEx(img_dilate, cv2.MORPH_OPEN, None)  # morphology opening
    preprocessed_img = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, None)  # morphology closing

    return preprocessed_img


def contours_of_shape(img, threshold):
    '''
    Finds contours of shape.
    :param img: HSV image
    :param threshold: threshold from trackbar
    :return: file of contours of shape
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(contours)
    return cnts

def blob_process(img, detector_blob):
    '''
    Function for blob process
    :param img: image in gray
    :param detector_blob: detector iniciated in iniciation part
    :return: kepoints of blob
    '''
    keypoints = detector_blob.detect(img)  # getting keypoints of eye
    return keypoints


# ------------------------------- Korneální reflexe ----------------------------------------------------------------- #
def corneal_reflection(img, threshold_from_trackbar):
    '''
    It removes conrneal reflection.
    :param img: image
    :param threshold: threshold from trackbar in image window
    :return: image withou corneal reflection
    '''
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    _, threshold = cv2.threshold(l, 210, 255, cv2.THRESH_BINARY)
    threshold = cv2.dilate(threshold, None, iterations=1)  # dilate picture
    #open = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, None)  # morphology opening
    #threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, None)  # morphology closing
    row = threshold.shape[0]
    coll = threshold.shape[1]
    #contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.imshow("threshold", threshold)
    #cv2.imshow("corneal reflection", gray_frame)
    #cv2.drawContours(gray_frame, contours, -1, (0, 0, 255), 1)
    #mask = np.zeros(3, 3)
    for i in range(0, row):
        for y in range(0, coll):
            if threshold[i, y] == 0:
                l[i, y] = l[i, y]
            elif threshold[i, y] == 255:
                if i == 0 and y == 0:
                    l[i, y] = threshold_from_trackbar/2
                elif i == 0 and y != 0:
                    l[i, y] = l[i, y - 1]
                elif y == 0 and i != 0:
                    l[i, y] = l[i - 1, y]
                else:
                    l[i, y] = l[i - 1, y]

                #if ((i != 0 or i != 1) and (y != 0 or y != 1)) and \
                #        ((i != (row-1) or i != row)) and (y != (coll-1) or y != coll):
                #    #mask [i-1:i+1, y-1:y+1] = l[i-1:i+1, y-1:y+1]
                 #   mask = np.array([[[i-1, y-1], [i-1, y], [i-1, y+1]],
                 #                    [[i, y-1], [i, y], [i, y+1]],
                 #                    [[i+1, y-1], [i+1, y], [i+1, y+1]]])
                #    mean_mask = mask.mean()
                 #   print(mean_mask)
                 #   print(mask)
                #    l[i, y] = mean_mask
    lab = cv2.merge([l, a, b])
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    gray_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return gray_frame


# ------------------------------- Detekce směru pohledu ------------------------------------------------------------- #
def vector(left_center_pupil, left_center_eye, right_center_pupil, right_center_eye):
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


# ------------------------------- Trackbar -------------------------------------------------------------------------- #
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
    cv2.namedWindow('Dlib Landmarks')  # Dlib landmark left eye
    cv2.createTrackbar('Left', 'Dlib Landmarks', 0, 255, nothing)  # threshold track bar
    cv2.createTrackbar('Right', 'Dlib Landmarks', 0, 255, nothing)  # threshold track bar
    while cap.isOpened():  # while th video capture is
        _, frame = cap.read()  # convert cap to matrix for future work
        frame = cv2.flip(frame, 1)  # flip video to not be mirrored
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# ---------------------------------- Dlib Landmark ------------------------------------------------------------------ #
        faces = detector_dlib(gray)
        for face in faces:
            #view_face_frame(face, frame)  # view face frame
            landmarks = predictor_dlib(gray, face)  # detect face structures using landmarks

            # crop eyes from the video
            left_landmarks_array = landmarks_array(36, 37, 38, 39, 40, 41, landmarks, gray, lines=0)
            right_landmarks_array = landmarks_array(42, 43, 44, 45, 46, 47, landmarks, gray, lines=0)

            eye_fill = fill_frame(gray, left_landmarks_array, right_landmarks_array)  # black mask with just eyes
            #cv2.imshow("eye_fill", eye_fill)

            # crop eyes from black rectangle
            left_eye_crop, min_left = crop_eyes(eye_fill, left_landmarks_array)
            right_eye_crop, min_right = crop_eyes(eye_fill, right_landmarks_array)

            # draw points into eyes
            for i in range(36, 48):
                draw_point(i, landmarks, frame)

            # draw points into face
            # for i in range(0, 67):
            #   draw_point(i, landmarks, frame)

            # left eye
            threshold_left = cv2.getTrackbarPos('Right', 'Dlib Landmarks')  # getting position of the trackbar
            no_reflex_left = corneal_reflection(left_eye_crop, threshold_left)
            hsv_img_left = converting_gray_to_hsv(no_reflex_left)
            filtrated_img_left = filtration(hsv_img_left)
            gama_corrected_left = gama_correction(filtrated_img_left, 1.2)
            eye_preprocessed_left = preprocessing(gama_corrected_left, threshold_left, method)
            cv2.imshow("eye_processed_left", eye_preprocessed_left)
            contours_left = contours_of_shape(eye_preprocessed_left, threshold_left)
            if contours_left is not None:
                for c_left in contours_left:
                    if c_left is not None:
                        m_left = cv2.moments(c_left)
                        if m_left["m00"] > 1:
                            cx_left = int(m_left["m10"] / m_left["m00"])
                            cy_left = int(m_left["m01"] / m_left["m00"])
                            # cv2.drawContours(eye_no_eyebrows_left, [c], -1, (0, 255, 0), 2)
                            #cv2.circle(left_eye_crop, (cx_left, cy_left), 1, (0, 0, 255), 2)
                            left_center_pupil = [cx_left + min_left[0], cy_left + min_left[1]]
                            left_center_pupil_in_eye_frame = [cx_left, cy_left]
                            cv2.circle(frame, (left_center_pupil[0], left_center_pupil[1]), 1, (255, 0, 0), 2)
            # right eye
            threshold_right = cv2.getTrackbarPos('Left', 'Dlib Landmarks')  # getting position of the trackbar
            no_reflex_right = corneal_reflection(right_eye_crop, threshold_right)
            hsv_img_right = converting_gray_to_hsv(no_reflex_right)
            filtrated_img_right = filtration(hsv_img_right)
            gama_corrected_right = gama_correction(filtrated_img_right, 1.2)
            eye_preprocessed_right = preprocessing(gama_corrected_right, threshold_right)
            cv2.imshow("preprocessing_right", eye_preprocessed_right)
            contours_right = contours_of_shape(eye_preprocessed_right, threshold_right)
            if contours_right is not None:
                for c_right in contours_right:
                    if c_right is not None:
                        m_right = cv2.moments(c_right)
                        if m_right["m00"] > 1:
                            cx_right = int(m_right["m10"] / m_right["m00"])
                            cy_right = int(m_right["m01"] / m_right["m00"])

                            #cv2.drawContours(eye_no_eyebrows_right, [c], -1, (0, 255, 0), 2)
                            #cv2.circle(right_eye_crop, (cx_right, cy_right), 1, (0, 0, 255), 2)
                            #cv2.imshow("right", right_eye_crop)

                            right_center_pupil = [cx_right + min_right[0], cy_right + min_right[1]]
                            right_center_pupil_in_eye_frame = [cx_right, cy_right]
                            cv2.circle(frame, (right_center_pupil[0], right_center_pupil[1]), 1,
                                               (255, 0, 0), 2)

                            # finding eye center
                            left_center_eye, left_center_eye_in_eye_frame = eye_center_dlib(left_eye_crop,
                                                                                                    [min_left[0],
                                                                                                     min_left[1]])
                            right_center_eye, right_center_eye_in_eye_frame = eye_center_dlib(right_eye_crop,
                                                                                                      [min_right[0],
                                                                                                       min_right[1]])

                            # finding vector
                            output_vector_in_eye_frame = vector(left_center_pupil_in_eye_frame,
                                                                    left_center_eye_in_eye_frame,
                                                                    right_center_pupil_in_eye_frame,
                                                                    right_center_eye_in_eye_frame)

                            # start of vector
                            start_left = (left_center_eye[0], left_center_eye[1])
                            start_right = (right_center_eye[0], right_center_eye[1])

                            # end of vector
                            end_left = (output_vector_in_eye_frame[0] + left_center_eye[0],
                                                   output_vector_in_eye_frame[1] + left_center_eye[1])
                            end_right = (output_vector_in_eye_frame[0] + right_center_eye[0],
                                                   output_vector_in_eye_frame[1] + right_center_eye[1])

                            if end_left == [0, 0] or end_right == [0, 0]:
                                print("Pupil not detected, set threshold in left/right trackbar.")
                            else:
                                if end_left is not None and end_right is not None:
                                    if output_vector_in_eye_frame[2] > 1:
                                        cv2.arrowedLine(frame, start_left, end_left,
                                                             color=(0, 255, 0), thickness=1)
                                        cv2.arrowedLine(frame, start_right, end_right,
                                                             color=(0, 255, 0), thickness=1)

        cv2.imshow('Dlib Landmarks', frame)  # visualization of detection

        if cv2.waitKey(1) & 0xFF == ord('q'):  # "q" means close the detection
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()