import cv2
import dlib
import ctypes
import numpy as np
from dlib_landmarks import view_face_frame, draw_point, eye_center_dlib, landmarks_array, fill_frame, crop_eyes
from detect_pupil import converting_gray_to_hsv, filtration, gama_correction, preprocessing, contours_of_shape
from corneal_reflection import delete_corneal_reflection
from vector import find_vector
from calibration import upper_left, upper_right, middle_screen, lower_left, lower_right, middle_bottom, middle_left,\
    middle_right, middle_up, prepare_mask_for_calibration
from interpolate import interpolation
from eyetracking import find_closest_in_array, show_eyetracking, make_bgr_mask, normalize_array,\
    accuracy_from_eyetracking


#######################################################################################################################
# ------------------------------- Initiation part ------------------------------------------------------------------- #
#######################################################################################################################
detector_dlib = dlib.get_frontal_face_detector()
predictor_dlib = dlib.shape_predictor("Dlib_landmarks/shape_predictor_68_face_landmarks.dat")

# ----------------------- Parametres for Blob Detection ------------------------------------------------------------- #
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True  # activates filtering by area
detector_params.filterByCircularity = 1  # activates filtering by circularity
detector_params.minCircularity = 0.5  # min circularity (0.75)
detector_params.maxCircularity = 1  # max circularity
detector_params.maxArea = 5000  # max area (1800)
detector_blob = cv2.SimpleBlobDetector_create(detector_params)  # saving parametres into detector

# ----------------------- Get screen size --------------------------------------------------------------------------- #
user32 = ctypes.windll.user32
screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
interpolation_size = (96, 54)
#screensize = (120, 1280) #rpi


print("Set threshold for left and right eye.")
print("Press v to show calibrating vector.")
#######################################################################################################################
# ------------------------------- Creating trackbar ----------------------------------------------------------------- #
#######################################################################################################################
def nothing(x):
    '''
    creating Trackbar
    '''
    pass


#######################################################################################################################
# ---------------------------------- Main --------------------------------------------------------------------------- #
#######################################################################################################################
# ---------------------------------- Making video capture and video writers ----------------------------------------- #
def main():
    img = cv2.imread("portret.jpg", cv2.IMREAD_COLOR)  # reaching the port 0 for video capture

# ---------------------------------- Creating window for result and trackbars in it --------------------------------- #
    cv2.namedWindow('Dlib Landmarks')
    cv2.createTrackbar('Right', 'Dlib Landmarks', 0, 255, nothing)  # threshold track bar
    cv2.createTrackbar('Left', 'Dlib Landmarks', 0, 255, nothing)  # threshold track bar

# ---------------------------------- Initiation part ---------------------------------------------------------------- #
    #mask_for_eyetracking_bgr = make_bgr_mask(255, 255, 255, interpolation_size)
    mask_for_eyetracking = np.zeros((interpolation_size[1], interpolation_size[0]), np.uint8) + 255
    left_center_pupil_in_eye_frame = [0, 0]
    right_center_pupil_in_eye_frame = [0, 0]
    output_vector_in_eye_frame = [0, 0, 0, 0]
    left_eye_crop = [0, 0]
    right_eye_crop = [0, 0]
    min_left = [0, 0]
    min_right = [0, 0]
    send_calibration_data_state = True
    upper_left_corner = [0, 0, 0, 0]
    middle_right_corner = [0, 0, 0, 0]
    upper_right_corner = [0, 0, 0, 0]
    middle_left_corner = [0, 0, 0, 0]
    middle = [0, 0, 0, 0]
    middle_up_corner = [0, 0, 0, 0]
    lower_left_corner = [0, 0, 0, 0]
    middle_bottom_corner = [0, 0, 0, 0]
    lower_right_corner = [0, 0, 0, 0]
    size_of_interpolated_map = 100
    u_interp = np.zeros((size_of_interpolated_map, size_of_interpolated_map), np.uint8)
    v_interp = np.zeros((size_of_interpolated_map, size_of_interpolated_map), np.uint8)
    press_v = False
    press_p = True
    press_1 = True
    press_2 = True
    press_3 = True
    press_4 = True
    press_5 = True
    press_6 = True
    press_7 = True
    press_8 = True
    press_9 = True
    press_detele = True
    press_s = True
    press_e = False
    k = 0
    a = 1
    b = 0
# ---------------------------------- Get the video frame and prepare it for detection ------------------------------- #
    if a == 1:  # while th video capture is
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # change color from rgb to gray
        # gray = gray[::4, ::4] #rpi

# ---------------------------------- Dlib Landmark face detection --------------------------------------------------- #
        faces = detector_dlib(gray)
        for face in faces:
           # view_face_frame(face, frame)  # view face frame
            landmarks = predictor_dlib(gray, face)  # detect face structures using landmarks

            # crop eyes from the video
            left_landmarks_array = landmarks_array(36, 37, 38, 39, 40, 41, landmarks, gray, lines=0)
            right_landmarks_array = landmarks_array(42, 43, 44, 45, 46, 47, landmarks, gray, lines=0)

            eye_fill = fill_frame(gray, left_landmarks_array, right_landmarks_array)  # black mask with just eyes

            # crop eyes from black rectangle
            left_eye_crop, min_left = crop_eyes(eye_fill, left_landmarks_array)
            right_eye_crop, min_right = crop_eyes(eye_fill, right_landmarks_array)

            # draw points into eyes
            for i in range(36, 48):
                draw_point(i, landmarks, img)

             #draw points into face
            #for i in range(0, 67):
             #  draw_point(i, landmarks, img)

# ---------------------------------- Left eye ---------------------------------------------------------------------- #
            threshold_left = 0
            no_reflex_left = delete_corneal_reflection(left_eye_crop, threshold_left)  # deleting corneal reflex
            hsv_img_left = converting_gray_to_hsv(no_reflex_left)  # converting frame to hsv
            filtrated_img_left = filtration(hsv_img_left)  # applying some filtration
            gama_corrected_left = gama_correction(filtrated_img_left, 1.2)  # gama correction
            eye_preprocessed_left = preprocessing(gama_corrected_left, threshold_left)  # morfological operations
            cv2.imshow("eye_processed_left", eye_preprocessed_left)
            contours_left = contours_of_shape(eye_preprocessed_left, threshold_left)  # get contours
            if contours_left is not None:
                for c_left in contours_left:
                    if c_left is not None:
                        m_left = cv2.moments(c_left)
                        if m_left["m00"] > 1:
                            cx_left = int(m_left["m10"] / m_left["m00"])  # x coordinate for centroid of blob
                            cy_left = int(m_left["m01"] / m_left["m00"])  # y coordinate for centroid of blob
                            # cv2.drawContours(eye_no_eyebrows_left, [c], -1, (0, 255, 0), 2)
                            # cv2.circle(left_eye_crop, (cx_left, cy_left), 1, (0, 0, 255), 2)
                            # position of left pupil in whole frame
                            left_center_pupil = [cx_left + min_left[0], cy_left + min_left[1]]
                            # position of left pupil in eye frame
                            left_center_pupil_in_eye_frame = [cx_left, cy_left]
                            # show pupil
                            cv2.circle(img, (left_center_pupil[0], left_center_pupil[1]), 1, (255, 0, 0), 2)

# ---------------------------------- Right eye ---------------------------------------------------------------------- #
            threshold_right = 0
            no_reflex_right = delete_corneal_reflection(right_eye_crop, threshold_right)  # deleting corneal reflex
            hsv_img_right = converting_gray_to_hsv(no_reflex_right)  # converting frame to hsv
            filtrated_img_right = filtration(hsv_img_right)  # applying some filtration
            gama_corrected_right = gama_correction(filtrated_img_right, 1.2)  # gama correction
            eye_preprocessed_right = preprocessing(gama_corrected_right, threshold_right)  # morfological operations
            cv2.imshow("preprocessing_right", eye_preprocessed_right)
            contours_right = contours_of_shape(eye_preprocessed_right, threshold_right)  # get contours
            if contours_right is not None:
                for c_right in contours_right:
                    if c_right is not None:
                        m_right = cv2.moments(c_right)
                        if m_right["m00"] > 1:
                            cx_right = int(m_right["m10"] / m_right["m00"])  # x coordinate for middle of blob
                            cy_right = int(m_right["m01"] / m_right["m00"])  # y coordinate for middle of blob
                            # cv2.drawContours(eye_no_eyebrows_right, [c], -1, (0, 255, 0), 2)
                            # cv2.circle(right_eye_crop, (cx_right, cy_right), 1, (0, 0, 255), 2)
                            # position of right pupil in whole frame
                            right_center_pupil = [cx_right + min_right[0], cy_right + min_right[1]]
                            # position of right pupil in eye frame
                            right_center_pupil_in_eye_frame = [cx_right, cy_right]
                            # show pupil
                            cv2.circle(img, (right_center_pupil[0], right_center_pupil[1]), 1, (255, 0, 0), 2)
                            b = 0

# ---------------------------------- Show vector after pressing v --------------------------------------------------- #
        if b == 1 and not press_v:
            press_v = True
            print("Vector mode activated.")
            print('For starting calibration mode press p.')

        if press_v:
            # finding eye center
            left_center_eye, left_center_eye_in_eye_frame = eye_center_dlib(left_eye_crop, [min_left[0], min_left[1]])
            right_center_eye, right_center_eye_in_eye_frame = eye_center_dlib(right_eye_crop, [min_right[0],
                                                                                               min_right[1]])

            # finding vector
            output_vector_in_eye_frame = find_vector(left_center_pupil_in_eye_frame, left_center_eye_in_eye_frame,
                                                     right_center_pupil_in_eye_frame, right_center_eye_in_eye_frame)

            # start of vector
            start_left = (left_center_eye[0], left_center_eye[1])
            start_right = (right_center_eye[0], right_center_eye[1])

            # end of vector
            end_left = (int(output_vector_in_eye_frame[0]*10) + left_center_eye[0],
                        int(output_vector_in_eye_frame[1]*10) + left_center_eye[1])
            end_right = (int(output_vector_in_eye_frame[0]*10) + right_center_eye[0],
                         int(output_vector_in_eye_frame[1]*10) + right_center_eye[1])

            # show vector in frame
            if end_left == [0, 0] or end_right == [0, 0]:
                print("Pupil not detected. Try to adjust threshold better and press v again..")
            if end_left is not None and end_right is not None:
                if output_vector_in_eye_frame[2] > 0:
                    cv2.arrowedLine(img, start_left, end_left, color=(0, 255, 0), thickness=1)
                    cv2.arrowedLine(img, start_right, end_right, color=(0, 255, 0), thickness=1)

            press_p = False

# ---------------------------------- Show result and keyboard check ------------------------------------------------- #
        cv2.imshow('Dlib Landmarks', img)  # visualization of detection
        cv2.imwrite("landmarks_eyes.jpg", img)
        cv2.imwrite("binarni_pred.jpg", eye_preprocessed_right)
        k = cv2.waitKey(100000) & 0xFF  # get key that is pressed on keyboard

# ---------------------------------- Quit program after pressing q -------------------------------------------------- #
        if k == ord('q'):
            print(a)
    cv2.destroyAllWindows()  # close all windows


if __name__ == "__main__":
    main()
