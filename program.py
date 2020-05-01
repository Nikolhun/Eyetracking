import cv2
import dlib
import ctypes
import keyboard
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
    cap = cv2.VideoCapture(0)  # reaching the port 0 for video capture
    fourcc_detection = cv2.VideoWriter_fourcc(*'XVID')
    out_detection = cv2.VideoWriter('detection.mkv', fourcc_detection, 20.0, (int(cap.get(3)), int(cap.get(4))))
    fourcc_mask = cv2.VideoWriter_fourcc(*'XVID')
    out_mask = cv2.VideoWriter('mask.mkv', fourcc_mask, 20.0, interpolation_size)

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

# ---------------------------------- Get the video frame and prepare it for detection ------------------------------- #
    while cap.isOpened():  # while th video capture is
        _, frame = cap.read()  # convert cap to matrix for future work
        frame = cv2.flip(frame, 1)  # flip video to not be mirrored
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # change color from rgb to gray

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
                draw_point(i, landmarks, frame)

            # draw points into face
            # for i in range(0, 67):
            #   draw_point(i, landmarks, frame)

# ---------------------------------- Left eye ---------------------------------------------------------------------- #
            threshold_left = cv2.getTrackbarPos('Right', 'Dlib Landmarks')  # getting position of the trackbar
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
                            cx_left = int(m_left["m10"] / m_left["m00"])  # x coordinate for middle of blob
                            cy_left = int(m_left["m01"] / m_left["m00"])  # y coordinate for middle of blob
                            # cv2.drawContours(eye_no_eyebrows_left, [c], -1, (0, 255, 0), 2)
                            # cv2.circle(left_eye_crop, (cx_left, cy_left), 1, (0, 0, 255), 2)
                            # position of left pupil in whole frame
                            left_center_pupil = [cx_left + min_left[0], cy_left + min_left[1]]
                            # position of left pupil in eye frame
                            left_center_pupil_in_eye_frame = [cx_left, cy_left]
                            # show pupil
                            cv2.circle(frame, (left_center_pupil[0], left_center_pupil[1]), 1, (255, 0, 0), 2)

# ---------------------------------- Right eye ---------------------------------------------------------------------- #
            threshold_right = cv2.getTrackbarPos('Left', 'Dlib Landmarks')  # getting position of the trackbar
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
                            cv2.circle(frame, (right_center_pupil[0], right_center_pupil[1]), 1, (255, 0, 0), 2)

# ---------------------------------- Show vector after pressing v --------------------------------------------------- #
        if k == ord('v') and not press_v:
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
                    cv2.arrowedLine(frame, start_left, end_left, color=(0, 255, 0), thickness=1)
                    cv2.arrowedLine(frame, start_right, end_right, color=(0, 255, 0), thickness=1)

            press_p = False

# ---------------------------------- Get main point for calibration  ------------------------------------------------ #
        if k == ord('p') and not press_p:
            prepare_mask_for_calibration(screensize, 1, output_vector_in_eye_frame)
            press_1 = False
            press_p = True
            print('Look into lower left corner and press 1.')

        if k == ord('1') and not press_1:
            prepare_mask_for_calibration(screensize, 2, output_vector_in_eye_frame)
            press_1 = True
            press_2 = False
            press_detele = False
            lower_left_corner = lower_left(output_vector_in_eye_frame)
            print("Lower left corner saved.")
            print('Look into middle left and press 2.')

        if k == ord('2') and not press_2:
            prepare_mask_for_calibration(screensize, 3, output_vector_in_eye_frame)
            press_2 = True
            press_3 = False
            middle_left_corner = middle_left(output_vector_in_eye_frame)
            print("Middle left saved.")
            print('Look into upper left corner and press 3.')

        if k == ord('3') and not press_3:
            prepare_mask_for_calibration(screensize, 4, output_vector_in_eye_frame)
            press_3 = True
            press_4 = False
            upper_left_corner = upper_left(output_vector_in_eye_frame)
            print("Upper left corner saved.")
            print('Look into middle bottom and press 4.')

        if k == ord('4') and not press_4:
            prepare_mask_for_calibration(screensize, 5, output_vector_in_eye_frame)
            press_4 = True
            press_5 = False
            middle_bottom_corner = middle_bottom(output_vector_in_eye_frame)
            print("Middle bottom saved.")
            print('Look into middle of the screen and press 5.')

        if k == ord('5') and not press_5:
            prepare_mask_for_calibration(screensize, 6, output_vector_in_eye_frame)
            press_5 = True
            press_6 = False
            middle = middle_screen(output_vector_in_eye_frame)
            print("Middle saved.")
            print('Look into middle top and press 6.')

        if k == ord('6') and not press_6:
            prepare_mask_for_calibration(screensize, 7, output_vector_in_eye_frame)
            press_6 = True
            press_7 = False
            middle_up_corner = middle_up(output_vector_in_eye_frame)
            print("Middle top saved.")
            print('Look into lower right corner and press 7.')

        if k == ord('7') and not press_7:
            prepare_mask_for_calibration(screensize, 8, output_vector_in_eye_frame)
            press_7 = True
            press_8 = False
            lower_right_corner = lower_right(output_vector_in_eye_frame)
            print("Lower right corner saved.")
            print('Look into middle right corner and press 8.')

        if k == ord('8') and not press_8:
            prepare_mask_for_calibration(screensize, 9, output_vector_in_eye_frame)
            press_8 = True
            press_9 = False
            middle_right_corner = middle_right(output_vector_in_eye_frame)
            print("Middle right saved.")
            print('Look into upper right corner and press 9.')

        if k == ord('9') and not press_9:
            press_9 = True
            send_calibration_data_state = True
            press_e = False
            upper_right_corner = upper_right(output_vector_in_eye_frame)
            print("Upper right corner saved.")
            print("Pres enter for saving measured data or d for deleting measured data")
            cv2.destroyWindow('calibration')

# ---------------------------------- Delete everything and start over ----------------------------------------------- #
        if k == ord('d') and not press_detele:
            press_detele = True
            press_v = False
            print("Vector mode deactivated.")
            print("Measured data from corners were deleted.")
            print("Ready to start new measurment.")
            print("Press v to show vector")
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
            lower_left_corner = [0, 0, 0, 0]
            upper_right_corner = [0, 0, 0, 0]
            upper_left_corner = [0, 0, 0, 0]
            lower_right_corner = [0, 0, 0, 0]
            middle = [0, 0, 0, 0]
            middle_right_corner = [0, 0, 0, 0]
            middle_up_corner = [0, 0, 0, 0]
            middle_bottom_corner = [0, 0, 0, 0]
            middle_left_corner = [0, 0, 0, 0]
            send_calibration_data_state = False
            press_s = True
            press_e = False

# ---------------------------------- Show calibration points and interpolate them ----------------------------------- #
        if upper_left_corner != [0, 0, 0, 0] and upper_right_corner != [0, 0, 0, 0] and \
           lower_left_corner != [0, 0, 0, 0] and lower_right_corner != [0, 0, 0, 0] and middle != [0, 0, 0, 0] and \
           middle_right_corner != [0, 0, 0, 0] and middle_up_corner != [0, 0, 0, 0] and \
           middle_bottom_corner != [0, 0, 0, 0] and middle_left_corner != [0, 0, 0, 0] and \
           send_calibration_data_state and keyboard.is_pressed("enter") and not press_e:

            print("Data for calibration were measured successfully.")
            print("Lower left corner: ", lower_left_corner)
            print("Middle left: ", middle_left_corner)
            print("Upper left corner: ", upper_left_corner)
            print("Middle bottom: ", middle_bottom_corner)
            print("Middle: ", middle)
            print("Middle top: ", middle_up_corner)
            print("Lower right corner: ", lower_right_corner)
            print("Middle right: ", middle_right_corner)
            print("Upper right corner: ", upper_right_corner)
            print("Wait please. Calibration in progress...")

            # interpolate with calibrated points
            u_interp, v_interp = interpolation(lower_left_corner, middle_left_corner, upper_left_corner,
                                               middle_bottom_corner, middle, middle_up_corner,
                                               lower_right_corner, middle_right_corner, upper_right_corner,
                                               interpolation_size)

            print("Calibration done successfully.")
            send_calibration_data_state = False

            print("For starting eyetracker press e. For stopping eyetracker press s.")

# ---------------------------------- Start eyetracking -------------------------------------------------------------- #
        if k == ord('e') and not press_e:
            press_e = True
            press_s = False
            print("Eyetracker starts...")

        if press_e:
            normalized_u_interp, normalized_u = normalize_array(u_interp, output_vector_in_eye_frame[2])  # normalize u
            normalized_v_interp, normalized_v = normalize_array(v_interp, output_vector_in_eye_frame[3])  # normalize v

            result_numbers, result_x,\
            result_y, result_diff = find_closest_in_array(normalized_u_interp, normalized_v_interp,
                                                          (normalized_u, normalized_v),
                                                          0.1, 0.1)  # find best vector in interpolated field

            # show eyetracking result in frame called 'Eyetracking'
            start_point_draw, end_point_draw, mask_for_eyetracking_output = show_eyetracking(result_x, result_y, "Eyetracking",
                                                               (output_vector_in_eye_frame[0],
                                                                output_vector_in_eye_frame[1]),
                                                                interpolation_size, mask_for_eyetracking)

            #cv2.arrowedLine(mask_for_eyetracking, start_point_draw, end_point_draw, color=(0, 255, 0), thickness=1)

# ---------------------------------- Write video and show image ----------------------------------------------------- #
            out_detection.write(frame)
            mask_for_eyetracking_bgr = cv2.cvtColor(mask_for_eyetracking_output, cv2.COLOR_GRAY2BGR)
            out_mask.write(mask_for_eyetracking_bgr)
            cv2.imshow("Eyetracking", mask_for_eyetracking_output)

# ---------------------------------- Analyse, accuracy result, ... -------------------------------------------------- #
            isize = [interpolation_size[0]-1, interpolation_size[1]-1]

            upper_left_accuracy = accuracy_from_eyetracking([0, 0, u_interp[0, 0], v_interp[0, 0]],
                                                  (output_vector_in_eye_frame[2], output_vector_in_eye_frame[3]),
                                                  (result_x, result_y))

            upper_right_accuracy = accuracy_from_eyetracking([0, isize[0], u_interp[0, isize[0]], v_interp[0, isize[1]]],
                                                  (output_vector_in_eye_frame[2], output_vector_in_eye_frame[3]),
                                                  (result_x, result_y))

            lower_right_accuracy = accuracy_from_eyetracking([isize[1], isize[0], u_interp[isize[1],
                                                   isize[0]], v_interp[isize[1], isize[0]]],
                                                  (output_vector_in_eye_frame[2], output_vector_in_eye_frame[3]),
                                                  (result_x, result_y))

            lower_left_accuracy = accuracy_from_eyetracking([isize[1], 0, u_interp[isize[1], 0], v_interp[isize[1], 0]],
                                                  (output_vector_in_eye_frame[2], output_vector_in_eye_frame[3]),
                                                  (result_x, result_y))

            print("upper left accuracy", upper_left_accuracy)
            print("upper right accuracy", upper_right_accuracy)
            print("lower right accuracy", lower_right_accuracy)
            print("lower left accuracy", lower_left_accuracy)

# ---------------------------------- Stop eyetracking --------------------------------------------------------------- #
        if k == ord('s') and not press_s:
            press_s = True
            press_e = False
            cv2.waitKey(1)
            cv2.destroyWindow("Eyetracking")
            print("Eyetracker stops...")

# ---------------------------------- Show result and keyboard check ------------------------------------------------- #
        cv2.imshow('Dlib Landmarks', frame)  # visualization of detection
        k = cv2.waitKey(1) & 0xFF  # get key that is pressed on keyboard

# ---------------------------------- Quit program after pressing q -------------------------------------------------- #
        if k == ord('q'):
            cap.release()
            out_detection.release()
            out_mask.release()
            cv2.destroyAllWindows()
            break
    cap.release()  # release recording and streaming videos
    out_detection.release()
    out_mask.release()
    cv2.destroyAllWindows()  # close all windows


if __name__ == "__main__":
    main()
