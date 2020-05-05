import cv2
import dlib
import ctypes
import keyboard
import numpy as np
from one_eye_functions import eye_center, find_vector
from detect_pupil import converting_gray_to_hsv, filtration, gama_correction, preprocessing, contours_of_shape
from corneal_reflection import delete_corneal_reflection
from calibration import prepare_mask_for_calibration, lower_right, lower_left, upper_right, upper_left, middle_up,\
    middle_right, middle_left, middle_bottom, middle_screen
from interpolate import interpolation
from eyetracking import normalize_array, find_closest_in_array, show_eyetracking, accuracy_from_eyetracking



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
    cv2.createTrackbar('Eye', 'Dlib Landmarks', 0, 255, nothing)  # threshold track bar

# ---------------------------------- Initiation part ---------------------------------------------------------------- #
    mask_for_eyetracking = np.zeros((interpolation_size[1], interpolation_size[0]), np.uint8) + 255
    pupil_center_in_frame = [0, 0]
    vector = [0, 0, 0, 0]
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

# ---------------------------------- Crop image to one eye ---------------------------------------------------------- #
        crop_gray = gray[180:300, 240:400]
        cv2.rectangle(frame, (240, 180), (240+160, 180+120), (0, 255, 0), 2)
        move_frame_x = 240
        move_frame_y = 180

# ---------------------------------- Eye detection ------------------------------------------------------------------ #
        threshold = cv2.getTrackbarPos('Eye', 'Dlib Landmarks')  # getting position of the trackbar
        no_reflex = delete_corneal_reflection(crop_gray, threshold)  # deleting corneal reflex
        hsv_img = converting_gray_to_hsv(no_reflex)  # converting frame to hsv
        filtrated_img = filtration(hsv_img)  # applying some filtration
        gama_corrected = gama_correction(filtrated_img, 1.2)  # gama correction
        eye_preprocessed = preprocessing(gama_corrected, threshold)  # morfological operations
        cv2.imshow("eye_processed", eye_preprocessed)
        contours = contours_of_shape(eye_preprocessed, threshold)  # get contours
        if contours is not None:
            for c in contours:
                if c is not None:
                    m = cv2.moments(c)
                    if m["m00"] > 1:
                        cx = int(m["m10"] / m["m00"])  # x coordinate for middle of blob
                        cy = int(m["m01"] / m["m00"])  # y coordinate for middle of blob
                        # cv2.drawContours(eye_no_eyebrows_left, [c], -1, (0, 255, 0), 2)
                        # cv2.circle(left_eye_crop, (cx, cy), 1, (0, 0, 255), 2)
                        # position of left pupil
                        pupil_center_in_frame = [cx, cy]
                        pupil_center = [cx + move_frame_x, cy + move_frame_y]
                        cv2.circle(frame, (pupil_center[0], pupil_center[1]), 1, (255, 0, 0), 2)


# ---------------------------------- Show vector after pressing v --------------------------------------------------- #
        if k == ord('v') and not press_v:
            press_v = True
            print("Vector mode activated.")
            print('For starting calibration mode press p.')

        if press_v:

            # finding eye center
            eye_center_coordinates_in_frame, eye_center_coordinates = eye_center(crop_gray, (move_frame_y, move_frame_x))

            # finding vector
            vector = find_vector(pupil_center_in_frame, eye_center_coordinates_in_frame)

            # start of vector
            start = (eye_center_coordinates[0], eye_center_coordinates[1])

            # end of vector
            end = (int(vector[0] + eye_center_coordinates[0]),
                   int(vector[1] + eye_center_coordinates[1]))

            # show vector in frame
            if end == [0, 0]:
                print("Pupil not detected. Try to adjust threshold better and press v again..")
            else:
                if vector[2] > 0:
                    cv2.arrowedLine(frame, start, end, color=(0, 255, 0), thickness=1)

            press_p = False

# ---------------------------------- Get main point for calibration  ------------------------------------------------ #
        if k == ord('p') and not press_p:
            prepare_mask_for_calibration(screensize, 1, vector)
            press_1 = False
            press_p = True
            print('Look into lower left corner and press 1.')

        if k == ord('1') and not press_1:
            prepare_mask_for_calibration(screensize, 2, vector)
            press_1 = True
            press_2 = False
            press_detele = False
            lower_left_corner = lower_left(vector)
            print("Lower left corner saved.")
            print('Look into middle left and press 2.')

        if k == ord('2') and not press_2:
            prepare_mask_for_calibration(screensize, 3, vector)
            press_2 = True
            press_3 = False
            middle_left_corner = middle_left(vector)
            print("Middle left saved.")
            print('Look into upper left corner and press 3.')

        if k == ord('3') and not press_3:
            prepare_mask_for_calibration(screensize, 4, vector)
            press_3 = True
            press_4 = False
            upper_left_corner = upper_left(vector)
            print("Upper left corner saved.")
            print('Look into middle bottom and press 4.')

        if k == ord('4') and not press_4:
            prepare_mask_for_calibration(screensize, 5, vector)
            press_4 = True
            press_5 = False
            middle_bottom_corner = middle_bottom(vector)
            print("Middle bottom saved.")
            print('Look into middle of the screen and press 5.')

        if k == ord('5') and not press_5:
            prepare_mask_for_calibration(screensize, 6, vector)
            press_5 = True
            press_6 = False
            middle = middle_screen(vector)
            print("Middle saved.")
            print('Look into middle top and press 6.')

        if k == ord('6') and not press_6:
            prepare_mask_for_calibration(screensize, 7, vector)
            press_6 = True
            press_7 = False
            middle_up_corner = middle_up(vector)
            print("Middle top saved.")
            print('Look into lower right corner and press 7.')

        if k == ord('7') and not press_7:
            prepare_mask_for_calibration(screensize, 8, vector)
            press_7 = True
            press_8 = False
            lower_right_corner = lower_right(vector)
            print("Lower right corner saved.")
            print('Look into middle right corner and press 8.')

        if k == ord('8') and not press_8:
            prepare_mask_for_calibration(screensize, 9, vector)
            press_8 = True
            press_9 = False
            middle_right_corner = middle_right(vector)
            print("Middle right saved.")
            print('Look into upper right corner and press 9.')

        if k == ord('9') and not press_9:
            press_9 = True
            send_calibration_data_state = True
            press_e = False
            upper_right_corner = upper_right(vector)
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
           send_calibration_data_state and k == 13 and not press_e:

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
            normalized_u_interp, normalized_u = normalize_array(u_interp, vector[2])  # normalize u
            normalized_v_interp, normalized_v = normalize_array(v_interp, vector[3])  # normalize v

            result_numbers, result_x,\
            result_y, result_diff = find_closest_in_array(normalized_u_interp, normalized_v_interp,
                                                          (normalized_u, normalized_v),
                                                          0.1, 0.1)  # find best vector in interpolated field

            # show eyetracking result in frame called 'Eyetracking'
            start_point_draw, end_point_draw,\
            mask_for_eyetracking_output = show_eyetracking(result_x, result_y, "Eyetracking",
                                                           (vector[0],
                                                            vector[1]),
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
                                                  (vector[2], vector[3]),
                                                  (result_x, result_y))

            upper_right_accuracy = accuracy_from_eyetracking([0, isize[0], u_interp[0, isize[0]], v_interp[0, isize[1]]],
                                                  (vector[2], vector[3]),
                                                  (result_x, result_y))

            lower_right_accuracy = accuracy_from_eyetracking([isize[1], isize[0], u_interp[isize[1],
                                                   isize[0]], v_interp[isize[1], isize[0]]],
                                                  (vector[2], vector[3]),
                                                  (result_x, result_y))

            lower_left_accuracy = accuracy_from_eyetracking([isize[1], 0, u_interp[isize[1], 0], v_interp[isize[1], 0]],
                                                  (vector[2], vector[3]),
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
