import cv2
import dlib
import ctypes
import keyboard
import numpy as np
from dlib_landmarks import view_face_frame, draw_point, eye_center_dlib, landmarks_array, fill_frame, crop_eyes
from detect_pupil import converting_gray_to_hsv, filtration, gama_correction, preprocessing, contours_of_shape
from corneal_reflection import delete_corneal_reflection
from vector import find_vector, calibrate_vector_eye_center, vector_start_center
from calibration import upper_left, upper_right, middle_screen, lower_left, lower_right, middle_bottom, middle_left,\
    middle_right, middle_up, prepare_mask_for_calibration
from interpolate import interpolation
from eyetracking import find_closest_in_array, show_eyetracking, normalize_array, dimension, \
    empty_mask_for_eyetracking, make_array_from_vectors
from make_target import change_coordinates_of_target, change_coordinates_of_target_random, draw_line,\
    check_target_spot_before, check_target_spot_after

speed_of_target = 3
#######################################################################################################################
# ------------------------------- Initiation part ------------------------------------------------------------------- #
#######################################################################################################################
detector_dlib = dlib.get_frontal_face_detector()
predictor_dlib = dlib.shape_predictor("Dlib_landmarks/shape_predictor_68_face_landmarks.dat")

#------------------------------ Settup and screen size setting ----------------------------------------------------- #
print("Welcome in Eyetracking application!")
print("Are you on Raspberry (r) or on Windows (w)?")
computer = input()
if computer == "r":
    print("You have chosen Raspberry.")
    screensize = (120, 1280)
    print("Choose resolution of your eyetraker.")
    print("a) 160 x 90")
    print("b) 80 x 45")
    print("c) 16 x 9")
    eyetracker_resolution = input()
    if eyetracker_resolution == "a":
        print("You have chosen resolution 160 x 90.")
        size_of_output_screen = (160, 90)
    elif eyetracker_resolution == "b":
        print("You have chosen resolution 80 x 45.")
        size_of_output_screen = (80, 45)
    elif eyetracker_resolution == "c":
        print("You have chosen resolution 16 x 9.")
        size_of_output_screen = (16, 9)
    else:
        print("Choose between a to c.")
        size_of_output_screen = []
elif computer == "w":
    print("You have chosen Windows")
    user32 = ctypes.windll.user32  # for windows
    screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)  # for windows
    print("Choose resolution of your eyetraker.")
    print("a) 192 x 108")
    print("b) 96 x 54")
    print("c) 48 x 27")
    print("d) 48 x 27")
    print("e) 16 x 9")
    eyetracker_resolution = input()
    if eyetracker_resolution == "a":
        print("You have chosen resolution 192 x 108.")
        size_of_output_screen = (192, 108)
    elif eyetracker_resolution == "b":
        print("You have chosen resolution 96 x 54.")
        size_of_output_screen = (96, 54)
    elif eyetracker_resolution == "c":
        print("You have chosen resolution 48 x 27.")
        size_of_output_screen = (48, 27)
    elif eyetracker_resolution == "d":
        print("You have chosen resolution 32 x 18.")
        size_of_output_screen = (32, 17)
    elif eyetracker_resolution == "e":
        print("You have chosen resolution 16 x 9.")
        size_of_output_screen = (16, 9)
    else:
        print("Choose between a to e.")
        size_of_output_screen = []
else:
    print("Choose between r for Raspberry or w for Windows.")
    size_of_output_screen = []

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
    print("Set threshold for left and right eye.")
    print("Press v to show calibrating vector.")

    mask_for_eyetracking_bgr = empty_mask_for_eyetracking(size_of_output_screen)
    mask_bgr_reshaped_nearest = mask_for_eyetracking_bgr
    mask_reshape_dimenstion = dimension(mask_for_eyetracking_bgr,
                                        int((screensize[1] * 100) / size_of_output_screen[0]))

    cap = cv2.VideoCapture(0)  # reaching the port 0 for video capture
    fourcc_detection = cv2.VideoWriter_fourcc(*'XVID')
    out_detection = cv2.VideoWriter('detection.mkv', fourcc_detection, 20.0, (int(cap.get(3)), int(cap.get(4))))
    fourcc_mask = cv2.VideoWriter_fourcc(*'XVID')
    out_mask = cv2.VideoWriter('mask.mkv', fourcc_mask, 20.0, mask_reshape_dimenstion)

# ---------------------------------- Creating window for result and trackbars in it --------------------------------- #
    cv2.namedWindow('Dlib Landmarks')
    cv2.createTrackbar('Right', 'Dlib Landmarks', 0, 255, nothing)  # threshold track bar
    cv2.createTrackbar('Left', 'Dlib Landmarks', 0, 255, nothing)  # threshold track bar

# ---------------------------------- Initiation part ---------------------------------------------------------------- #
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
    press_c = True
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
    target = False
    coordinates_of_center_dot = (int((20 * size_of_output_screen[0] - 1) / 100),
                                 int((20 * size_of_output_screen[1] - 1) / 100))
    draw_point_after_next_target = False
    hit_target = False
    hit_target_value = []
    value_of_point = 0
    result_eyetracker_coordinate_x = []
    result_eyetracker_coordinate_y = []
    result_eyetracker_found_u_normalized = []
    result_eyetracker_found_v_normalized = []
    result_eyetracker_found_u = []
    result_eyetracker_found_v = []
    target_coordinate_x = []
    target_coordinate_y = []
    measured_vector_true_u_normalized = []
    measured_vector_true_v_normalized = []
    measured_vector_true_u = []
    measured_vector_true_v = []
    part = 1
    acceptation_of_change = []
    change_accepted = False
# ---------------------------------- Get the video frame and prepare it for detection ------------------------------- #
    while cap.isOpened():  # while th video capture is
        _, frame = cap.read()  # convert cap to matrix for future work
        frame = cv2.flip(frame, 1)  # flip video to not be mirrored
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # change color from rgb to gray
        # gray = gray[::4, ::4]  # for rpi

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

            # finding calibration vector
            calibrating_vector_in_frame_left = calibrate_vector_eye_center(left_center_pupil_in_eye_frame)
            calibrating_vector_in_frame_right = calibrate_vector_eye_center(right_center_pupil_in_eye_frame)

        if press_v:
            press_c = False
            # get vector coordinates in frame
            left_vector_center = vector_start_center(calibrating_vector_in_frame_left, [min_left[0], min_left[1]])
            right_vector_center = vector_start_center(calibrating_vector_in_frame_right, [min_right[0], min_right[1]])

            # finding vector
            output_vector_in_eye_frame = find_vector(left_center_pupil_in_eye_frame,
                                                     (calibrating_vector_in_frame_left[0],
                                                      calibrating_vector_in_frame_left[1]),
                                                     right_center_pupil_in_eye_frame,
                                                     (calibrating_vector_in_frame_right[0],
                                                      calibrating_vector_in_frame_right[1]))

            # start of vector
            start_left = (left_vector_center[0], left_vector_center[1])
            start_right = (right_vector_center[0], right_vector_center[1])

            # end of vector
            end_left = (int(output_vector_in_eye_frame[0] * 10) + left_vector_center[0],
                        int(output_vector_in_eye_frame[1] * 10) + left_vector_center[1])
            end_right = (int(output_vector_in_eye_frame[0] * 10) + right_vector_center[0],
                         int(output_vector_in_eye_frame[1] * 10) + right_vector_center[1])

            if end_left == [0, 0] or end_right == [0, 0]:
                print("Pupil not detected. Try to adjust threshold better and press v again..")

            if end_left is not None and end_right is not None:
                if output_vector_in_eye_frame[2] > 0:
                    cv2.arrowedLine(frame, start_left, end_left, color=(0, 255, 0), thickness=1)
                    cv2.arrowedLine(frame, start_right, end_right, color=(0, 255, 0), thickness=1)

# ---------------------------------- Get main point for calibration  ------------------------------------------------ #
        if k == ord('c') and not press_c:
            prepare_mask_for_calibration(screensize, 1)
            press_1 = False
            press_c = True
            print('Look into lower left corner and press 1.')

        if k == ord('1') and not press_1:
            prepare_mask_for_calibration(screensize, 2)
            press_1 = True
            press_2 = False
            press_detele = False
            lower_left_corner = lower_left(output_vector_in_eye_frame)
            print("Lower left corner saved.")
            print('Look into middle left and press 2.')

        if k == ord('2') and not press_2:
            prepare_mask_for_calibration(screensize, 3)
            press_2 = True
            press_3 = False
            middle_left_corner = middle_left(output_vector_in_eye_frame)
            print("Middle left saved.")
            print('Look into upper left corner and press 3.')

        if k == ord('3') and not press_3:
            prepare_mask_for_calibration(screensize, 4)
            press_3 = True
            press_4 = False
            upper_left_corner = upper_left(output_vector_in_eye_frame)
            print("Upper left corner saved.")
            print('Look into middle bottom and press 4.')

        if k == ord('4') and not press_4:
            prepare_mask_for_calibration(screensize, 5)
            press_4 = True
            press_5 = False
            middle_bottom_corner = middle_bottom(output_vector_in_eye_frame)
            print("Middle bottom saved.")
            print('Look into middle of the screen and press 5.')

        if k == ord('5') and not press_5:
            prepare_mask_for_calibration(screensize, 6)
            press_5 = True
            press_6 = False
            middle = middle_screen(output_vector_in_eye_frame)
            print("Middle saved.")
            print('Look into middle top and press 6.')

        if k == ord('6') and not press_6:
            prepare_mask_for_calibration(screensize, 7)
            press_6 = True
            press_7 = False
            middle_up_corner = middle_up(output_vector_in_eye_frame)
            print("Middle top saved.")
            print('Look into lower right corner and press 7.')

        if k == ord('7') and not press_7:
            prepare_mask_for_calibration(screensize, 8)
            press_7 = True
            press_8 = False
            lower_right_corner = lower_right(output_vector_in_eye_frame)
            print("Lower right corner saved.")
            print('Look into middle right corner and press 8.')

        if k == ord('8') and not press_8:
            prepare_mask_for_calibration(screensize, 9)
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
            prepare_mask_for_calibration(screensize, 0)

# ---------------------------------- Delete everything and start over ----------------------------------------------- #
        if k == ord('d') and not press_detele:
            press_detele = True
            press_v = False
            print("Vector mode deactivated.")
            print("Measured data from corners were deleted.")
            print("Ready to start new measurment.")
            print("Press v to show vector")
            press_c = True
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
            cv2.destroyWindow('calibration')

# ---------------------------------- Show calibration points and interpolate them ----------------------------------- #
        if upper_left_corner != [0, 0, 0, 0] and upper_right_corner != [0, 0, 0, 0] and \
           lower_left_corner != [0, 0, 0, 0] and lower_right_corner != [0, 0, 0, 0] and middle != [0, 0, 0, 0] and \
           middle_right_corner != [0, 0, 0, 0] and middle_up_corner != [0, 0, 0, 0] and \
           middle_bottom_corner != [0, 0, 0, 0] and middle_left_corner != [0, 0, 0, 0] and \
           send_calibration_data_state and keyboard.is_pressed("enter") and not press_e:

            send_calibration_data_state = False
            cv2.destroyWindow('calibration')

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
                                               size_of_output_screen)

            print("Calibration done successfully.")
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
            result_y, result_diff, nothing_found = find_closest_in_array(normalized_u_interp, normalized_v_interp,
                                                          (normalized_u, normalized_v),
                                                          0.1, 0.1)  # find best vector in interpolated field

# ---------------------------------- Change target after pressing m -------------------------------------------------- #
            if k == ord('m'):
                target = True
            if target == True:
                # step = int((3 * size_of_output_screen[1])/100)
                step = 0

                draw_line(mask_for_eyetracking_bgr, coordinates_of_center_dot, step, (255, 255, 255))

                mask_for_eyetracking_bgr_out, draw_point_after_next_target, \
                value_of_point = check_target_spot_before(draw_point_after_next_target, hit_target,
                                                          mask_for_eyetracking_bgr,
                                                          coordinates_of_center_dot, value_of_point, hit_target_value)
                mask_for_eyetracking_bgr = mask_for_eyetracking_bgr_out

                # get new center
                coordinates_of_center_dot_out, part_out, \
                acceptation_of_change, change_accepted = change_coordinates_of_target(coordinates_of_center_dot,
                                                                                      size_of_output_screen, part,
                                                                                      change_accepted,
                                                                                      acceptation_of_change)
                coordinates_of_center_dot = coordinates_of_center_dot_out
                if len(acceptation_of_change) == speed_of_target:
                    change_accepted = True
                    acceptation_of_change = []
                part = part_out

                if part == 16:
                    draw_line(mask_for_eyetracking_bgr, coordinates_of_center_dot, step, (255, 255, 255))
                    target = False

                hit_target_value = []
                hit_target = False

                mask_for_eyetracking_bgr_out, draw_point_after_next_target, \
                value_of_point = check_target_spot_after(mask_for_eyetracking_bgr, coordinates_of_center_dot)
                mask_for_eyetracking_bgr = mask_for_eyetracking_bgr_out

                draw_line(mask_for_eyetracking_bgr, coordinates_of_center_dot, step, (255, 0, 0))
# ---------------------------------- Random target after pressing n -------------------------------------------------- #
            if k == ord('n'):
                # step = int((3 * size_of_output_screen[1])/100)
                step = 0

                draw_line(mask_for_eyetracking_bgr, coordinates_of_center_dot, step, (255, 255, 255))

                mask_for_eyetracking_bgr_out, draw_point_after_next_target, \
                value_of_point = check_target_spot_before(draw_point_after_next_target, hit_target,
                                                          mask_for_eyetracking_bgr,
                                                          coordinates_of_center_dot, value_of_point, hit_target_value)
                mask_for_eyetracking_bgr = mask_for_eyetracking_bgr_out

                # get new center
                coordinates_of_center_dot = change_coordinates_of_target_random(size_of_output_screen)

                hit_target_value = []
                hit_target = False

                mask_for_eyetracking_bgr_out, draw_point_after_next_target, \
                value_of_point = check_target_spot_after(mask_for_eyetracking_bgr, coordinates_of_center_dot)
                mask_for_eyetracking_bgr = mask_for_eyetracking_bgr_out

                draw_line(mask_for_eyetracking_bgr, coordinates_of_center_dot, step, (255, 0, 0))

# ---------------------------------- Draw/show eyetracking ---------------------------------------------------------- #
            coor_x, coor_y, \
            mask_for_eyetracking_bgr, hit_target, \
            hit_target_value = show_eyetracking(result_x, result_y, "Eyetracking", size_of_output_screen,
                                                mask_for_eyetracking_bgr, coordinates_of_center_dot,
                                                hit_target, hit_target_value)

# ---------------------------------- Saving results ----------------------------------------------------------------- #
            dot_0 = coordinates_of_center_dot[1]
            dot_1 = coordinates_of_center_dot[0]
            u_found_normalized = normalized_u_interp[coor_x, coor_y]
            v_found_normalized = normalized_v_interp[coor_x, coor_y]
            u_found = u_interp[coor_x, coor_y]
            v_found = v_interp[coor_x, coor_y]

            if nothing_found == 1:
                print("Vector can't be found.")
                coor_x = -1
                coor_y = -1
                u_found_normalized = -1
                v_found_normalized = -1
                u_found = -1
                v_found = -1

                dot_0 = -1
                dot_1 = -1
                normalized_u = -1
                normalized_v = -1
                output_vector_in_eye_frame[2] = -1
                output_vector_in_eye_frame[3] = -1

            result_eyetracker_coordinate_x.append(coor_x)
            result_eyetracker_coordinate_y.append(coor_y)
            result_eyetracker_found_u_normalized.append(u_found_normalized)
            result_eyetracker_found_v_normalized.append(v_found_normalized)
            result_eyetracker_found_u.append(u_found)
            result_eyetracker_found_v.append(v_found)

            target_coordinate_x.append(dot_0)
            target_coordinate_y.append(dot_1)
            measured_vector_true_u_normalized.append(normalized_u)
            measured_vector_true_v_normalized.append(normalized_v)
            measured_vector_true_u.append(output_vector_in_eye_frame[2])
            measured_vector_true_v.append(output_vector_in_eye_frame[3])

# ---------------------------------- Write video and show image ----------------------------------------------------- #
            mask_bgr_reshaped_nearest = cv2.resize(mask_for_eyetracking_bgr, mask_reshape_dimenstion,
                                           interpolation=cv2.INTER_NEAREST)
            #cv2.imwrite("result/output.jpg", mask_bgr_reshaped_nearest)

            out_detection.write(frame)
            out_mask.write(mask_bgr_reshaped_nearest)
            cv2.imshow("Eyetracking", mask_bgr_reshaped_nearest)

# ---------------------------------- Stop eyetracking --------------------------------------------------------------- #
        if k == ord('s') and not press_s:
            press_s = True
            press_e = False
            cv2.waitKey(1)
            cv2.destroyWindow("Eyetracking")
            mask_for_eyetracking_bgr = empty_mask_for_eyetracking(size_of_output_screen)
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

            # make array from found and measured data
            result_eyetracker_array = make_array_from_vectors(result_eyetracker_coordinate_x,
                                                              result_eyetracker_coordinate_y,
                                                              result_eyetracker_found_u_normalized,
                                                              result_eyetracker_found_v_normalized,
                                                              result_eyetracker_found_u,
                                                              result_eyetracker_found_v)

            target_and_measured_vector_array = make_array_from_vectors(target_coordinate_x,
                                                                       target_coordinate_y,
                                                                       measured_vector_true_u_normalized,
                                                                       measured_vector_true_v_normalized,
                                                                       measured_vector_true_u,
                                                                       measured_vector_true_v)

            # save measured and found data
            np.save("results/result_eyetracker_array", result_eyetracker_array)
            np.save("results/target_and_measured_vector_array", target_and_measured_vector_array)
            np.save("results/eyetracker_screen_nearest", mask_bgr_reshaped_nearest)
            np.save("results/eyetracker_screen", mask_for_eyetracking_bgr)
            break
    cap.release()  # release recording and streaming videos
    out_detection.release()
    out_mask.release()
    cv2.destroyAllWindows()  # close all windows


if __name__ == "__main__":
    main()
