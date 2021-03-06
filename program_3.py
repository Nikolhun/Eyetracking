import cv2
import dlib
import numpy as np
import math
from detect_pupil import converting_gray_to_hsv, filtration, gama_correction, preprocessing, contours_of_shape
from corneal_reflection import delete_corneal_reflection
from calibration import prepare_mask_for_calibration, lower_right, lower_left, upper_right, upper_left, middle_up,\
    middle_right, middle_left, middle_bottom, middle_screen
from interpolate import interpolation
from eyetracking import find_closest_in_array, show_eyetracking, normalize_array, dimension, \
    empty_mask_for_eyetracking, make_array_from_vectors
from make_target import change_coordinates_of_target, change_coordinates_of_target_random, draw_line,\
    check_target_spot_before, check_target_spot_after, heat_map, show_target
# import ctypes  # for windows


#######################################################################################################################
# ------------------------------- Initiation part ------------------------------------------------------------------- #
#######################################################################################################################
detector_dlib = dlib.get_frontal_face_detector()
predictor_dlib = dlib.shape_predictor("Dlib_landmarks/shape_predictor_68_face_landmarks.dat")

# ------------------------------ Settup and screen size setting ----------------------------------------------------- #
print("Welcome in Eyetracking application!")
screensize = (120, 1280) # rpi
# user32 = ctypes.windll.user32  # for windows
# screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)  # for windows
print("Choose resolution of your eyetraker.")
print("a) 16 x 9")
print("b) 11 x 4")
print("c) 9 x 2")
print("d) 8 x 1")
eyetracker_resolution = input()
if eyetracker_resolution == "a":
    print("You have chosen resolution 16 x 9.")
    size_of_output_screen = (16, 9)
    speed_of_target = 3
elif eyetracker_resolution == "b":
    print("You have chosen resolution 11 x 4.")
    size_of_output_screen = (11, 4)
    speed_of_target = 4
elif eyetracker_resolution == "c":
    print("You have chosen resolution 9 x 2.")
    size_of_output_screen = (9, 2)
    speed_of_target = 5
elif eyetracker_resolution == "d":
    print("You have chosen resolution 8 x 1.")
    size_of_output_screen = (8, 1)
    speed_of_target = 6
else:
    print("Choose between a to d.")
    size_of_output_screen = []

#######################################################################################################################
# ------------------------------- Creating trackbar ----------------------------------------------------------------- #
#######################################################################################################################
def nothing(x):
    '''
    creating Trackbar
    '''
    pass


def eye_center(position_in_frame, move_frame):
    """
    gets center of eye
    :param position_in_frame: position in frame (x,y)
    :param move_frame: moving with frame
    :return: coordinates of eye center in frame; coordinates of eye center
    """
    height, width = position_in_frame.shape
    eye_center_coordinates_in_frame = (int(width/2), int(height/2))
    eye_center_coordinates = (int(width/2 + move_frame[1]), int(height/2 + move_frame[0]))
    return eye_center_coordinates_in_frame, eye_center_coordinates


def find_vector(pupil_center, eye_center_coordinates):
    """
    finds vector of eye movement
    :param pupil_center: coordinates of pupil center (x, y)
    :param eye_center_coordinates: coocoordinates of eye center (x, y)
    :return: x coordinate of vector, y coordinate of vector, magnitude of vector, direction of vector
    """
    x = pupil_center[0] - eye_center_coordinates[0]  # vector x
    y = pupil_center[1] - eye_center_coordinates[1]  # vector y

    if y == 0:
        y = 0.000001

    magnitude = math.sqrt(x * x + y * y)  # magnitude of vector
    direction_radian = math.atan(x / y)  # direction in radians
    direction = (direction_radian * 180) / math.pi  # from radians to degrees

    return x, y, magnitude, direction

#######################################################################################################################
# ---------------------------------- Main --------------------------------------------------------------------------- #
#######################################################################################################################
# ---------------------------------- Making video capture and video writers ----------------------------------------- #
def main():
    print("Set threshold for left and right eye.")
    print("Press v to show calibrating vector.")

# ---------------------------------- Making empty arrays for future use -------------------------------------------- #
    mask_for_eyetracking_bgr = empty_mask_for_eyetracking(size_of_output_screen)
    mask_for_eyetracking_target = empty_mask_for_eyetracking(size_of_output_screen)
    mask_for_eyetracking_target_save = empty_mask_for_eyetracking(size_of_output_screen)
    mask_bgr_reshaped_nearest = mask_for_eyetracking_bgr
    mask_reshape_dimenstion = dimension(mask_for_eyetracking_bgr,
                                        int((screensize[1] * 100) / size_of_output_screen[0]))
# ---------------------------------- Reaching the port 0 for video capture ------------------------------------------ #
    cap = cv2.VideoCapture(0)

# ---------------------------------- Starting making video ---------------------------------------------------------- #
    fourcc_detection = cv2.VideoWriter_fourcc(*'XVID')
    out_detection = cv2.VideoWriter('detection.mkv', fourcc_detection, 20.0, (int(cap.get(3)), int(cap.get(4))))
    fourcc_mask = cv2.VideoWriter_fourcc(*'XVID')
    out_mask = cv2.VideoWriter('mask.mkv', fourcc_mask, 20.0, mask_reshape_dimenstion)

# ---------------------------------- Creating window for result and trackbars in it --------------------------------- #
    cv2.namedWindow('Dlib Landmarks')
    cv2.createTrackbar('Eye', 'Dlib Landmarks', 0, 255, nothing)  # threshold track bar


# ---------------------------------- Initiation part ---------------------------------------------------------------- #
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
    u_interp = np.zeros(size_of_output_screen, np.uint8)
    v_interp = np.zeros(size_of_output_screen, np.uint8)
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
    press_n = False
    target_m = False
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
    method = "n"

# ---------------------------------- Get the video frame and prepare it for detection ------------------------------- #
    while cap.isOpened():  # while th video capture is
        _, frame = cap.read()  # convert cap to matrix for future work
        frame = frame[::4, ::4]  # for rpi
        frame = cv2.flip(frame, 1)  # flip video to not be mirrored
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # change color from rgb to gray

# ---------------------------------- Crop image and show frame for one eye ------------------------------------------ #
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
                        # position of left pupil
                        pupil_center_in_frame = [cx, cy]
                        pupil_center = [cx + move_frame_x, cy + move_frame_y]
                        cv2.circle(frame, (pupil_center[0], pupil_center[1]), 1, (255, 0, 0), 2)

# ---------------------------------- Show vector after pressing v --------------------------------------------------- #
        if k == ord('v') and not press_v:
            press_v = True  # for active vector drawing
            print("Vector mode activated.")
            print('For starting calibration mode press p.')

        if press_v:  # active vector
            press_c = False
            # finding eye center
            eye_center_coordinates_in_frame, eye_center_coordinates = eye_center(crop_gray,
                                                                                 (move_frame_y, move_frame_x))

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

# ---------------------------------- Get nine calibration points ------------------------------------------------ #
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
            lower_left_corner = lower_left(vector)
            print("Lower left corner saved.")
            print('Look into middle left and press 2.')

        if k == ord('2') and not press_2:
            prepare_mask_for_calibration(screensize, 3)
            press_2 = True
            press_3 = False
            middle_left_corner = middle_left(vector)
            print("Middle left saved.")
            print('Look into upper left corner and press 3.')

        if k == ord('3') and not press_3:
            prepare_mask_for_calibration(screensize, 4)
            press_3 = True
            press_4 = False
            upper_left_corner = upper_left(vector)
            print("Upper left corner saved.")
            print('Look into middle bottom and press 4.')

        if k == ord('4') and not press_4:
            prepare_mask_for_calibration(screensize, 5)
            press_4 = True
            press_5 = False
            middle_bottom_corner = middle_bottom(vector)
            print("Middle bottom saved.")
            print('Look into middle of the screen and press 5.')

        if k == ord('5') and not press_5:
            prepare_mask_for_calibration(screensize, 6)
            press_5 = True
            press_6 = False
            middle = middle_screen(vector)
            print("Middle saved.")
            print('Look into middle top and press 6.')

        if k == ord('6') and not press_6:
            prepare_mask_for_calibration(screensize, 7)
            press_6 = True
            press_7 = False
            middle_up_corner = middle_up(vector)
            print("Middle top saved.")
            print('Look into lower right corner and press 7.')

        if k == ord('7') and not press_7:
            prepare_mask_for_calibration(screensize, 8)
            press_7 = True
            press_8 = False
            lower_right_corner = lower_right(vector)
            print("Lower right corner saved.")
            print('Look into middle right corner and press 8.')

        if k == ord('8') and not press_8:
            prepare_mask_for_calibration(screensize, 9)
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
            mask_for_eyetracking_bgr = empty_mask_for_eyetracking(size_of_output_screen)
            mask_for_eyetracking_target = empty_mask_for_eyetracking(size_of_output_screen)
            mask_for_eyetracking_target_save = empty_mask_for_eyetracking(size_of_output_screen)

# ---------------------------------- Show calibration points and interpolate them ----------------------------------- #
        if upper_left_corner != [0, 0, 0, 0] and upper_right_corner != [0, 0, 0, 0] and \
           lower_left_corner != [0, 0, 0, 0] and lower_right_corner != [0, 0, 0, 0] and middle != [0, 0, 0, 0] and \
           middle_right_corner != [0, 0, 0, 0] and middle_up_corner != [0, 0, 0, 0] and \
           middle_bottom_corner != [0, 0, 0, 0] and middle_left_corner != [0, 0, 0, 0] and \
           send_calibration_data_state and k == 13 and not press_e:

            send_calibration_data_state = False
            cv2.destroyWindow('calibration')

            # print calibrating points vectors
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
            press_e = True  # activates eyetracking mode
            press_s = False
            print("You can choose between random target and fix target."
                  "For random target press n, for fix target press m.")
            print("Eyetracker starts...")

        if press_e:  # active eytracking mode
            normalized_u_interp, normalized_u = normalize_array(u_interp, vector[2])  # normalize u
            normalized_v_interp, normalized_v = normalize_array(v_interp, vector[3])  # normalize v

            # find best vector in interpolated field
            result_numbers, result_x,\
            result_y, result_diff, nothing_found = find_closest_in_array(normalized_u_interp, normalized_v_interp,
                                                                        (normalized_u, normalized_v),
                                                                         0.2, 0.2)

# ---------------------------------- Start moving target after pressing m ------------------------------------------- #
            if k == ord('m'):
                print("Target mode activated.")
                target_m = True
                mask_for_eyetracking_target = empty_mask_for_eyetracking(size_of_output_screen)
                method = "m"
                print("Moving target started")

            if target_m:  # method is moving target
                step = 0

                # delete old target
                draw_line(mask_for_eyetracking_target, coordinates_of_center_dot, step, (255, 255, 255))

                # get new center
                coordinates_of_center_dot_out, part_out,\
                acceptation_of_change, change_accepted = change_coordinates_of_target(coordinates_of_center_dot,
                                                                                      size_of_output_screen, part,
                                                                                      change_accepted,
                                                                                      acceptation_of_change)
                coordinates_of_center_dot = coordinates_of_center_dot_out
                if len(acceptation_of_change) == speed_of_target:  # buffer for target
                    change_accepted = True
                    acceptation_of_change = []
                part = part_out

                if part == 16:
                    draw_line(mask_for_eyetracking_target, coordinates_of_center_dot, step, (255, 255, 255))

                hit_target_value = []
                hit_target = False

                # draw target
                draw_line(mask_for_eyetracking_target, coordinates_of_center_dot, step, (255, 0, 0))

# ---------------------------------- Random target after pressing n -------------------------------------------------- #
            if k == ord('n') and not press_n:  # recount coordinates for target
                mask_for_eyetracking_bgr = empty_mask_for_eyetracking(size_of_output_screen)
                press_n = True
            elif k == ord('n') and press_n:  # start moving target if not started
                print("Target mode activated.")
                print("Press n for next target.")
                method = "n"
                step = 0

                # delete old target
                draw_line(mask_for_eyetracking_bgr, coordinates_of_center_dot, step, (255, 255, 255))

                # check if there is a eyetracker value before placing target
                mask_for_eyetracking_bgr_out, draw_point_after_next_target, \
                value_of_point = check_target_spot_before(draw_point_after_next_target, hit_target,
                                                          mask_for_eyetracking_bgr, coordinates_of_center_dot,
                                                          value_of_point, hit_target_value)
                mask_for_eyetracking_bgr = mask_for_eyetracking_bgr_out

                # get new center
                coordinates_of_center_dot = change_coordinates_of_target_random(size_of_output_screen)

                hit_target_value = []
                hit_target = False

                # check if eyetracket hitted target
                mask_for_eyetracking_bgr_out, draw_point_after_next_target, \
                value_of_point = check_target_spot_after(mask_for_eyetracking_bgr, coordinates_of_center_dot)
                mask_for_eyetracking_bgr = mask_for_eyetracking_bgr_out

                # draw target
                draw_line(mask_for_eyetracking_bgr, coordinates_of_center_dot, step, (255, 0, 0))

# ---------------------------------- Draw/show eyetracking ---------------------------------------------------------- #
            if method == "n":
                cv2.namedWindow("Eyetracking", cv2.WINDOW_NORMAL)  # make new window with window name
                cv2.setWindowProperty("Eyetracking", cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)  # set window to full screen
            elif method == "m":
                cv2.namedWindow("Target", cv2.WINDOW_NORMAL)  # make new window with window name
                cv2.setWindowProperty("Target", cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)  # set window to full screen
            # show eyetracking result in frame called 'Eyetracking'
            coor_x, coor_y, \
            mask_for_eyetracking_bgr, hit_target,\
            hit_target_value = show_eyetracking(result_x, result_y, size_of_output_screen,
                                                mask_for_eyetracking_bgr, coordinates_of_center_dot,
                                                hit_target, hit_target_value)
            # show target
            coor_x_target, coor_y_target, \
            mask_for_eyetracking_target_save = show_target(mask_for_eyetracking_target_save, coordinates_of_center_dot)

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
            measured_vector_true_u.append(vector[2])
            measured_vector_true_v.append(vector[3])

# ---------------------------------- Write video and show image ----------------------------------------------------- #
            mask_bgr_reshaped_nearest = cv2.resize(mask_for_eyetracking_bgr, mask_reshape_dimenstion,
                                                   interpolation=cv2.INTER_NEAREST)
            mask_bgr_target_reshaped_nearest = cv2.resize(mask_for_eyetracking_target, mask_reshape_dimenstion,
                                                          interpolation=cv2.INTER_NEAREST)

            out_detection.write(frame)
            out_mask.write(mask_bgr_reshaped_nearest)

            if method == "n":  # for random target
                cv2.imshow("Eyetracking", mask_bgr_reshaped_nearest)
            elif method == "m":  # for animated target
                cv2.imshow("Target", mask_bgr_target_reshaped_nearest)

# ---------------------------------- Stop eyetracking --------------------------------------------------------------- #
        if k == ord('s') and not press_s:
            press_s = True
            press_e = False
            cv2.waitKey(1)
            cv2.destroyWindow("Eyetracking")
            cv2.destroyWindow("Target")
            mask_for_eyetracking_bgr = empty_mask_for_eyetracking(size_of_output_screen)
            mask_for_eyetracking_target = empty_mask_for_eyetracking(size_of_output_screen)
            mask_for_eyetracking_target_save = empty_mask_for_eyetracking(size_of_output_screen)
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

            heat_map(mask_for_eyetracking_bgr, (screensize[1], screensize[0]), "eyetracking")
            heat_map(mask_for_eyetracking_target_save, (screensize[1], screensize[0]), "target")
            # save measured and found data
            np.save("results/result_eyetracker_array", result_eyetracker_array)
            np.save("results/target_and_measured_vector_array", target_and_measured_vector_array)
            np.save("results/eyetracker_screen_nearest", mask_bgr_reshaped_nearest)
            np.save("results/eyetracker_screen", mask_for_eyetracking_bgr)
            np.save("results/eyetracker_target", mask_for_eyetracking_target_save)
            break
    cap.release()  # release recording and streaming videos
    out_detection.release()
    out_mask.release()
    cv2.destroyAllWindows()  # close all windows


if __name__ == "__main__":
    main()
