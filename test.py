if k == ord('v') and not press_v:  # "q" means close the detection
    press_v = True
    print("Vector mode activated.")
    print('For starting calibration mode press p.')

    # finding calibration vector
    calibrating_vector_in_frame_left = calibrate_vector_eye_center(left_center_pupil_in_eye_frame)
    calibrating_vector_in_frame_right = calibrate_vector_eye_center(right_center_pupil_in_eye_frame)

if press_v:
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
    press_p = False