# Eyetracking
Eye movement tracking using the Raspberry Pi platform.

There are 3 types of eyetracking:
1. One eye measuring - "one_eye_version.py"
2. Two eyes measuring with vector starting in eye region center - "program.py" 
3. Two eyes measuring with vector starting in position given by key press - "program_vector_center.py"

For Raspberry Pi compatibility, uncomment few notes with sign: #rpi and comment few notes with sign #windows.

Eyetracking scripts (one_eye_version.py, program.py, program_vector_center.py) makes this files into resuts file:
- eyetracker_screen.npy
- eyetracker_screen_nearest.npy
- results_eyetracker_array.npy
- target_and_measured_vector_array.npy. 
These are usef for analysing script described down below.

# Analyse
Script for calculating result. "analyse_measured_data.py". It makes: 
- two heat maps (nerest neighbor interpolation method and 	cubic interpolation method) "version_heat_map_cubic.png" and "version_heat_map_nearest.png" 
- excel file "version_results.xlsx"
- accuracy for x coordinate "version_accuracy_x.npy
- accuracy for y coordinate "version_accuracy_y.npy
- accuracy for normalized u "version_accuracy_u_normalized.npy
- accuracy for normalized v "version_accuracy_v_normalized.npy
- accuracy for u "version_accuracy_u.npy
- accuracy for v "version_accuracy_v.npy







