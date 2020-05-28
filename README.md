# Eyetracking
Eye movement tracking using the Raspberry Pi platform.

There are 3 types of eyetracking:
1. Two eyes measuring with vector starting in eye region center - "program_1.py" 
2. Two eyes measuring with vector starting in position given by key press - "program_2.py"
3. One eye measuring - "program_3.py"

Eyetracking scripts (program_1, program_2 and program_3) makes this NumPy arrays in folder "results":
- eyetracker_screen.npy
- eyetracker_screen_nearest.npy
- results_eyetracker_array.npy
- target_and_measured_vector_array.npy. 
These are used for analysing script described down below.

## Analyse data
Script for calculating differencies: "analyse_measured_data.py". It makes: 
- two heat maps (nerest neighbor interpolation method and 	cubic interpolation method): "version_heat_map_cubic.png" and "version_heat_map_nearest.png" 
- excel file: "version_results.xlsx"
- accuracy for x: coordinate "version_accuracy_x.npy
- accuracy for y: coordinate "version_accuracy_y.npy
- accuracy for :normalized u "version_accuracy_u_normalized.npy
- accuracy for normalized v: "version_accuracy_v_normalized.npy
- accuracy for u: "version_accuracy_u.npy
- accuracy for v: "version_accuracy_v.npy

## Controlling the program
Program is manageable with keyboard buttons, navigation is in command line or at the screen. Buttons are listed in table below.

| Button | Function |
| --- | --- |
| v | vector mode activation|
| c | calibration starts |
| 1-9 | saving calibration points |
| d | deleting measured values |
| e | starting eyetracker |
| s | stopping eyetracker |
| n | random target |
| m | moving target |
| q | quit eyetracker and save results |


(Program is made for Raspberry Pi 3, but it also runs on Windowns. For Windows compability uncomment few notes with sign #windows and comment few notes with sign #rpi.)






