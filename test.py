import numpy as np

size_of_output_screen = (16, 9)
coordinates_of_center_dot = (int((20*(size_of_output_screen[0]-1))/100), int((20*(size_of_output_screen[1]-1))/100))
part = "one"
cislo = int((80*size_of_output_screen[0])/100)-1

def change_coordinates_of_target(coordinates_of_center_dot, size_of_output_screen, part):
    # prvni = (20*size_of_output_screen[0]-1)/100
    left_right = coordinates_of_center_dot[0]
    top_bottom = coordinates_of_center_dot[1]
    #print(coordinates_of_center_dot[0], coordinates_of_center_dot[1])
    #print(size_of_output_screen)
    if (left_right << int((80*size_of_output_screen[0])/100)-1) and part == "one":
        left_right = left_right + 1
    elif (left_right == int((80*size_of_output_screen[0])/100)-1) and part == "one":
        part = "two"
    elif top_bottom << int((80*size_of_output_screen[1])/100)-1 and part == "two":
        top_bottom = top_bottom + 1
    elif top_bottom == int((80*size_of_output_screen[1])/100)-1 and part == "two":
        part = "three"
    elif left_right >> int((30*size_of_output_screen[0])/100)-1 and part == "three":
        left_right = left_right - 1
        print(left_right)
    elif left_right == int((30*size_of_output_screen[0])/100)-1 and part == "three":
        part = "four"
    elif top_bottom >> int((45 * size_of_output_screen[1]) / 100) - 1 and part == "four":
        top_bottom = top_bottom - 1
    elif top_bottom == int((45 * size_of_output_screen[1]) / 100) - 1 and part == "four":
        part = "five"
    elif left_right >> int((65*size_of_output_screen[0])/100)-1 and part == "five":
        left_right = left_right + 1
    elif left_right == int((65*size_of_output_screen[0])/100)-1 and part == "five":
        part = "six"
    elif top_bottom >> int((55 * size_of_output_screen[1]) / 100) - 1 and part == "six":
        top_bottom = top_bottom - 1
    elif top_bottom == int((55 * size_of_output_screen[1]) / 100) - 1 and part == "six":
        part = "seven"



    coordinates_of_center_dot = (left_right, top_bottom)
    return coordinates_of_center_dot, part

#while True:

    #coordinates_of_center_dot, part = change_coordinates_of_target(coordinates_of_center_dot, size_of_output_screen, part)
    #print(coordinates_of_center_dot, part)
print(int((30*(size_of_output_screen[0]-1))/100))