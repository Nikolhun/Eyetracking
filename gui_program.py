from tkinter import *

main_window = Tk()
main_window.title("Eyetracking")
main_window.geometry("900x600")
threshold_left = Scale(main_window, label="Right eye", from_=0, to=255, orient=HORIZONTAL)
threshold_left.pack(side=TOP)
threshold_right = Scale(main_window, label="Left eye", from_=0, to=255, orient=HORIZONTAL)
threshold_right.pack(side=RIGHT)
frame = Frame(main_window, bg="white")
#frame.grid()

lmain = Label(frame)
lmain.grid()






main_window.mainloop()






