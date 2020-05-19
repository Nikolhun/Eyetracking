import numpy as np
import matplotlib.pyplot as plt
import os

#######################################################################################################################
specification = "11x4_M_F_2_38"
max_rozliseni_y = 11
max_rozliseni_x = 4
#######################################################################################################################
path = r'C:\Users\notebook\Desktop\Eyetracking\grafy\11x4_M_F_2_38'
if not os.path.exists(path):
    os.makedirs(path)
target = np.load("mereni/" + specification + "/target_and_measured_vector_array.npy")
eyetracker = np.load("mereni/" + specification + "/result_eyetracker_array.npy")

target_x = target[0]+1
target_y = target[1]+1
eyetracker_x = eyetracker[0] + 1
eyetracker_y = eyetracker[1] + 1

eyetracker_x_append = []
eyetracker_y_append = []
target_x_append = []
target_y_append = []
for i in range(0, len(target_x[0])):
    now_eyetracker_x = eyetracker_x[0][i]
    if now_eyetracker_x != 0:
        eyetracker_x_append.append(now_eyetracker_x)

    now_eyetracker_y = eyetracker_y[0][i]
    if now_eyetracker_y != 0:
        eyetracker_y_append.append(now_eyetracker_y)

    now_target_x = target_x[0][i]
    if now_target_x != 0:
        target_x_append.append(now_target_x)

    now_target_y = target_y[0][i]
    if now_target_y != 0:
        target_y_append.append(now_target_y)

t_x = np.linspace(0, len(target_x[0]), num=len(target_x[0]))
t_y = np.linspace(0, len(target_y[0]), num=len(target_y[0]))
t_x_no_0 = np.linspace(0, len(target_x_append), num=len(target_x_append))
t_y_no_0 = np.linspace(0, len(target_y_append), num=len(target_y_append))

plt.plot(t_y_no_0[10:len(t_y_no_0) - 10], eyetracker_y_append[10:len(eyetracker_y_append) - 10], 'r')
plt.plot(t_y_no_0[10:len(t_y_no_0) - 10], target_y_append[10:len(target_y_append) - 10], 'b')
plt.ylim(1, max_rozliseni_y)
plt.xlabel('Číslo snímku')
plt.ylabel('Souřadnice x')
plt.grid(False)
plt.legend(('Eyetracker', 'Terč'), loc='best')
plt.savefig("grafy/" + specification + "/" + specification + "_graf_x_no_0.jpg")
plt.show()

plt.plot(t_y[10:len(t_y) - 10], eyetracker_y[0][10:len(eyetracker_y) - 11], 'r')
plt.plot(t_y[10:len(t_y) - 10], target_y[0][10:len(target_y) - 11], 'b')
plt.ylim(1, max_rozliseni_y)
plt.xlabel('Číslo snímku')
plt.ylabel('Souřadnice x')
plt.grid(False)
plt.legend(('Eyetracker', 'Terč'), loc='best')
plt.savefig("grafy/" + specification + "/" + specification + "_graf_x_with_0.jpg")
plt.show()

plt.plot(t_x_no_0[10:len(t_x_no_0) - 10], eyetracker_x_append[10:len(eyetracker_x_append) - 10], 'r')
plt.plot(t_x_no_0[10:len(t_x_no_0) - 10], target_x_append[10:len(target_x_append) - 10], 'b')
plt.ylim(1, max_rozliseni_x)
if max_rozliseni_y == 8:
    plt.yticks([0, 1, 2])
    plt.ylim(0, 2)
if max_rozliseni_y == 9:
    plt.yticks([0, 1, 2, 3])
    plt.ylim(0, 3)
if max_rozliseni_y == 11:
    plt.yticks([1, 2, 3, 4])
    plt.ylim(1, 4)
plt.xlabel('Číslo snímku')
plt.ylabel('Souřadnice y')
plt.grid(False)
plt.legend(('Eyetracker', 'Terč'), loc='best')
plt.savefig("grafy/" + specification + "/" + specification + "_graf_y_no_0.jpg")
plt.show()

plt.plot(t_x[10:len(t_x) - 10], eyetracker_x[0][10:len(eyetracker_x) - 11], 'r')
plt.plot(t_x[10:len(t_x) - 10], target_x[0][10:len(target_x) - 11], 'b')
plt.ylim(1, max_rozliseni_x)
if max_rozliseni_y == 8:
    plt.yticks([0, 1, 2])
    plt.ylim(0, 2)
if max_rozliseni_y == 9:
    plt.yticks([0, 1, 2, 3])
    plt.ylim(0, 3)
if max_rozliseni_y == 11:
    plt.yticks([1, 2, 3, 4])
    plt.ylim(1, 4)
plt.xlabel('Číslo snímku')
plt.ylabel('Souřadnice y')
plt.grid(False)
plt.legend(('Eyetracker', 'Terč'), loc='best')
plt.savefig("grafy/" + specification + "/" + specification + "_graf_y_with_0.jpg")
plt.show()

