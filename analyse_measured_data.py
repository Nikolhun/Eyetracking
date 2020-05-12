import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import cv2
import ctypes

specification = "ver1"

user32 = ctypes.windll.user32  # for windows
screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)  # for windows
#screensize = (120, 1280) rpi
# ---------------------------------- Import measured data ----------------------------------------------------------- #
eyetracker_data = np.load('results/result_eyetracker_array.npy')
target_vector_data = np.load('results/target_and_measured_vector_array.npy')

eyetracker_screen_bgr_nearest = np.load('results/eyetracker_screen_nearest.npy')
eyetracker_screen_gray_nearest = cv2.cvtColor(eyetracker_screen_bgr_nearest, cv2.COLOR_BGR2GRAY)
eyetracker_screen_gray_nearest = eyetracker_screen_gray_nearest / eyetracker_screen_gray_nearest.max()
eyetracker_screen_gray_nearest = np.abs(eyetracker_screen_gray_nearest - 1)

eyetracker_screen_bgr = np.load('results/eyetracker_screen.npy')
eyetracker_screen_bgr = cv2.resize(eyetracker_screen_bgr, screensize,
                                           interpolation=cv2.INTER_CUBIC)
eyetracker_screen_gray = cv2.cvtColor(eyetracker_screen_bgr, cv2.COLOR_BGR2GRAY)
eyetracker_screen_gray = eyetracker_screen_gray / eyetracker_screen_gray.max()
eyetracker_screen_gray = np.abs(eyetracker_screen_gray - 1)

eyetracker_data_x = eyetracker_data[0]
eyetracker_data_y = eyetracker_data[1]
eyetracker_data_u_normalized = eyetracker_data[2]
eyetracker_data_v_normalized = eyetracker_data[3]
eyetracker_data_u = eyetracker_data[4]
eyetracker_data_v = eyetracker_data[5]

target_vector_data_x = target_vector_data[0]
target_vector_data_y = target_vector_data[1]
target_vector_data_u_normalized = target_vector_data[2]
target_vector_data_v_normalized = target_vector_data[3]
target_vector_data_u = target_vector_data[4]
target_vector_data_v = target_vector_data[5]

x = []
y = []
u_normalized = []
v_normalized = []
u = []
v = []
unfined_vectors = 0

for i in range(0, (eyetracker_data.shape[2]-1)):

    if eyetracker_data_x[0][i] != float(-1):
        x0 = np.abs(eyetracker_data_x[0][i] - target_vector_data_x[0][i])
        y0 = np.abs(eyetracker_data_y[0][i] - target_vector_data_y[0][i])
        u0_normalized = np.abs(eyetracker_data_u_normalized[0][i] - target_vector_data_u_normalized[0][i])
        v0_normalized = np.abs(eyetracker_data_v_normalized[0][i] - target_vector_data_v_normalized[0][i])
        u0 = np.abs(eyetracker_data_u[0][i] - target_vector_data_u[0][i])
        v0 = np.abs(eyetracker_data_v[0][i] - target_vector_data_v[0][i])

    elif eyetracker_data_x[0][i] == float(-1):
        unfined_vectors = unfined_vectors + 1
        x0 = 0 #float(-1)
        y0 = 0 #float(-1)
        u0_normalized = 0 #float(-1)
        v0_normalized = 0 #float(-1)
        u0 = 0 #float(-1)
        v0 = 0 #float(-1)

    x.append(x0)
    y.append(y0)
    u_normalized.append(u0_normalized)
    v_normalized.append(v0_normalized)
    u.append(u0)
    v.append(v0)

#x = np.concatenate(x)
#y = np.concatenate(y)
#u_normalized = np.concatenate(u_normalized)
#v_normalized = np.concatenate(v_normalized)
#u = np.concatenate(u)
#v = np.concatenate(v)
x = np.array(x)
y = np.array(y)
u_normalized = np.array(u_normalized)
v_normalized = np.array(v_normalized)
u = np.array(u)
v = np.array(v)
# ---------------------------------- Save data ---------------------------------------------------------------------- #
np.save("results/" + specification + "_accuracy_x", x)
np.save("results/" + specification + "_accuracy_y", y)
np.save("results/" + specification + "_accuracy_u_normalized", u_normalized)
np.save("results/" + specification + "_accuracy_v_normalized", v_normalized)
np.save("results/" + specification + "_accuracy_u", u)
np.save("results/" + specification + "_accuracy_v", v)

# ---------------------------------- Excel -------------------------------------------------------------------------- #

data = {'X': x,
        'Y': y,
        'U': u,
        'V': v,
        'U normalized': u_normalized,
        'V normalized': v_normalized,
        'U percent': u_normalized*100,
        'V percent': v_normalized*100}
df = pd.DataFrame(data, columns=['X', 'Y', 'U', 'V', 'U normalized', 'V normalized', 'U percent', 'V percent'])

writer = pd.ExcelWriter("results/" + specification + "_results.xlsx", engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1')
writer.save()
writer.close()
# ---------------------------------- Heat map ----------------------------------------------------------------------- #
plt.rcParams['figure.figsize'] = (16, 9)

midpoint_nearest = (eyetracker_screen_gray_nearest.max() - eyetracker_screen_gray_nearest.min())/1.5
heat_map_nearest = sb.heatmap(eyetracker_screen_gray_nearest, center=midpoint_nearest, vmin=0, vmax=1, xticklabels=False,
                              yticklabels=False, cmap="Blues", cbar=False)
plt.show()
figure_nearest = heat_map_nearest.get_figure().savefig("results/" + specification + "_heat_map_nearest.png")

midpoint_cubic = (eyetracker_screen_gray.max() - eyetracker_screen_gray.min())/1.5
heat_map_cubic = sb.heatmap(eyetracker_screen_gray, center=midpoint_nearest, vmin=0, vmax=1, xticklabels=False,
                            yticklabels=False, cmap="Blues", cbar=False)
plt.show()
figure_cubic = heat_map_cubic.get_figure().savefig("results/" + specification + "_heat_map_cubic.png") #dpi = 400

print("In this measurement was", unfined_vectors, "not found vectors.")






