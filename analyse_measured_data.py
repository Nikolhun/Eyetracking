import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import cv2
import ctypes

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
for i in range(0, len(eyetracker_data[0])):

    x0 = np.abs(eyetracker_data_x[i] - target_vector_data_x[i])
    y0 = np.abs(eyetracker_data_y[i] - target_vector_data_y[i])
    u0_normalized = np.abs(eyetracker_data_u_normalized[i] - target_vector_data_u_normalized[i])
    v0_normalized = np.abs(eyetracker_data_v_normalized[i] - target_vector_data_v_normalized[i])
    u0 = np.abs(eyetracker_data_u[i] - target_vector_data_u[i])
    v0 = np.abs(eyetracker_data_v[i] - target_vector_data_v[i])

    x.append(x0)
    y.append(y0)
    u_normalized.append(u0_normalized)
    v_normalized.append(v0_normalized)
    u.append(u0)
    v.append(v0)

x = np.concatenate(x)
y = np.concatenate(y)
u_normalized = np.concatenate(u_normalized)
v_normalized = np.concatenate(v_normalized)
u = np.concatenate(u)
v = np.concatenate(v)

# ---------------------------------- Save data ---------------------------------------------------------------------- #
np.save("results/accuracy_x", x)
np.save("results/accuracy_y", y)
np.save("results/accuracy_u_normalized", u_normalized)
np.save("results/accuracy_z_normalized", v_normalized)
np.save("results/accuracy_u", u)
np.save("results/accuracy_v", v)

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

writer = pd.ExcelWriter('results/results.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1')
writer.save()
writer.close()
# ---------------------------------- Heat map ----------------------------------------------------------------------- #
plt.rcParams['figure.figsize'] = (16, 9)

midpoint_nearest = (eyetracker_screen_gray_nearest.max() - eyetracker_screen_gray_nearest.min())/1.5
heat_map_nearest = sb.heatmap(eyetracker_screen_gray_nearest, center=midpoint_nearest, vmin=0, vmax=1, xticklabels=False,
                              yticklabels=False, cmap="Blues", cbar=False)
plt.show()
figure_nearest = heat_map_nearest.get_figure().savefig('results/heat_map_nearest.png')

midpoint_cubic = (eyetracker_screen_gray.max() - eyetracker_screen_gray.min())/1.5
heat_map_cubic = sb.heatmap(eyetracker_screen_gray, center=midpoint_nearest, vmin=0, vmax=1, xticklabels=False,
                            yticklabels=False, cmap="Blues", cbar=False)
plt.show()
figure_cubic = heat_map_cubic.get_figure().savefig('results/heat_map_cubic.png') #dpi = 400





