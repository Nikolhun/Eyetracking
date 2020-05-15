import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import cv2
import ctypes

specification = "8x1_M_N_3_26"

user32 = ctypes.windll.user32  # for windows
screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)  # for windows

#screensize = (1280, 720) rpi
# ---------------------------------- Import measured data ----------------------------------------------------------- #
eyetracker_data = np.load('results/result_eyetracker_array.npy')
target_vector_data = np.load('results/target_and_measured_vector_array.npy')

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
        x0 = 0
        y0 = 0
        u0_normalized = 0
        v0_normalized = 0
        u0 = 0
        v0 = 0

    x.append(x0)
    y.append(y0)
    u_normalized.append(u0_normalized)
    v_normalized.append(v0_normalized)
    u.append(u0)
    v.append(v0)

x = np.array(x)
y = np.array(y)
u_normalized = np.array(u_normalized)
v_normalized = np.array(v_normalized)
u = np.array(u)
v = np.array(v)
u_normalized_percent = u_normalized*100
v_normalized_percent = v_normalized*100

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
        'U percent': u_normalized_percent,
        'V percent': v_normalized_percent}


df = pd.DataFrame(data, columns=['X', 'Y', 'U', 'V', 'U normalized', 'V normalized', 'U percent', 'V percent', '', '',
                                 'Mean', 'Max'])
writer = pd.ExcelWriter("results/" + specification + "_results.xlsx", engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1')

workbook = writer.book
worksheet = writer.sheets['Sheet1']
worksheet.set_column(1, 2, 7)
worksheet.set_column(3, 10, 12)

# Mean
worksheet.write(1, 11, round(x.mean(), 2))
worksheet.write(2, 11, round(y.mean(), 2))
worksheet.write(3, 11, round(u.mean(), 2))
worksheet.write(4, 11, round(v.mean(), 2))
worksheet.write(5, 11, round(u_normalized.mean(), 2))
worksheet.write(6, 11, round(v_normalized.mean(), 2))
worksheet.write(7, 11, round((u_normalized*100).mean(), 2))
worksheet.write(8, 11, round((v_normalized*100).mean(), 2))

# Max
worksheet.write(1, 12, round(x.max(), 2))
worksheet.write(2, 12, round(y.max(), 2))
worksheet.write(3, 12, round(u.max(), 2))
worksheet.write(4, 12, round(v.max(), 2))
worksheet.write(5, 12, round(u_normalized.max(), 2))
worksheet.write(6, 12, round(v_normalized.max(), 2))
worksheet.write(7, 12, round((u_normalized*100).max(), 2))
worksheet.write(8, 12, round((v_normalized*100).max(), 2))

# label
worksheet.write(1, 10, 'X')
worksheet.write(2, 10, 'Y')
worksheet.write(3, 10, 'U')
worksheet.write(4, 10, 'V')
worksheet.write(5, 10, 'U norm')
worksheet.write(6, 10, 'V norm')
worksheet.write(7, 10, 'U percent')
worksheet.write(8, 10, 'V percent')

writer.save()
writer.close()
# ---------------------------------- Loading arrays ---------------------------------------------------------------- #
target_save = np.load('results/eyetracker_target.npy')
mask_for_eyetracking = np.load('results/eyetracker_screen.npy')
cv2.imshow("mask", mask_for_eyetracking)
cv2.imshow("target", target_save)

eyetracker_screen_bgr_nearest = np.load('results/eyetracker_screen_nearest.npy')
eyetracker_screen_bgr = np.load('results/eyetracker_screen.npy')
# ---------------------------------- counting intepolation ----------------------------------------------------------- #
# for nearest target
target_save_nearest = cv2.resize(target_save, screensize,
                                           interpolation=cv2.INTER_NEAREST)
target_save_nearest_bgr = cv2.cvtColor(target_save_nearest, cv2.COLOR_BGR2GRAY)
target_save_nearest_bgr = target_save_nearest_bgr / target_save_nearest_bgr.max()
target_save_nearest_bgr = np.abs(target_save_nearest_bgr - 1)
#target_save_nearest_bgr = target_save_nearest_bgr*target_save_nearest_bgr

# for cubic target
target_save_cubic = cv2.resize(target_save, screensize,
                                           interpolation=cv2.INTER_CUBIC)
target_save_cubic_bgr = cv2.cvtColor(target_save_cubic, cv2.COLOR_BGR2GRAY)
target_save_cubic_bgr = target_save_cubic_bgr / target_save_cubic_bgr.max()
target_save_cubic_bgr = np.abs(target_save_cubic_bgr - 1)
#target_save_cubic_bgr = target_save_cubic_bgr*target_save_cubic_bgr

# for nearest eyetracker
eyetracker_screen_gray_nearest = cv2.cvtColor(eyetracker_screen_bgr_nearest, cv2.COLOR_BGR2GRAY)
eyetracker_screen_gray_nearest = eyetracker_screen_gray_nearest / eyetracker_screen_gray_nearest.max()
eyetracker_screen_gray_nearest = np.abs(eyetracker_screen_gray_nearest - 1)
eyetracker_screen_gray_nearest = eyetracker_screen_gray_nearest * eyetracker_screen_gray_nearest

# for cubic eyetracker
eyetracker_screen_bgr = cv2.resize(eyetracker_screen_bgr, screensize,
                                           interpolation=cv2.INTER_CUBIC)
eyetracker_screen_gray = cv2.cvtColor(eyetracker_screen_bgr, cv2.COLOR_BGR2GRAY)
eyetracker_screen_gray = eyetracker_screen_gray / eyetracker_screen_gray.max()
eyetracker_screen_gray = np.abs(eyetracker_screen_gray - 1)
eyetracker_screen_gray = eyetracker_screen_gray*eyetracker_screen_gray
# ---------------------------------- Heat map ---------------------------------------------------------------- #
plt.rcParams['figure.figsize'] = (16, 9)
# heatmap for nearest target
midpoint_nearest = (target_save_nearest_bgr.max() - target_save_nearest_bgr.min())/1.5
heat_map_nearest = sb.heatmap(target_save_nearest_bgr, center=midpoint_nearest, vmin=0, vmax=1, xticklabels=False,
                              yticklabels=False, cmap="Blues", cbar=False)
plt.show()
figure_nearest = heat_map_nearest.get_figure().savefig("results/" + specification + "_heat_map_nearest_target.png")

# heatmap for cubic target
midpoint_nearest = (target_save_cubic_bgr.max() - target_save_cubic_bgr.min())/1.5
heat_map_nearest = sb.heatmap(target_save_cubic_bgr, center=midpoint_nearest, vmin=0, vmax=1, xticklabels=False,
                              yticklabels=False, cmap="Blues", cbar=False)
plt.show()
figure_nearest = heat_map_nearest.get_figure().savefig("results/" + specification + "_heat_map_cubic_target.png")

# heatmap for nearest eyetracker
midpoint_nearest = (eyetracker_screen_gray_nearest.max() - eyetracker_screen_gray_nearest.min())/1.5
heat_map_nearest = sb.heatmap(eyetracker_screen_gray_nearest, center=midpoint_nearest, vmin=0, vmax=1, xticklabels=False,
                              yticklabels=False, cmap="Blues", cbar=False)
plt.show()
figure_nearest = heat_map_nearest.get_figure().savefig("results/" + specification + "_heat_map_nearest.png")

# heatmap for cubic eyetracker
midpoint_cubic = (eyetracker_screen_gray.max() - eyetracker_screen_gray.min())/1.5
heat_map_cubic = sb.heatmap(eyetracker_screen_gray, center=midpoint_cubic, vmin=0, vmax=1, xticklabels=False,
                            yticklabels=False, cmap="Blues", cbar=False)
plt.show()
figure_cubic = heat_map_cubic.get_figure().savefig("results/" + specification + "_heat_map_cubic.png") #dpi = 400

print("In this measurement was", unfined_vectors, "not found vectors.")






