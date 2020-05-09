import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import cv2

# ---------------------------------- Import measured data ----------------------------------------------------------- #
eyetracker_data = np.load("result_eyetracker_array.npy")
target_vector_data = np.load("target_and_measured_vector_array.npy")
eyetracker_screen_bgr = np.load("eyetracker_screen.npy")
eyetracker_screen_gray = cv2.cvtColor(eyetracker_screen_bgr, cv2.COLOR_BGR2GRAY)
eyetracker_screen_gray = eyetracker_screen_gray/eyetracker_screen_gray.max()
eyetracker_screen_gray = np.abs(eyetracker_screen_gray - 1)


eyetracker_data_x = eyetracker_data[0]
eyetracker_data_y = eyetracker_data[1]
eyetracker_data_u = eyetracker_data[2]
eyetracker_data_v = eyetracker_data[3]

target_vector_data_x = target_vector_data[0]
target_vector_data_y = target_vector_data[1]
target_vector_data_u = target_vector_data[2]
target_vector_data_v = target_vector_data[3]

x = []
y = []
u = []
v = []
for i in range(0, len(eyetracker_data[0])):

    x0 = np.abs(eyetracker_data_x[i] - target_vector_data_x[i])
    y0 = np.abs(eyetracker_data_y[i] - target_vector_data_y[i])
    u0 = np.abs(eyetracker_data_u[i] - target_vector_data_u[i])
    v0 = np.abs(eyetracker_data_v[i] - target_vector_data_v[i])
    x.append(x0)
    y.append(y0)
    u.append(u0)
    v.append(v0)

x = np.concatenate(x)
y = np.concatenate(y)
u = np.concatenate(u)
v = np.concatenate(v)

# ---------------------------------- Save data ---------------------------------------------------------------------- #
np.save("accuracy_x", x)
np.save("accuracy_y", y)
np.save("accuracy_u", u)
np.save("accuracy_v", v)

# ---------------------------------- Excel -------------------------------------------------------------------------- #
data = {'X': x,
        'Y': y,
        'U': u,
        'V': v}

df = pd.DataFrame(data, columns=['X', 'Y', 'U', 'V'])
writer = pd.ExcelWriter('results.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1')
writer.save()
writer.close()
# ---------------------------------- Heat map ----------------------------------------------------------------------- #
heat_map = sb.heatmap(eyetracker_screen_gray, xticklabels=False, yticklabels=False, cmap="Blues")
plt.show()




