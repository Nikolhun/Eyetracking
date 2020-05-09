import numpy as np
import pandas as pd

x = []
y = []
u = []
v = []

# load maeasured data
eyetracker_data = np.load("result_eyetracker_array.npy")
target_vector_data = np.load("target_and_measured_vector_array.npy")
eyetracker_data_x = eyetracker_data[0]
target_vector_data_x = target_vector_data[0]
#eyetracker_data_y = eyetracker_data[1]
#eyetracker_data_u = eyetracker_data[2]
#eyetracker_data_v = eyetracker_data[3]

for i in range(0, len(eyetracker_data[0])):

    x0 = np.abs(eyetracker_data_x[i] - target_vector_data_x[i])
    #y0 = np.abs(eyetracker_data[1] - target_vector_data[1])
    #u0 = np.abs(eyetracker_data[2] - target_vector_data[2])
    #v0 = np.abs(eyetracker_data[3] - target_vector_data[3])
    x.append(x0)
    #y.append(y0)
    #u.append(u0)
    #v.append(v0)

np.save("accuracy_x", x)
#np.save("accuracy_y", y)
#np.save("accuracy_u", u)
#np.save("accuracy_v", v)

# Create a Pandas dataframe from the data.
df = pd.DataFrame({'X': x})

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('results.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
df.to_excel(writer, sheet_name='Sheet1')

# Close the Pandas Excel writer and output the Excel file.
writer.save()
