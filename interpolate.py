import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate


def interpolate(lower_left_corner, upper_left_corner, lower_right_corner, upper_right_corner):
 '''
 Interpolates vectors from measured corners. You need x, y, u and v of a vector.
 :param lower_left_corner: [x, y, u, v] of lower left corner
 :param upper_left_corner: [x, y, u, v] of upper left corner
 :param lower_right_corner: [x, y, u, v] of lower right corner
 :param upper_right_corner: [x, y, u, v] of upper right corner
 :return:
 '''

 x = [lower_left_corner[0], upper_left_corner[0], lower_right_corner[0], upper_right_corner[0]]
 y = [lower_left_corner[1], upper_left_corner[1], lower_right_corner[1], upper_right_corner[1]]
 u = [lower_left_corner[2], upper_left_corner[2], lower_right_corner[2], upper_right_corner[2]]
 v = [lower_left_corner[3], upper_left_corner[3], lower_right_corner[3], upper_right_corner[3]]

 plt.figure(1)
 plt.quiver(x, y, u, v)  # show measured vectors

 xx = np.linspace(min(x), max(x), 20)  # new x ax for interpolated data
 yy = np.linspace(min(y), max(y), 20)  # new y ax for interpolated data
 xx, yy = np.meshgrid(xx, yy)

 points = np.transpose(np.vstack((x, y)))
 u_interp = interpolate.griddata(points, u, (xx, yy), method='cubic')  # interpolate u
 v_interp = interpolate.griddata(points, v, (xx, yy), method='cubic')  # interpolate v

 plt.figure(2)
 plt.quiver(xx, yy, u_interp, v_interp)  # show interpolated vectors
 plt.show()

 print("u", u_interp)
 print("v", v_interp)

 return u_interp, v_interp

#ll = [-14, -1, 15, 85]
#ul = [-15, -4, 16, 74]
#lr = [5, 1, 6, -24]
#ur = [6, -6, 9, -45]
