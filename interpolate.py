import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.interpolate import griddata


def interpolation(lower_left_corner, upper_left_corner, middle, lower_right_corner, upper_right_corner, screen_size):
 '''
 Interpolates vectors from measured corners. You need x, y, u and v of a vector.
 :param lower_left_corner: [x, y, u, v] of lower left corner
 :param upper_left_corner: [x, y, u, v] of upper left corner
 :param middle: [x, y, u, v] of middle
 :param lower_right_corner: [x, y, u, v] of lower right corner
 :param upper_right_corner: [x, y, u, v] of upper right corner
 :param screen_size: screensize in [x, y]
 :return:
 '''

 #x = [0, 0, int(screen_size[0] / 2), screen_size[0], screen_size[0]]
 x = [0, 0, int(32/ 2), 32, 32]

 #y = [0, screen_size[1], int(screen_size[1] / 2), 0, screen_size[1]]
 y = [0, 18, int(18 / 2), 0, 18]

 u = [int(lower_left_corner[2]), int(upper_left_corner[2]), int(middle[2]), int(lower_right_corner[2]), int(upper_right_corner[2])]
 v = [int(lower_left_corner[3]), int(upper_left_corner[3]), int(middle[3]), int(lower_right_corner[3]), int(upper_right_corner[3])]

 #ll = math.sqrt(lower_left_corner[2] * lower_left_corner[2] + lower_left_corner[3] * lower_left_corner[3])
 #up = math.sqrt(upper_left_corner[2] * upper_left_corner[2] + upper_left_corner[3] * upper_left_corner[3])
 #m = math.sqrt(middle[2] * middle[2] + middle[3] * middle[3])
 #lr = math.sqrt(lower_right_corner[2] * lower_right_corner[2] + lower_right_corner[3] * lower_right_corner[3])
 #ur = math.sqrt(upper_right_corner[2] * upper_right_corner[2] + upper_right_corner[3] * upper_right_corner[3])
 #uv = [ll, up, m, lr, ur]

 plt.figure(1)
 plt.quiver(x, y, u, v)  # show measured vectors

 xx = np.linspace(min(x), max(x), screen_size[0])  # new x ax for interpolated data screen_size[0]
 yy = np.linspace(min(y), max(y), screen_size[1])  # new y ax for interpolated data screen_size[1]
 xx, yy = np.meshgrid(xx, yy)

 points = np.transpose(np.vstack((x, y)))
 u_interp = griddata(points, u, (xx, yy), method='cubic')  # interpolate u
 v_interp = griddata(points, v, (xx, yy), method='cubic')  # interpolate v
# uv_interp = griddata(points, uv, (xx, yy), method='cubic')  # interpolate u

 plt.figure(2)
 plt.quiver(xx, yy, u_interp, v_interp)  # show interpolated vectors
 plt.show()

 return u_interp, v_interp  # uv_interp
