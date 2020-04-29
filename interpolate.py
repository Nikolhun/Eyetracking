import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


def interpolation(lower_left_corner, middle_left_corner, upper_left_corner, middle_bottom_corner, middle,
                  middle_up_corner, lower_right_corner, middle_right_corner, upper_right_corner, screen_size):
 '''
 Interpolates vectors from measured corners. You need x, y, u and v of a vector.
 :param lower_left_corner: [x, y, u, v] of lower left corner
 :param middle_left_corner: [x, y, u, v] of middle left
 :param upper_left_corner: [x, y, u, v] of upper left corner
 :param middle_bottom_corner: [x, y, u, v] of middle bottom
 :param middle: [x, y, u, v] of middle
 :param middle_up_corner: [x, y, u, v] of middle top
 :param lower_right_corner: [x, y, u, v] of lower right corner
 :param middle_right_corner: [x, y, u, v] of middle right
 :param upper_right_corner: [x, y, u, v] of upper right corner
 :param screen_size: screensize in [x, y]
 :return:
 '''

 x = [0, 0, 0, int(screen_size[0]/2), int(screen_size[0]/2), int(screen_size[0]/2),
      screen_size[0], screen_size[0], screen_size[0]]
 y = [0, int(screen_size[1]/2), screen_size[1], 0, int(screen_size[1]/2), screen_size[1],
      0, int(screen_size[1]/2), screen_size[1]]
 u = [int(lower_left_corner[2]), int(middle_left_corner[2]), int(upper_left_corner[2]),
      int(middle_bottom_corner[2]), int(middle[2]), int(middle_up_corner[2]),
      int(lower_right_corner[2]), int(middle_right_corner[2]), int(upper_right_corner[2])]
 v = [int(lower_left_corner[3]), int(middle_left_corner[3]), int(upper_left_corner[3]),
      int(middle_bottom_corner[3]), int(middle[3]), int(middle_up_corner[3]),
      int(lower_right_corner[3]), int(middle_right_corner[3]), int(upper_right_corner[3])]

 plt.figure(1)
 plt.quiver(x, y, u, v)  # show measured vectors

 xx = np.linspace(min(x), max(x), screen_size[0])  # new x ax for interpolated data screen_size[0]
 yy = np.linspace(min(y), max(y), screen_size[1])  # new y ax for interpolated data screen_size[1]
 xx, yy = np.meshgrid(xx, yy)

 points = np.transpose(np.vstack((x, y)))
 u_interp = griddata(points, u, (xx, yy), method='cubic')  # interpolate u
 v_interp = griddata(points, v, (xx, yy), method='cubic')  # interpolate v

 plt.figure(2)
 plt.quiver(xx, yy, u_interp, v_interp)  # show interpolated vectors
 plt.show()

 u_interp_normalized = u_interp / u_interp.max()
 v_interp_normalized = v_interp / v_interp.max()

 print("u_interp", u_interp)
 print("u_interp_normalized", u_interp_normalized)

 plt.figure(3)
 plt.quiver(xx, yy, u_interp_normalized, v_interp_normalized)  # show interpolated vectors
 plt.show()

 return u_interp, v_interp  # uv_interp
