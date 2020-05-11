import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

data = np.random.rand(4, 6)
heat_map = sb.heatmap(data, xticklabels=False, yticklabels=False)
plt.show()


