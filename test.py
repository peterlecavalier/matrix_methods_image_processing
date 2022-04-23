import numpy as np

im = np.zeros((3024, 4032, 3))
y_max = im.shape[0] - 1
x_max = im.shape[1] - 1

theta = np.radians(25)
rot_mat = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])

bounds = []
bounds.append(np.matmul(rot_mat, [0, 0]))
bounds.append(np.matmul(rot_mat, [0, y_max]))
bounds.append(np.matmul(rot_mat, [x_max, 0]))
bounds.append(np.matmul(rot_mat, [x_max, y_max]))

bounds = np.array(bounds)

print(bounds)

low_bounds = np.min(bounds, axis=0)
high_bounds = np.max(bounds, axis=0)

print(low_bounds, high_bounds)