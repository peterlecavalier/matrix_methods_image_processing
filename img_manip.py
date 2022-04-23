from cmath import exp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
from scipy.signal import convolve2d, convolve
from cv2 import copyMakeBorder, BORDER_REFLECT

import math
import timeit



# convolution - edge detection, blurring

# rotation: [cos, -sin]
#           [sin, cos]

def rotation(im, theta):
    # im as a np array, theta in degrees
    theta = np.radians(theta)
    # Set up the rotation matrix to be applied to each pixel
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])

    y_max = im.shape[0] - 1
    x_max = im.shape[1] - 1

    bounds = []
    bounds.append(np.matmul(rot_mat, [0, 0]))
    bounds.append(np.matmul(rot_mat, [0, y_max]))
    bounds.append(np.matmul(rot_mat, [x_max, 0]))
    bounds.append(np.matmul(rot_mat, [x_max, y_max]))

    bounds = np.array(bounds)

    low_bounds = np.min(bounds, axis=0)
    high_bounds = np.max(bounds, axis=0)
    
    # Floor/ceiling the bounds to make sure out-of-bounds doesn't happen
    if low_bounds[0] < 0:
        left_shift = -int(np.floor(low_bounds[0]))
    else:
        left_shift = -int(np.ceil(low_bounds[0]))
    if low_bounds[1] < 0:
        up_shift = -int(np.floor(low_bounds[1]))
    else:
        up_shift = -int(np.ceil(low_bounds[1]))
    
    up_upper = high_bounds[1] - y_max
    if up_upper < 0:
        up_upper = int(np.floor(up_upper))
    else:
        up_upper = int(np.ceil(up_upper))
    left_upper = high_bounds[0] - x_max
    if left_upper < 0:
        left_upper = int(np.floor(left_upper))
    else:
        left_upper = int(np.ceil(left_upper))

    new_y_size = up_shift + up_upper + im.shape[0]
    new_x_size = left_shift + left_upper + im.shape[1]

    # Set up a new matrix with each entry being the new pixel placement
    new_px_mat = np.ones((new_y_size, new_x_size, im.shape[2])).astype(int) + 254

    #make an array
    xs, ys = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[0]))
    idxs = np.dstack((xs, ys)).reshape((xs.shape[0], xs.shape[1], 2))
    idxs = idxs.reshape((idxs.shape[0] * idxs.shape[1], 2))
    idxs = np.swapaxes(idxs, 0, 1)
    rot_idxs = np.matmul(rot_mat, idxs)
    rot_idxs[0] = np.clip(rot_idxs[0] + left_shift, 0, new_x_size - 1)
    rot_idxs[1] = np.clip(rot_idxs[1] + up_shift, 0, new_y_size - 1)
    rot_idxs = np.swapaxes(rot_idxs, 0, 1).astype(int)
    
    for y in range(len(im)):
        for x in range(len(im[y])):
            new_x, new_y = rot_idxs[y*len(im[y]) + x].astype(int)
            # shift the pixel values to the location on the new image
            # clip to avoid out-of-bounds indexing
            #new_px_mat[np.clip(new_y + up_shift, 0, new_y_size - 1), np.clip(new_x + left_shift, 0, new_x_size - 1)] = im[y, x]
            new_px_mat[new_y, new_x] = im[y, x]
    return new_px_mat

# Taken from https://stackoverflow.com/a/12201744
# Converts an rgb image into grayscale
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def gaussian_kernel(size=5):
    # radius of the kernel - used to conveniently get the right index
    radius = int(np.floor(size / 2))
    # sigma for gaussian distribution
    sigma = size/4.
    #coefficient to go in front of each exp
    coef = (1/(2*math.pi*sigma**2))

    # make an array
    # doing it this way will make matrix math a lot more efficient
    xs, ys = np.meshgrid(np.arange(-radius, radius+1), np.arange(-radius, radius+1))
    x2_plus_y2 = np.square(xs) + np.square(ys)
    # calculate the gaussian at each pixel
    kernel = coef*np.exp(-(x2_plus_y2)/(2*sigma**2))
    
    #normalize the kernel
    kernel = kernel / np.sum(kernel)
    return kernel

def box_kernel(size=5):
    # Creates a kernel for box blurring
    # Matrix has all matching values, with a sum of 1.
    kernel = np.ones((size, size))
    kernel = kernel / np.sum(kernel)
    return kernel

#Loading in images
im_filenames = ["./images/wall.JPG", "./images/wall_bottle.JPG", "./images/acne_scars.jpg", "./images/us_smiling.jpg"]
ims = []
for i in im_filenames:
    cur_im = np.array(Image.open(i)).astype(int)
    ims.append(cur_im)

wall_arr = ims[0]
bottle_arr = ims[1]
acne_arr = ims[2]
smile_arr = ims[3]


# Rotation
"""
wall_arr = rotation(wall_arr, 90)
bottle_arr = rotation(bottle_arr, 90)
"""
rot_wall_arr = np.rot90(wall_arr, 3)
rot_bottle_arr = np.rot90(bottle_arr, 3)

fig, axs = plt.subplots(2, 2)
axs[0][0].imshow(wall_arr)
axs[0][0].set_title("Original Wall image", fontsize=20)
axs[0][1].imshow(rot_wall_arr)
axs[0][1].set_title("Rotated Wall image", fontsize=20)
axs[1][0].imshow(bottle_arr)
axs[1][0].set_title("Original Bottle image", fontsize=20)
axs[1][1].imshow(rot_bottle_arr)
axs[1][1].set_title("Rotated Bottle image", fontsize=20)
# ------ END Rotation

# Subtraction
subtract = 255 - np.abs(np.subtract(rot_bottle_arr, rot_wall_arr))
subtract = np.clip(subtract, 0, 255)

fig, axs = plt.subplots(1, 3, sharey=True, sharex=True)
axs[0].imshow(rot_bottle_arr)
axs[1].imshow(rot_wall_arr)
axs[2].imshow(subtract)
axs[0].set_title("Bottle in front of wall", fontsize=20)
axs[1].set_title("Wall", fontsize=20)
axs[2].set_title("Absolute value of wall img\nsubtracted from bottle img.", fontsize=20)
# ------ END Subtraction

# Scalar Multiplication
darker_im = (0.4 * subtract).astype(int)

fig, axs = plt.subplots(1, 2, sharey=True, sharex=True)
axs[0].imshow(subtract)
axs[1].imshow(darker_im)
axs[0].set_title("Water bottle subtracted image", fontsize=20)
axs[1].set_title("Scalar multiplied image (x0.4)", fontsize=20)
# ------ END Scalar Multiplication

plt.show()



# Convolution

# Edge detection
smile_gray = rgb2gray(smile_arr)
kernel_size = 5
kernel_radius = int(np.floor(kernel_size / 2))
smooth_55_kernel = gaussian_kernel(kernel_size)


start = timeit.default_timer()

smooth_smile = convolve2d(smile_gray, smooth_55_kernel, boundary="symm")

stop = timeit.default_timer()

print(f"convolve2d: {stop - start}")


# Crop the image to remove padding during the convolution process
smooth_smile = smooth_smile[kernel_radius:-kernel_radius, kernel_radius:-kernel_radius]

edge = smile_gray - smooth_smile
edge = edge < -5

fig, axs = plt.subplots(1, 3, sharey=True, sharex=True)
axs[0].imshow(smile_gray, cmap="gray")
axs[1].imshow(smooth_smile, cmap="gray")
axs[2].imshow(edge, cmap="gray")
axs[0].set_title("Original image", fontsize=20)
axs[1].set_title("Gaussian smoothed image", fontsize=20)
axs[2].set_title("Edge detected image", fontsize=20)

plt.show()
# ------ END Edge detection


# Set the kernel size/radius
kernel_size = 61
kernel_radius = int(np.floor(kernel_size / 2))
# Generate the kernel (expand_dims adds an extra dimension for convolution)
smooth_55_kernel = np.expand_dims(gaussian_kernel(kernel_size), axis=-1)

# Focus on a specific portion of the image
smooth_acne = np.copy(acne_arr)
acne_crop = smooth_acne[200:1800, 150:2000]#[1300:1900, 775:1550]

# Add borders to the image for convolution, reflecting the borders
acne_crop = copyMakeBorder(acne_crop, kernel_radius, kernel_radius, kernel_radius, kernel_radius, BORDER_REFLECT)

start = timeit.default_timer()

smooth_acne_crop = convolve(acne_crop, smooth_55_kernel).astype(int)

stop = timeit.default_timer()

print(f"Convolution: {stop - start}")

smooth_acne_crop = smooth_acne_crop[kernel_size - 1:-kernel_size + 1, kernel_size - 1:-kernel_size + 1]

smooth_acne[200:1800, 150:2000] = smooth_acne_crop

rect = Rectangle((150,200),1850,1600, edgecolor='r', facecolor="none")

fig, axs = plt.subplots(1, 3, sharey=True, sharex=True)
axs[0].imshow(acne_arr)
axs[1].imshow(acne_arr)
axs[1].add_patch(rect)
axs[2].imshow(smooth_acne)

axs[0].set_title("Original acne image", fontsize=20)
axs[1].set_title("Box highlighting convolution area", fontsize=20)
axs[2].set_title("Gaussian smoothed acne image", fontsize=20)

plt.show()
