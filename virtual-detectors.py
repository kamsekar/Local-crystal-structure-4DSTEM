# A script to calculate virtual bright-field and dark-field images from 4D-STEM data.
# ana.rebeka.kamsek@ki.si, 2023

import numpy as np
import matplotlib.pyplot as plt
import tifffile


def get_coordinates(input_image):
    """
    Displays an image and records the (x, y) data after the user clicks on it.
    :param input_image: image to be displayed
    :return: coordinates of clicked points
    """
    coords = []

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.imshow(input_image, cmap='gray'), plt.xticks([]), plt.yticks([])

    def onclick(event):
        coords.append((event.xdata, event.ydata))
        ax1.plot(event.xdata, event.ydata, 'o', color='r')
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show()

    return np.array(coords)


input_path = r"C:\folder_name\file_name.tif"
output_path = r"C:\new_folder_name"

# import a tiff file
dataset = tifffile.imread(input_path)

print("Shape of dataset: ", dataset.shape)
# the target shape includes two navigation dimensions and two detector dimensions
if len(dataset.shape) == 3:
    dataset = np.reshape(dataset, (int(np.sqrt(dataset.shape[0])), int(np.sqrt(dataset.shape[0])),
                                   dataset.shape[1], dataset.shape[2]))

# optional cropping of diffraction patterns since almost all of the signal is in the middle
dataset = (lambda data, margin=0: data[:, :, margin:data.shape[-2] - margin, margin:data.shape[-1] - margin])(dataset)

# a standard image as an average of diffraction space
ave_image = np.average(dataset, axis=(2, 3))
# an average diffraction pattern as an average of real space
ave_dp = np.average(dataset, axis=(0, 1))

# use the logarithm values of the average diffraction pattern to enhance lower-intensity features
minimum = np.amin(ave_dp[np.nonzero(ave_dp)])
ave_dp[ave_dp < minimum] = minimum
ave_dp = np.log(ave_dp)

# finding center of the DP by manually clicking on it
print("First, click on the center of the diffraction pattern. \n"
      "Then, click on the edge of the center disk and close the window.")
coordinates = get_coordinates(ave_dp)
center_y, center_x = np.asarray((coordinates[0, 0], coordinates[0, 1]))
edge_y, edge_x = np.asarray((coordinates[1, 0], coordinates[1, 1]))

# radius of the bright-field mask is determined by the chosen points
radius_BF = np.sqrt((center_y - edge_y) ** 2 + (center_x - edge_x) ** 2)

# create a disk mask for the bright-field image and initialize the image with zeros
x, y, = np.indices((ave_dp.shape[0], ave_dp.shape[1]))
mask_BF = np.asarray((x - center_x) ** 2 + (y - center_y) ** 2 < radius_BF ** 2)
bf_image = np.zeros((ave_image.shape[0], ave_image.shape[1]))

# same with an annular mask for the dark-field image
mask_inner = (x - center_x) ** 2 + (y - center_y) ** 2 > radius_BF ** 2
mask_outer = (x - center_x) ** 2 + (y - center_y) ** 2 < (10 * radius_BF) ** 2
mask_DF = np.logical_and(mask_inner, mask_outer)
df_image = np.zeros((ave_image.shape[0], ave_image.shape[1]))

# create virtual images
for i in range(ave_dp.shape[0]):
    for j in range(ave_dp.shape[1]):
        if mask_BF[i, j]:
            bf_image += dataset[:, :, i, j]
        if mask_DF[i, j]:
            df_image += dataset[:, :, i, j]

# plot and save the results
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
# average image and diffraction pattern
axs[0, 0].imshow(ave_image, cmap='gray')
axs[1, 0].imshow(ave_dp, cmap='gray')
# virtual bright-field image and the corresponding virtual detector
axs[0, 1].imshow(bf_image, cmap='gray')
axs[1, 1].imshow(ave_dp, cmap='gray')
axs[1, 1].imshow(mask_BF, alpha=0.5, cmap='Reds')
# virtual dark-field image and the corresponding virtual detector
axs[0, 2].imshow(df_image, cmap='gray')
axs[1, 2].imshow(ave_dp, cmap='gray')
axs[1, 2].imshow(mask_DF, alpha=0.5, cmap='Reds')
for ax in axs.flat:
    ax.set_xticks([]), ax.set_yticks([])

plt.savefig(output_path + "/virtual-detectors.png")
plt.savefig(output_path + "/virtual-detectors.svg")
plt.close()
