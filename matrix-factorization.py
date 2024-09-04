# A script to perform preprocessing and non-negative matrix factorization on 4D-STEM data.
# Saves the eigenvectors (diffraction patterns) and their loading maps as .npz files and as images.
# ana.rebeka.kamsek@ki.si, 2023

import tifffile
import numpy as np
import cv2
from utils import log_data, normalize_data, reshape_data
from sklearn import decomposition
import matplotlib.pyplot as plt
from matplotlib import colors

# define relevant paths
clustering_path = r"C:\folder_name\labels_file_name.npz"
input_path = r"C:\folder_name\file_name.tif"
output_path = r"C:\new_folder_name"

# define details for the analysis
cluster_id = 1  # cluster label, corresponding to the nanoparticle
morph_labels = False  # eliminate trailing cluster labels if needed
n_components = 7  # if 0, PCA is performed to estimate the initial number of components

# load data as a 3D numpy array
original_data = tifffile.imread(input_path)
original_shape = original_data.shape

scan_shape = (int(np.sqrt(original_shape[0])), int(np.sqrt(original_shape[0])))
input_shape = (original_shape[1], original_shape[2])

# use only the diffraction patterns, determined by a single cluster
labels = np.load(clustering_path)['arr_0']

labels = labels.astype("uint8")
if morph_labels:
    labels = cv2.morphologyEx(labels, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    labels = cv2.morphologyEx(labels, cv2.MORPH_ERODE, np.ones((3, 3), np.uint8))
labels = np.ndarray.flatten(labels)

chosen_dps = np.argwhere(labels == cluster_id)
labeled_data = original_data[chosen_dps[:, 0]]
data = labeled_data.copy()

# preprocess the data using utils functions
data = log_data(data)
data = normalize_data(data)
data = reshape_data(data)

if n_components == 0:
    # perform PCA to estimate the number of components
    pca = decomposition.PCA(n_components=100)
    pca_transform = pca.fit_transform(data)

    # cumulative variance plot including the estimated elbow point
    variance = np.cumsum(pca.explained_variance_ratio_)
    slope = np.diff(variance)

    ratio = np.divide(slope, variance[:-1])
    n_components = np.amax(np.argwhere(ratio > np.amax(slope) * 0.1))

    # save the cumulative explained variance plot
    plt.plot(variance)
    plt.vlines(n_components, np.amin(variance), np.amax(variance), linestyles='dashed')
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.savefig(output_path + "/" + "cumvar_PCA.png")
    plt.close()

# non-negative matrix factorization (NMF)
nmf = decomposition.NMF(n_components=n_components, max_iter=1000)
nmf_transform = nmf.fit_transform(data)

# in case those eigenvectors should then be used to construct their abundance maps for a different dataset:
# another_nmf_transform = nmf.transform(another_dataset)
# another_nmf_transform then replaces nmf_transform below

# constructing abundance maps and eigenvectors (endmembers)
endmembers = nmf.components_.reshape(n_components, input_shape[0], input_shape[1])
maps_new = np.zeros((scan_shape[0] * scan_shape[1], n_components))
maps_new[chosen_dps[:, 0]] = nmf_transform

temp_1 = np.reshape(maps_new, (scan_shape[0], scan_shape[1], n_components))
temp_2 = np.swapaxes(temp_1, 0, 2)
maps = np.swapaxes(temp_2, 1, 2)

# save endmembers and abundance maps in a .npz file
with open(output_path + "/" + "maps_NMF.npz", "wb") as f:
    np.savez(f, maps)
with open(output_path + "/" + "endmembers_NMF.npz", "wb") as g:
    np.savez(g, endmembers)

# plot the endmembers and maps, maps are on the same color scale, save the figure
fig2, axs2 = plt.subplots(2, maps.shape[0], figsize=(20, 6))
plt.rcParams.update({'font.size': 36})

images = []
for j in range(maps.shape[0]):
    images.append(axs2[0, j].imshow(endmembers[j], cmap='RdBu_r'))
    axs2[0, j].set_xticks([]), axs2[0, j].set_yticks([])

    images.append(axs2[1, j].imshow(maps[j], cmap='RdBu_r'))
    axs2[1, j].set_xticks([]), axs2[1, j].set_yticks([])

limit = np.amax(maps)  # or manually set a desired value
norm = colors.Normalize(vmin=0, vmax=limit)
[image.set_norm(norm) for image in images[maps.shape[0]:]]

fig2.colorbar(images[-1], ax=axs2, ticks=[0, 0.25, 0.5], orientation='vertical',
              fraction=.05, location='right', pad=0.05)

plt.savefig(output_path + "/maps_endmembers_NMF.png")
plt.savefig(output_path + "/maps_endmembers_NMF.svg")
plt.close()
