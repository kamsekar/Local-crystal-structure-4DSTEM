# A script to perform preprocessing and k-means clustering on 4D-STEM data.
# Saves centroids and labels as .npz files and as images.
# ana.rebeka.kamsek@ki.si, 2023

import tifffile
from utils import log_data, normalize_data, reshape_data
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# the 4D-STEM dataset should be prepared as a .tif file
input_path = r"file_name.tif"
num_clusters = 2

# load data as a 3D numpy array
original_data = tifffile.imread(input_path)

# preprocess the data using utils functions
data = log_data(original_data)
data = normalize_data(data)
data = reshape_data(data)

# k-means clustering
clustering = KMeans(n_clusters=num_clusters, random_state=0)
clustered = clustering.fit(data)

labels = clustering.labels_
centroids = clustering.cluster_centers_

# prepare for visualization
dims_r = int(np.sqrt(labels.shape[0]))
dims_d = int(np.sqrt(centroids.shape[1]))

labels = np.reshape(labels, (dims_r, dims_r))
centroids = np.reshape(centroids, (num_clusters, dims_d, dims_d))

centroids_viz = centroids.clip(min=0)
minimum = np.amin(centroids_viz[np.nonzero(centroids_viz)])
centroids_viz[centroids_viz < minimum] = minimum

# save clustering results as .npz files
with open("centroids.npz", "wb") as f:
    np.savez(f, centroids)
with open("labels.npz", "wb") as g:
    np.savez(g, labels)

# save centroids as images and labels as a color-coded real-space image
for i in range(num_clusters):
    plt.imsave("centroid_" + str(i + 1) + ".png", centroids_viz[i, :], cmap='gray')
colors = matplotlib.colormaps['Paired']
plt.imsave("labels.png", labels, cmap=colors, vmin=0, vmax=11)
