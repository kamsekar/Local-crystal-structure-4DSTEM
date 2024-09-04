# Helper functions to process 4D-STEM data.
# ana.rebeka.kamsek@ki.si, 2023

import numpy as np
import cv2
import scipy.ndimage as nd


def binning_diffraction(input_data, bin_factor_d=1):
    """Bins the diffraction patterns by a set factor.

    :param input_data: data as a 3D array (navigation, detector_x, detector_y)
    :param bin_factor_d: binning factor (default is 1)
    :return: data, binned in diffraction space
    """

    # initialize the binned data array
    input_shape = input_data.shape
    data_binned_d = np.zeros((input_data.shape[0],
                              int(input_data.shape[1] // bin_factor_d),
                              int(input_data.shape[2] // bin_factor_d)))

    # compute the binned patterns one by one
    for i in range(input_shape):
        temp = cv2.resize(input_data[i, :, :], dsize=(input_shape[1] // bin_factor_d, input_shape[2] // bin_factor_d))
        data_binned_d[i, :, :] = temp

    return data_binned_d


def binning_real(input_data, bin_factor_r=1):
    """Bins the data in real space by a set factor.

    :param input_data: data as a 3D array (navigation, detector_x, detector_y)
    :param bin_factor_r: binning factor (default is 1)
    :return: data, binned in real space
    """

    # initialize the binned data array
    input_shape = input_data.shape

    # compute the binned data
    temp_shape = (int(np.sqrt(input_shape[0])), int(np.sqrt(input_shape[0])), input_shape[1] * input_shape[2])
    input_data = np.reshape(input_data, temp_shape)

    data_binned_r = np.zeros((int(temp_shape[0] // bin_factor_r), int(temp_shape[1] // bin_factor_r), temp_shape[2]))

    for i in range(temp_shape[-1]):
        temp = cv2.resize(input_data[:, :, i], dsize=(temp_shape[0] // bin_factor_r, temp_shape[0] // bin_factor_r))
        data_binned_r[:, :, i] = temp

    # return the data with the same axes as the input
    data_binned_r = np.reshape(data_binned_r, input_shape)

    return data_binned_r


def mask_central(input_data):
    """Sets the values within the central Bragg disks in the diffraction patterns to zero.

    :param input_data: data as a 3D array (navigation, detector_x, detector_y)
    :return: data with masked central disks
    """

    input_shape = input_data.shape

    # convert the data to a 4D array (navigation_x, navigation_y, detector_x, detector_y)
    input_data = np.reshape(input_data, (int(np.sqrt(input_shape[0])), int(np.sqrt(input_shape[0])),
                                         input_shape[1], input_shape[2]))

    # determine the position of the central disk by calculating the average pattern's center of mass
    ave_pattern = np.average(input_data, axis=(0, 1))
    center_xy = nd.center_of_mass(ave_pattern)

    # determine the radius of the central disk by segmenting and measuring it
    ret, central_disk = cv2.threshold(ave_pattern, np.amax(ave_pattern) * 0.5, np.amax(ave_pattern), cv2.THRESH_BINARY)
    central_disk = central_disk.astype('uint8')
    contours, hierarchy = cv2.findContours(central_disk, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    disk_area = cv2.contourArea(contours[0])
    disk_radius = np.sqrt(disk_area / np.pi)

    # creating a mask for the central disk
    x, y, = np.indices((ave_pattern.shape[0], ave_pattern.shape[1]))
    disk_mask = np.asarray((x - center_xy[0]) ** 2 + (y - center_xy[1]) ** 2 > disk_radius ** 2)

    # multiply all diffraction patterns by the mask
    masked_data = input_data[:, :] * disk_mask

    masked_data = np.reshape(masked_data, input_shape)

    return masked_data


def log_data(input_data):
    """Takes the natural logarithm of the data to enhance lower-intensity features.

    :param input_data: data as a 3D array (navigation, detector_x, detector_y)
    :return: log of the data
    """

    # all pixels with zero values are assigned the minimum non-zero value of the dataset
    minimum = np.amin(input_data[np.nonzero(input_data)])
    input_data[input_data < minimum] = minimum

    data = np.log(input_data)
    return data


def normalize_data(input_data):
    """Normalizes the values in the dataset to (0, 1).

    :param input_data: data as a 3D array (navigation, detector_x, detector_y)
    :return: data, normalized to (0, 1)
    """

    return (input_data - np.amin(input_data)) / (np.amax(input_data) - np.amin(input_data))


def reshape_data(input_data, flattened=False):
    """Takes the input data and returns it as a 2D array to pass it to appropriate algorithms.

    :param input_data: data as a 2D, 3D, or 4D array
    :param flattened: Boolean value to indicate a (navigation_x, navigation_y, detector)
    axes configuration (default is False)
    :return: reshaped data as (navigation, detector)
    :raises TypeError: if the input does not have one of the permitted shapes
    """

    input_shape = input_data.shape

    if len(input_shape) == 4:  # (navigation_x, navigation_y, detector_x, detector_y)
        data_reshaped = input_data.reshape(input_shape[0] * input_shape[1], input_shape[2] * input_shape[3])
    elif len(input_shape) == 3:
        if flattened:  # (navigation_x, navigation_y, detector)
            data_reshaped = input_data.reshape(input_shape[0] * input_shape[1], input_shape[2])
        else:  # (navigation, detector_x, detector_y)
            data_reshaped = input_data.reshape(input_shape[0], input_shape[1] * input_shape[2])
    elif len(input_shape) == 2:  # (navigation, detector)
        data_reshaped = input_data
    else:
        raise TypeError("Invalid input data shape. Data should be a 2D, 3D or 4D array.")

    data_reshaped = np.real(data_reshaped)
    return data_reshaped
