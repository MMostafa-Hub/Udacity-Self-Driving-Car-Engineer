import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt


def calculate_mean_std(image_list):
    """
    calculate mean and std of image list
    args:
    - image_list [list[str]]: list of image paths
    returns:
    - mean [array]: 1x3 array of float, channel wise mean
    - std [array]: 1x3 array of float, channel wise std
    """
    means, stds = [], []
    for image in image_list:
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        means.append(img.mean(axis=(0, 1)))
        stds.append(img.std(axis=(0, 1)))

    return np.mean(means, axis=0), np.std(stds, axis=0)


def channel_histogram(image_list):
    """
    calculate channel wise pixel value
    args:
    - image_list [list[str]]: list of image paths
    """
    # IMPLEMENT THIS FUNCTION


if __name__ == "__main__":
    image_list = glob.glob(
        "E:/Udacity's Self-Driving Car Engineer/Udacity-Self-Driving-Car-Engineer/Computer Vision/Image Manipulation/workspace/data/images/*"
    )
    mean, std = calculate_mean_std(image_list)
    print(mean, std)
