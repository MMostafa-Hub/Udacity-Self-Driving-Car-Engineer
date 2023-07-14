import glob

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image, ImageStat
import cv2
from utils import check_results


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

    return np.mean(means, axis=0), np.mean(stds, axis=0)


def calculate_mean_std_pil(image_list):
    """
    calculate mean and std of image list
    args:
    - image_list [list[str]]: list of image paths
    returns:
    - mean [array]: 1x3 array of float, channel wise mean
    - std [array]: 1x3 array of float, channel wise std
    """
    means = []
    stds = []
    for path in image_list:
        img = Image.open(path).convert("RGB")
        stat = ImageStat.Stat(img)
        means.append(np.array(stat.mean))
        stds.append(np.array(stat.var) ** 0.5)

    total_mean = np.mean(means, axis=0)
    total_std = np.mean(stds, axis=0)

    return total_mean, total_std


def channel_histogram(image_list):
    """
    calculate channel wise pixel value
    args:
    - image_list [list[str]]: list of image paths
    """
    red = []
    green = []
    blue = []
    for path in image_list:
        img = np.array(Image.open(path).convert("RGB"))
        R, G, B = img[..., 0], img[..., 1], img[..., 2]
        red.extend(R.flatten().tolist())
        green.extend(G.flatten().tolist())
        blue.extend(B.flatten().tolist())

    plt.figure()
    sns.kdeplot(red, color="r")
    sns.kdeplot(green, color="g")
    sns.kdeplot(blue, color="b")
    plt.show()


if __name__ == "__main__":
    image_list = glob.glob(
        "E:/Udacity's Self-Driving Car Engineer/Udacity-Self-Driving-Car-Engineer/Computer Vision/Image Manipulation/workspace/data/images/*"
    )
    mean, std = calculate_mean_std(image_list)
    # channel_histogram(image_list[:2])
    check_results(mean, std)
