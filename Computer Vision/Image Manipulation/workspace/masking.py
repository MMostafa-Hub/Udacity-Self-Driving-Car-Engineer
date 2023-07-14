from glob import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple, List


def create_mask(
    image_path: str, color_threshold: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    create a binary mask of an image using a color threshold
    args:
    - path [str]: path to image file
    - color_threshold [array]: 1x3 array of RGB value
    returns:
    - img [array]: RGB image array
    - mask [array]: binary array
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.inRange(img, np.array(color_threshold), np.array([255, 255, 255]))

    return img, mask


def mask_and_display(img, mask) -> None:
    """
    display 3 plots next to each other: image, mask and masked image
    args:
    - img [array]: HxWxC image array
    - mask [array]: HxW mask array
    """

    figure, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    ax1.imshow(img)
    ax2.imshow(mask, cmap="gray")
    ax3.imshow(cv2.bitwise_and(img, img, mask=mask))

    plt.show()


if __name__ == "__main__":
    path = "E:/Udacity's Self-Driving Car Engineer/Udacity-Self-Driving-Car-Engineer/Computer Vision/Image Manipulation/workspace/data/images/segment-1231623110026745648_480_000_500_000_with_camera_labels_38.png"
    color_threshold = [128, 128, 128]
    img, mask = create_mask(path, color_threshold)
    # print(img.shape, mask.shape)
    mask_and_display(img, mask)
