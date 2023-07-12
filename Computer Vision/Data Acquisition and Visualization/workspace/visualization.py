from utils import get_data, ROOT_DIR
from matplotlib import pyplot as plt, patches
import numpy as np
import os


def viz(ground_truth):
    """
    create a grid visualization of images with color coded bboxes
    args:
    - ground_truth [list[dict]]: ground truth data
    """
    color_map = {1: "red", 2: "blue"}
    # create a figure with 3x3 subplots
    fig, ax = plt.subplots(3, 3, figsize=(20, 20))
    # loop over the subplots and ground truth data
    for axi, gt in zip(np.ndarray.flatten(ax), ground_truth):
        # get the image
        image = plt.imread(
            os.path.join(
                ROOT_DIR,
                "workspace",
                "data",
                "images",
                gt["filename"],
            )
        )
        # plot the image on the subplot
        axi.imshow(image)
        # loop over the bboxes
        for bbox, class_ in zip(gt["boxes"], gt["classes"]):
            # get the coordinates
            y1, x1, y2, x2 = bbox

            # calculate the width and height of the rectangle
            x = min(x1, x2)  # left
            y = min(y1, y2)  # top
            w = max(x1, x2) - x
            h = max(y1, y2) - y

            # plot the rectangle on the image
            rect = patches.Rectangle(
                (x, y), w, h, fill=False, edgecolor=color_map[class_], linewidth=1
            )
            axi.add_patch(rect)
        # set the title
        axi.set(title=None)
        # turn off the axis
        axi.axis("off")
    # display the plot
    plt.show()


if __name__ == "__main__":
    ground_truth, _ = get_data()
    viz(ground_truth)
