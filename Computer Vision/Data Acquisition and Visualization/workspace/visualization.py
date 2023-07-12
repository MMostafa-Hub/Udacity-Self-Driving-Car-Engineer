from utils import get_data, ROOT_DIR
from matplotlib import pyplot as plt, patches
import numpy as np
import os


def get_rect(bbox, class_, color_map) -> patches.Rectangle:
    # get the coordinates
    # (x1, y1) is the top left corner
    # (x2, y2) is the bottom right corner
    y1, x1, y2, x2 = bbox

    # calculate the width and height of the rectangle
    # (x, y) is the bottom left corner of the rectangle when using matplotlib
    x = min(x1, x2)
    y = min(y1, y2)
    w = max(x1, x2) - x
    h = max(y1, y2) - y

    # create the rectangle to plot on the image
    rect = patches.Rectangle(
        (x, y), w, h, fill=False, edgecolor=color_map[class_], linewidth=1
    )

    return rect


def viz(ground_truth, predictions):
    """
    create a grid visualization of images with color coded bboxes
    args:
    - ground_truth [list[dict]]: ground truth data
    - predictions [list[dict]]: prediction data
    """
    color_map = {1: "red", 2: "blue"}
    # create a figure with 3x3 subplots
    fig, ax = plt.subplots(4, 4, figsize=(15, 15))
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
            # add the rectangle to the plot
            axi.add_patch(get_rect(bbox, class_, color_map))

            for pred in predictions:
                if pred["filename"] == gt["filename"]:
                    for bbox, class_ in zip(pred["boxes"], pred["classes"]):
                        # add the rectangle to the plot
                        rect = get_rect(bbox, class_, color_map)
                        rect.set_linestyle("dashed")
                        rect.set_linewidth(1)
                        axi.add_patch(rect)

        # set the title
        axi.set(title=None)
        # turn off the axis
        axi.axis("off")
    # display the plot
    plt.show()


if __name__ == "__main__":
    ground_truth, predictions = get_data()
    viz(ground_truth, predictions)
