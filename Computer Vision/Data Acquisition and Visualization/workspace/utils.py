import json
import os

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))


def get_data():
    """simple wrapper function to get data"""
    with open(os.path.join(ROOT_DIR, "workspace", "data", "ground_truth.json")) as f:
        ground_truth = json.load(f)

    with open(os.path.join(ROOT_DIR, "workspace", "data", "predictions.json")) as f:
        predictions = json.load(f)

    return ground_truth, predictions
