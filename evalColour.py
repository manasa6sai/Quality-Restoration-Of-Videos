import os
import cv2
import numpy as np
#from skimage.metrics import niqe
import matplotlib.pyplot as plt



# Metrics calculation functions
def calculate_colorfulness(image):
    """
    Computes the colorfulness metric of an image.
    :param image: Input image as a NumPy array (RGB).
    :return: Colorfulness score.
    """
    rg = image[..., 0] - image[..., 1]
    yb = 0.5 * (image[..., 0] + image[..., 1]) - image[..., 2]
    sigma_rg, mean_rg = np.std(rg), np.mean(rg)
    sigma_yb, mean_yb = np.std(yb), np.mean(yb)
    return np.sqrt(sigma_rg ** 2 + sigma_yb ** 2) + 0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2)

def calculate_mean_saturation(image):
    """
    Computes the mean saturation of an image.
    :param image: Input image as a NumPy array (RGB).
    :return: Mean saturation value.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation_channel = hsv_image[..., 1]
    return np.mean(saturation_channel)


def EvaluateColorMetrics(grayscale_dir,colorized_dir):


    # Check if directories exist
    if not os.path.exists(grayscale_dir) or not os.path.exists(colorized_dir):
        print("Grayscale or colorized directory does not exist.")
        exit()

    # Initialize lists for metrics
    colorfulness_scores = []
    saturation_scores = []


    # Process frames
    for frame_name in os.listdir(colorized_dir):
        grayscale_path = os.path.join(grayscale_dir, frame_name)
        colorized_path = os.path.join(colorized_dir, frame_name)

        # Check if both files exist
        if not os.path.exists(grayscale_path) or not os.path.exists(colorized_path):
            print(f"Frame missing: {frame_name}")
            continue

        if ".png" in frame_name:
            continue 

        # Read the colorized frame
        colorized = cv2.imread(colorized_path)
        colorized = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)

        # Calculate metrics
        colorfulness = calculate_colorfulness(colorized)
        saturation = calculate_mean_saturation(colorized)
        #niqe_score = calculate_niqe(colorized)

        # Append metrics
        colorfulness_scores.append(colorfulness)
        saturation_scores.append(saturation)
        #niqe_scores.append(niqe_score)

        # Print per-frame metrics
        #print(f"Frame: {frame_name}")
        #print(f"  Colorfulness: {colorfulness:.2f}")
        #print(f"  Mean Saturation: {saturation:.2f}")
        #print(f"  NIQE: {niqe_score:.2f}\n")

    # Compute average metrics
    if colorfulness_scores and saturation_scores :
        avg_colorfulness = np.mean(colorfulness_scores)
        avg_saturation = np.mean(saturation_scores)

        print("\nSummary:")
        print(f"  Average Colorfulness: {avg_colorfulness:.2f}")
        print(f"  Average Saturation: {avg_saturation:.2f}")
    else:
        print("No valid frames processed.")


