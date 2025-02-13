import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def compute_grayscale_histogram(image_path):
    """
    Computes the histogram for a grayscale image.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist

def compute_color_histogram(image_path):
    """
    Computes the histogram for a color image (RGB channels).
    """
    image = cv2.imread(image_path)
    color = ('b', 'g', 'r')  # OpenCV uses BGR order
    histograms = {}

    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        histograms[col] = hist

    return histograms

def plot_histograms_with_subplots(grayscale_hist, colorized_hist, upscaled_hist, frame_name):
    """
    Plots the histograms for the grayscale, colorized, and upscaled images in separate subplots.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

    # Grayscale (original)
    axes[0].plot(grayscale_hist, color='gray', label='Original (Grayscale)', linestyle='dashed')
    axes[0].set_title("Grayscale (Original)")
    axes[0].set_xlabel('Pixel Intensity')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True)
    axes[0].legend()

    # RGB Channels
    colors = ['b', 'g', 'r']
    channel_titles = ['Blue Channel', 'Green Channel', 'Red Channel']
    for i, col in enumerate(colors):
        axes[i + 1].plot(colorized_hist[col], color=col, linestyle='solid', label=f'Colorized - {col.upper()}')
        axes[i + 1].plot(upscaled_hist[col], color=col, linestyle='dotted', label=f'Upscaled - {col.upper()}')
        axes[i + 1].set_title(channel_titles[i])
        axes[i + 1].set_xlabel('Pixel Intensity')
        axes[i + 1].grid(True)
        axes[i + 1].legend()

    plt.suptitle(f'Pixel Intensity Histogram Comparison for {frame_name}', fontsize=16)
    plt.tight_layout()
    plot_file_path = os.path.join("./video/result/", f"{frame_name}_comparison_plot.png")
    plt.savefig(plot_file_path)

    plt.show()

def compare_histograms_with_subplots(original_dir, colorized_dir, upscaled_dir, frame_name):
    """
    Compares histograms of original (grayscale), colorized, and upscaled frames with subplots.
    """
    original_path = os.path.join(original_dir, frame_name)
    colorized_path = os.path.join(colorized_dir, frame_name)
    upscaled_path = os.path.join(upscaled_dir, frame_name)

    # Compute histograms
    grayscale_hist = compute_grayscale_histogram(original_path)
    colorized_hist = compute_color_histogram(colorized_path)
    upscaled_hist = compute_color_histogram(upscaled_path)

    # Plot histograms
    plot_histograms_with_subplots(grayscale_hist, colorized_hist, upscaled_hist, frame_name)



import os
import cv2
import matplotlib.pyplot as plt

def get_frame_resolution(image_path):
    """
    Retrieves the resolution (width, height) of the given image.
    """
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    return width, height

def plot_resolution_comparison(original_resolution, colorized_resolution, upscaled_resolution, frame_name):
    """
    Plots a bar chart comparing the resolutions of original, colorized, and upscaled frames.
    """
    labels = ['Original Resolution', 'Colorized Resolution', 'Upscaled Resolution']
    resolutions = [original_resolution, colorized_resolution, upscaled_resolution]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, [res[0] * res[1] for res in resolutions], color=['gray', 'blue', 'orange'], alpha=0.7)
    plt.title(f'Resolution Comparison for {frame_name}', fontsize=16)
    plt.ylabel('Total Pixels (Width x Height)', fontsize=14)
    plt.xlabel('Frame Type', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save the plot
    plot_file_path = os.path.join("./video/result/", f"{frame_name}_resolution_comparison.png")
    plt.savefig(plot_file_path)
    print(f"Resolution comparison plot saved to {plot_file_path}")

    # Show the plot
    plt.show()

def compare_frame_resolutions(original_dir, colorized_dir, upscaled_dir, frame_name):
    """
    Compares the resolutions of the original, colorized, and upscaled frames and plots the comparison.
    """
    original_path = os.path.join(original_dir, frame_name)
    colorized_path = os.path.join(colorized_dir, frame_name)
    upscaled_path = os.path.join(upscaled_dir, frame_name)

    # Get resolutions
    original_resolution = get_frame_resolution(original_path)
    colorized_resolution = get_frame_resolution(colorized_path)
    upscaled_resolution = get_frame_resolution(upscaled_path)

    # Plot resolution comparison
    plot_resolution_comparison(original_resolution, colorized_resolution, upscaled_resolution, frame_name)

















