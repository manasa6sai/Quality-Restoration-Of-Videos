import os
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def EvaluateUpscaleMetrics(colourise_dir , upscaled_dir):

    # Initialize metrics
    psnr_values = []
    ssim_values = []

    # Process all files
    for frame_name in os.listdir(colourise_dir):
        original_path = os.path.join(colourise_dir, frame_name)
        upscaled_path = os.path.join(upscaled_dir, frame_name)

        # Check if both files exist
        if not os.path.exists(original_path) or not os.path.exists(upscaled_path):
            print(f"Frame missing: {frame_name}")
            continue


        if ".png" in frame_name:
            continue 

        # Read images
        original = cv2.imread(original_path)
        upscaled = cv2.imread(upscaled_path)

        # Resize original to match upscaled size (if necessary)
        original_resized = cv2.resize(original, (upscaled.shape[1], upscaled.shape[0]))

        # Debug dimensions
        print(f"Processing {frame_name}: Original resized shape = {original_resized.shape}, Upscaled shape = {upscaled.shape}")

        # Calculate PSNR
        psnr_value = psnr(original_resized, upscaled)

        # Calculate SSIM with explicit window size
        try:
            ssim_value, _ = ssim(original_resized, upscaled, channel_axis=-1, full=True, win_size=3)
        except ValueError as e:
            print(f"Skipping {frame_name} due to SSIM error: {e}")
            continue

        # Append results
        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)

        print(f"Processed {frame_name}: PSNR = {psnr_value:.2f} dB, SSIM = {ssim_value:.4f}")

    # Compute average metrics
    if psnr_values and ssim_values:
        avg_psnr = sum(psnr_values) / len(psnr_values)
        avg_ssim = sum(ssim_values) / len(ssim_values)
        print("\nSummary:")
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
    else:
        print("No valid frames processed.")
