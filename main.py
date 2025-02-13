import os
import sys
import shutil
import cv2
import numpy as np
from pathlib import Path
from DeOldify.fastai.imports import torch
from evalColour import EvaluateColorMetrics
from evalUpscale import EvaluateUpscaleMetrics
import cv2
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from RRDBNet_arch import RRDBNet  # Ensure this file is accessible in your environment
from Plots import *

# Add the DeOldify directory to Python path
deoldify_path = os.path.abspath("DeOldify")
sys.path.append(deoldify_path)

from DeOldify.deoldify.visualize import get_video_colorizer


class VideoColorizationPipeline:
    def __init__(self, video_path, render_factor=21):
        self.video_path = video_path
        self.video_name = Path(video_path).stem  # Extract the video name without extension
        self.workfolder = Path('./video')
        self.source_dir = self.workfolder / "source"
        self.bwframes_dir = self.workfolder / "bwframes" / self.video_name
        self.colorframes_dir = self.workfolder / "colorframes" / self.video_name
        #self.colorized_video_output = Path("ColourisedVideo.mp4")
        self.render_factor = render_factor
        self.upscaled_frames_dir = self.workfolder / "upscaledframes" / self.video_name
        self.output_video_path =  f"{self.video_name}_UpscaledVideo.mp4"
        
        
        self._initialize_directories()
        self.model = self._initialize_model()

    def _initialize_directories(self):
        os.makedirs(self.bwframes_dir, exist_ok=True)
        os.makedirs(self.colorframes_dir, exist_ok=True)
        os.makedirs(self.source_dir, exist_ok=True)
        os.makedirs(self.upscaled_frames_dir, exist_ok=True)

        self.file_name = os.path.basename(sys.argv[1])

        # Destination path in ./video/source
        destination_path = Path('./video/source') / self.file_name

            # Copy the file to ./video/source
        try:
            shutil.copy(sys.argv[1], destination_path)
            print(f"Copied {sys.argv[1]} to {destination_path}")
        except FileNotFoundError:
            print(f"Error: File '{sys.argv[1]}' not found.")
            sys.exit(1)
        except Exception as e:
            print(f"Error copying file: {e}")
            sys.exit(1)


    # def extract_frames(self):
    #     """
    #     Extract frames from a video and save them as grayscale images.
    #     """
    #     print(f"Extracting frames from {self.video_path}...")
    #     cap = cv2.VideoCapture(self.video_path)
    #     frame_count = 0

    #     while cap.isOpened():
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         frame_path = self.bwframes_dir / f"frame_{frame_count:04d}.png"
    #         cv2.imwrite(str(frame_path), gray_frame)
    #         frame_count += 1

    #     cap.release()
    #     print(f"Extracted {frame_count} frames to {self.bwframes_dir}")

    def generate_color_video(self):
        """
        Colorize the grayscale frames and generate a colorized video.
        """
        print("Starting colorization process...")
        # file_name = os.path.basename(self.video_path)
        # destination_path = self.source_dir / file_name

        # try:
        #     shutil.copy(self.video_path, destination_path)
        #     print(f"Copied {self.video_path} to {destination_path}")
        # except Exception as e:
        #     print(f"Error copying file: {e}")
        #     sys.exit(1)

        colorizer = get_video_colorizer()
        colorizer.colorize_from_file_name(
            file_name=self.file_name,
            render_factor=self.render_factor,
            watermarked=True,
            post_process=True
        )
        print(f"Colorization completed. Output saved to {self.colorframes_dir}")

    def evaluate_colorization(self):
        """
        Evaluate the colorization quality.
        """
        print("Evaluating colorization...")
        print(self.bwframes_dir,self.colorframes_dir)
        EvaluateColorMetrics(self.bwframes_dir,self.colorframes_dir)
        print("Evaluation completed.")

    def evaluate_upscaling(self):

        EvaluateUpscaleMetrics(self.colorframes_dir,self.upscaled_frames_dir)



    def _initialize_model(self):
        """
        Initialize the RRDBNet model for upscaling.
        """
        from RRDBNet_arch import RRDBNet  # Import the RRDBNet architecture
        model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23)
        model_path = "./models/RRDB_ESRGAN_x4.pth"  # Replace with the actual path to the model weights

        # Load model weights
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            model.load_state_dict(checkpoint, strict=True)
            model.eval()  # Set model to evaluation mode
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            sys.exit(1)
        return model
    



    def upscale_image(self, image_path):
        """
        Upscales a single image using the RRDBNet model.
        """
        img = Image.open(image_path).convert("RGB")
        transform = transforms.ToTensor()
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            upscaled_tensor = self.model(img_tensor)

        # Convert tensor back to image
        upscaled_img = transforms.ToPILImage()(upscaled_tensor.squeeze())
        return upscaled_img
    

    def upscale_frames(self):
        """
        Loops through all frames and upscales them using the RRDBNet model.
        """
        print(f"Upscaling frames in {self.colorframes_dir}...")
        for frame_file in os.listdir(self.colorframes_dir):
            if ".png" in frame_file:
                continue 

            frame_path = os.path.join(self.colorframes_dir, frame_file)
            upscaled_img = self.upscale_image(frame_path)

            # Save the upscaled image
            upscaled_img.save(self.upscaled_frames_dir / frame_file)
            print(f"Upscaled and saved {frame_file}")
        print(f"Upscaled frames saved to {self.upscaled_frames_dir}")




    def combine_frames_to_video(self):
        """
        Combines all upscaled frames into a video.
        """
        print(f"Combining frames from {self.upscaled_frames_dir} into a video...")
        frame_files = sorted(
            [f for f in os.listdir(self.upscaled_frames_dir) if f.endswith(('.png', '.jpg'))]
        )
        if not frame_files:
            print("No frames found in the folder.")
            return

        # Read the first frame to get dimensions
        first_frame_path = os.path.join(self.upscaled_frames_dir, frame_files[0])
        first_frame = cv2.imread(first_frame_path)
        height, width, _ = first_frame.shape
        print(f"Frame dimensions: {width}x{height}")

        # Initialize the video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
        video_writer = cv2.VideoWriter(str(self.output_video_path), fourcc, self.frame_rate, (width, height))

        # Loop through all frames and write them to the video
        for frame_file in frame_files:
            frame_path = os.path.join(self.upscaled_frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            video_writer.write(frame)
            print(f"Added frame {frame_file} to the video.")

        # Release the video writer
        video_writer.release()
        print(f"Video saved to {self.output_video_path}")





# ----------------------- Main Pipeline -----------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python color_pipeline.py <path_to_video>")
        sys.exit(1)

    input_video_path = sys.argv[1]
    pipeline = VideoColorizationPipeline(video_path=input_video_path, render_factor=21)
    #pipeline.extract_frames()
    # pipeline.generate_color_video()
    # pipeline.evaluate_colorization()
    # pipeline.upscale_frames()
    # pipeline.combine_frames_to_video()
    # pipeline.evaluate_upscaling()
    # compare_histograms_with_subplots(pipeline.bwframes_dir, pipeline.colorframes_dir, pipeline.upscaled_frames_dir, "00001.jpg")
    compare_frame_resolutions(pipeline.bwframes_dir,pipeline.colorframes_dir ,  pipeline.upscaled_frames_dir, "00018.jpg")
