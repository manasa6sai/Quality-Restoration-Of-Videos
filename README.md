# 🎨 CV Restoration Pipeline: Video Colorization & Super-Resolution

This project restores old or degraded videos by combining **deep learning-based colorization** and **super-resolution upscaling** techniques. Using DeOldify and ESRGAN (RRDBNet), it processes grayscale or low-resolution videos into vivid, high-resolution outputs.

---

## 🧠 What It Does

- 📼 Takes a grayscale or faded video as input  
- 🖼️ Extracts and colorizes video frames using DeOldify  
- 🔍 Upscales the colorized frames using RRDBNet (ESRGAN)  
- 🎞️ Recombines enhanced frames into a high-resolution color video  
- 📊 Evaluates the quality of both colorization and upscaling

---

## 📁 Project Structure

```bash
├── color_pipeline.py           # Main pipeline
├── evalColour.py               # Evaluation script for colorization
├── evalUpscale.py              # Evaluation script for upscaling
├── RRDBNet_arch.py             # ESRGAN RRDBNet model definition
├── Plots.py                    # Visualization and comparison plots
├── DeOldify/                   # DeOldify repo (as subfolder)
│   └── ...                     # Includes visualize.py and weights
├── models/
│   └── RRDB_ESRGAN_x4.pth      # Pretrained weights for super-resolution
└── video/
    ├── source/                 # Input videos
    ├── bwframes/               # Grayscale frames
    ├── colorframes/            # Colorized frames
    ├── upscaledframes/         # Final high-res frames

---

## Requirements
Python 3.7+

PyTorch

OpenCV

PIL (Pillow)

torchvision

DeOldify (cloned in local directory)



---

## 🚀 How to Run
1. Clone the Repository

git clone https://github.com/manasa6sai/Quality-Restoration-Of-Videos
cd Quality-Restoration-Of-Videos

2. Prepare Environment
Make sure DeOldify is placed in the folder structure as shown, and download the ESRGAN weights into models/RRDB_ESRGAN_x4.pth.

3. Run the Pipeline
python main.py path/to/your_video.mp4

## 🧪 Evaluation Metrics
Colorization Evaluation: Compares grayscale and colorized frames using custom metrics in evalColour.py

Upscaling Evaluation: Assesses resolution gain and visual quality using evalUpscale.py


## 🙏 Acknowledgments
DeOldify

ESRGAN by xinntao

FastAI

