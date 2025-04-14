# 🔬 SAR-RARP50-seg

<div align="center">

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.11-brightgreen)
![YOLOv8](https://img.shields.io/badge/model-YOLOv8--xlarge-orange)
![License](https://img.shields.io/badge/license-MIT-green)

**Advanced Semantic Segmentation for Robotic Surgery Imagery**
</div>

<p align="center">
  <img src="https://img.shields.io/badge/%F0%9F%A4%96%20Powered%20by-YOLOv8-yellow" alt="Powered by YOLOv8">
</p>

## 📋 Overview

This project leverages the cutting-edge YOLOv8 segmentation model (specifically the `xlarge` variant) for high-precision semantic segmentation tasks on the SAR-RARP50 dataset, enabling accurate identification and delineation of surgical instruments and anatomical structures.

## ⚙️ Installation

<details>
<summary><b>📥 Setup Instructions</b></summary>

```bash
# Step 1: Clone the repository
git clone https://github.com/MahdiMohseni0033/SAR-RARP50-seg.git

# Step 2: Navigate to project directory
cd SAR-RARP50-seg

# Step 3: Create and activate Conda environment
conda create -n yolo-sar python=3.11 -y
conda activate yolo-sar

# Step 4: Install dependencies
pip install -r requirements.txt
```

</details>

## 🗃️ Dataset Preparation

<table>
  <tr>
    <th width="50%">Training Dataset</th>
    <th width="50%">Testing Dataset</th>
  </tr>
  <tr>
    <td>
      <pre><code>bash prepare_data.sh train</code></pre>
      <ul>
        <li>Downloads the training dataset</li>
        <li>Processes and splits into training/validation sets</li>
        <li>Converts to YOLO format</li>
        <li>Stores in <code>yolo_dataset2</code> directory</li>
      </ul>
    </td>
    <td>
      <pre><code>bash prepare_data.sh test</code></pre>
      <ul>
        <li>Downloads the testing dataset</li>
        <li>Processes the data</li>
        <li>Converts to YOLO format</li>
        <li>Stores in <code>yolo_test_set</code> directory</li>
      </ul>
    </td>
  </tr>
</table>

## 🧩 Project Components

### 🔄 Huggingface Repository Manager

<div align="center">
  
![HF Manager Badge](https://img.shields.io/badge/tool-repository_manager-blue)
  
</div>

A sophisticated command-line utility designed for seamless management of Huggingface repositories.

📁 **Location**: [Huggingface_repo_manager/](Huggingface_repo_manager/)

<table>
  <tr>
    <th colspan="2">✨ Key Features</th>
  </tr>
  <tr>
    <td width="50%">• Upload local files to Huggingface</td>
    <td width="50%">• Delete files with built-in safeguards</td>
  </tr>
  <tr>
    <td width="50%">• View repository metadata</td>
    <td width="50%">• Browse files with tree visualization</td>
  </tr>
</table>

For detailed usage information, see the [component README](Huggingface_repo_manager/README.md).


# YOLOv8 Segmentation Evaluator

`yolov8_seg_evaluator.py` provides comprehensive evaluation of YOLOv8 segmentation models on test datasets. The tool calculates key metrics including IoU, Dice coefficient, pixel accuracy, and mAP while handling class imbalance.

## Features

- Processes YOLO format test data (images + text labels)
- Calculates per-class and weighted metrics
- Generates visualization plots (PR curves, confusion matrix, etc.)
- Exports results as CSV files for further analysis
- Handles unbalanced class distributions

## Usage

```bash
python yolov8_seg_evaluator.py --model path/to/model.pt --img-dir path/to/images --label-dir path/to/labels
```

### Arguments

- `--model`: Path to your YOLOv8 model file
- `--img-dir`: Directory containing test images
- `--label-dir`: Directory containing YOLO format label files
- `--num-classes`: Number of segmentation classes (default: 9)
- `--save-dir`: Directory to save evaluation results (default: 'evaluation_results')

## Output

The tool generates metrics CSV files and visualization plots in the specified output directory.



# YOLOv8 Segmentation Video Inference

`yolo_video_inference.py` processes videos using a trained YOLOv8 segmentation model, overlaying colored masks for each detected class and adding a visual legend.

## Features

- Processes video files with YOLOv8 segmentation models
- Overlays semi-transparent colored masks for each detected object
- Displays a color-coded legend showing all 9 surgical tool classes
- Shows real-time processing statistics (FPS, frame count)
- Optimized for surgical tool segmentation with distinct colors for easy identification

## Usage

```bash
python yolo_video_inference.py --model path/to/model.pt --video path/to/input.mp4 --output path/to/output.mp4
```

### Arguments

- `--model`: Path to your YOLOv8 segmentation model
- `--video`: Path to input video file
- `--output`: Path for output video (default: 'output_video.mp4')
- `--conf`: Confidence threshold for detections (default: 0.3)

## Output

The script generates a video with colored segmentation masks and a class legend in the corner.

# YOLOv8 Segmentation Image Processor

`yolo_image_inference.py` processes images using a trained YOLOv8 segmentation model, overlaying colored masks and creating a visual color legend for easy class identification.

## Features

- Processes single images or entire directories
- Overlays semi-transparent colored masks for each detected object
- Displays a comprehensive color-coded legend for all 9 surgical tool classes
- Adds contour outlines for better boundary visualization
- Supports multiple image formats (JPG, PNG, BMP)

## Usage

For a single image:
```bash
python yolo_image_inference.py --model path/to/model.pt --image path/to/image.jpg --output path/to/output.jpg
```

For a directory of images:
```bash
python yolo_image_inference.py --model path/to/model.pt --dir path/to/images --output path/to/output_dir
```

### Arguments

- `--model`: Path to your YOLOv8 segmentation model
- `--image`: Path to input image (use this OR --dir)
- `--dir`: Path to directory containing images (use this OR --image)
- `--output`: Path for output image or directory (optional)
- `--conf`: Confidence threshold for detections (default: 0.3)

## Output

The script generates images with colored segmentation masks and a legend in the top corner, making it easy to identify different surgical tools.