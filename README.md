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