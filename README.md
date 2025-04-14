# SAR-RARP50-seg

## Method

This project utilizes the YOLOv8 segmentation model (specifically, the `xlarge` variant) for semantic segmentation tasks, likely involving the SAR-RARP50 dataset.

## Installation

To set up the project, follow these steps:

1.  **Clone the repository:**

    You can clone the repository using either HTTPS (recommended) or SSH:

    * **HTTPS:**
        ```bash
        git clone [https://github.com/MahdiMohseni0033/SAR-RARP50-seg.git](https://github.com/MahdiMohseni0033/SAR-RARP50-seg.git)
        ```
    * **SSH:**
        ```bash
        git clone git@github.com:MahdiMohseni0033/SAR-RARP50-seg.git
        ```

2.  **Navigate into the project directory:**
    ```bash
    cd SAR-RARP50-seg
    ```

3.  **Create and activate a Conda environment:**

    This project requires Python 3.11. Create a dedicated environment using Conda:
    ```bash
    conda create -n yolo-sar python=3.11 -y
    conda activate yolo-sar
    ```

4.  **Install the required dependencies:**

    Install all necessary packages listed in the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset Preparation

This project requires downloading and processing specific datasets for training and testing. We've simplified this process with a single command.

### Download and prepare the training dataset

```bash
bash prepare_data.sh train
```

This will:
1. Download the training dataset
2. Extract and process the data
3. Split it into training and validation sets
4. Convert to YOLO format
5. Clean up temporary files

The processed training dataset will be available in the `yolo_dataset2` directory.

### Download and prepare the testing dataset

```bash
bash prepare_data.sh test
```

This will:
1. Download the testing dataset
2. Extract and process the data
3. Convert to YOLO format
4. Clean up temporary files

The processed testing dataset will be available in the `yolo_test_set` directory.


## Project Components

### Huggingface Repository Manager

![HF Manager Badge](https://img.shields.io/badge/tool-repository_manager-blue)

A command-line utility for managing Huggingface repositories. Handles file uploads, deletions, and repository exploration.

📁 **Location**: [Huggingface_repo_manager/](Huggingface_repo_manager/)

✨ **Features**:
- Upload local files to Huggingface repositories
- Delete files with built-in safeguards
- View repository metadata
- Browse files with tree visualization

For detailed usage information, see the [component README](Huggingface_repo_manager/README.md).





