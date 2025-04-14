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

# Downlaod the dataset

using following command to download train set:
```bash
wget -O train_set.zip "https://rdr.ucl.ac.uk/ndownloader/articles/24932529/versions/1"
mkdir train_set && unzip train_set.zip -d train_set
rm -rf train_set.zip 
bash unzip_all.sh train_set
python dataset_manager2.py \
    --input_dir train_set \
    --masks_dir Dataset/all_masks \
    --frames_dir Dataset/all_frames \
    --format png

rm -rf train_set

python split_data.py --image_dir Dataset/all_masks \
                    --mask_dir Dataset/all_frames \
                    --output_dir splited_dataset \
                    --split_ratio 0.85

rm -rf  Dataset   
python convert_to_yolo.py \
    --mode train_val \
    --train-images splited_dataset/images/train \
    --train-masks splited_dataset/labels/train \
    --val-images splited_dataset/images/val \
    --val-masks splited_dataset/labels/val \
    --output-dir yolo_dataset2  

          
```



using following command to download test set:
```bash
wget -O test_set.zip "https://rdr.ucl.ac.uk/ndownloader/articles/24932499/versions/1"
mkdir test_set && unzip test_set.zip -d test_set
rm -rf test_set.zip 
bash unzip_all.sh test_set

python dataset_manager2.py \
    --input_dir test_set \
    --masks_dir Dataset/all_masks \
    --frames_dir Dataset/all_frames \
    --format png

python convert_to_yolo.py \
    --mode test \
    --test-images Dataset/all_frames \
    --test-masks Dataset/all_masks \
    --output-dir yolo_test_set

rm -rf test_set


```


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





