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


# Hugging Face Repository Manager

A simple Python script to interactively manage files and view information for your Hugging Face Hub repositories (models, datasets, or spaces).

## Features

* Upload individual files to a specified repository.
* Delete individual files from a specified repository (with confirmation prompt).
* View repository metadata (ID, author, last modified, tags, download counts, etc.).
* List all files currently stored within a repository.
* Interactive, menu-driven command-line interface.
* Supports `model`, `dataset`, and `space` repository types.
* Flexible authentication:
    * Enter token directly via secure prompt.
    * Use `HF_TOKEN` environment variable.
    * Use cached token from `huggingface-cli login`.

## Prerequisites
* **Make sure the requirements in the requirements.txt file are installed.**
* **Hugging Face Hub Account:** You need an account on [huggingface.co](https://huggingface.co/).
* **Hugging Face User Access Token:**
    * You need a token to authenticate with the API.
    * Generate one from your settings: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
    * The token requires **`read`** permissions for viewing info and listing files.
    * The token requires **`write`** permissions for uploading or deleting files.


## Usage

1. un the Script:** Execute the script using Python:
    ```bash
    python hf_manager.py
    ```
2.  **Initial Configuration:** The script will prompt you for setup information:
    * **Repository ID:** Enter the ID of the repository you want to manage (e.g., `YourUsername/YourRepoName`).
    * **Hugging Face API Token:**
        * You can paste your access token here. The input will be hidden for security. Press Enter when done.
        * **Alternatively:** If you have configured the `HF_TOKEN` environment variable or logged in using `huggingface-cli login`, you can just press Enter without pasting a token. The script will attempt to use those methods.
    * **Repository Type:** Enter the type: `model`, `dataset`, or `space`. If you press Enter without typing, it will default to `model`.

3.  **Interact with the Menu:**
    * After successful configuration, a menu will appear showing the available actions and the currently configured repository.
    * Enter the number corresponding to your desired action (1-5) and press Enter.

    ```
    ============================================================
     Hugging Face Manager | Repo: YourUsername/YourRepoName (model)
    ============================================================
    Please choose an action:
      1. Upload a file
      2. Delete a file
      3. Show repository information
      4. List repository files (tree)
      5. Exit
    ------------------------------------------------------------
    Enter your choice (1-5):
    ```

4.  **Follow Prompts:** For actions like uploading or deleting, the script will ask for further details (e.g., local file path, path within the repo, commit message). Provide the requested information.
5.  **Return to Menu:** After an action completes (or is cancelled), press Enter to go back to the main menu.
6.  **Exit:** Choose option `5` to close the script.

## Actions Explained

* **1. Upload a file:**
    * Asks for the full path to the file on your local machine.
    * Asks for the desired path/filename within the Hugging Face repository (defaults to the original filename if left blank).
    * Asks for a commit message (provides a default if left blank).
* **2. Delete a file:**
    * Asks for the exact path of the file *within* the Hugging Face repository that you want to remove.
    * Asks for a commit message (provides a default if left blank).
    * **Requires explicit confirmation (`yes`) before proceeding.** **Warning: Deletion is permanent and cannot be undone!**
* **3. Show repository information:**
    * Fetches and displays details about the configured repository, such as author, privacy status, last modification date, tags, download/like counts, and the repository URL.
* **4. List repository files (tree):**
    * Fetches and displays a sorted list of all files currently present in the configured repository.
* **5. Exit:**
    * Closes the script.






