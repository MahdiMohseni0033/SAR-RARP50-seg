import os
import argparse
import yaml
import logging
from ultralytics import YOLO

# Import utilities
from utils import get_device, log_config, logger # Use the configured logger

# --- Set Environment Variable (Optional, if needed) ---
# Uncomment if you encounter specific library conflicts or hangs (e.g., on Windows)
# os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' # If using MPS and encountering issues


def run_training(config: dict):
    """
    Executes the YOLO segmentation model training pipeline using settings from the config dict.

    Args:
        config: A dictionary containing all configuration settings loaded from YAML.
    """
    logger.info("Starting training process...")

    # --- Setup Device ---
    # Allow device override from config, otherwise auto-detect
    requested_device = config.get('device', None)
    selected_device = get_device(requested_device)
    # Update config dict with the actually selected device for clarity in logs/results
    config['device'] = selected_device
    logger.info(f"Using device: {selected_device}")

    # --- Log Configuration ---
    log_config(config) # Log the final configuration being used

    # --- Load Model ---
    model_path = config.get('model', 'yolov8n-seg.pt') # Default if not specified
    logger.info(f"Loading base model: {model_path}")
    try:
        # YOLO constructor handles both building from YAML and loading .pt weights
        # It automatically uses pretrained weights if the path ends with .pt
        model = YOLO(model_path)

    except Exception as e:
        logger.error(f"Error loading model '{model_path}': {e}", exc_info=True)
        logger.error("Please ensure the model name/path in config.yaml is correct.")
        logger.error("Available Ultralytics segmentation models include: yolov8n-seg.pt, yolov8s-seg.pt, etc.")
        return # Exit if model loading fails

    # --- Start Training ---
    logger.info("Initiating model training...")
    try:
        # Pass the entire configuration dictionary to the train method.
        # YOLO's train method will parse the relevant arguments.
        # We exclude keys that are not direct arguments to train(), like 'model' itself.
        train_args = config.copy()
        train_args.pop('model', None) # Remove model path, it's used in constructor

        results = model.train(**train_args)

        logger.info("Training finished successfully.")
        # Access results metrics if needed: results.maps, results.box_loss, etc.
        save_dir = getattr(model.trainer, 'save_dir', f"{config.get('project', 'runs/segment')}/{config.get('name', 'exp')}")
        logger.info(f"Results saved to: {save_dir}")

    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
        # Consider adding more specific error handling if needed (e.g., OutOfMemoryError)


if __name__ == "__main__":
    # --- Parse Command Line Arguments ---
    parser = argparse.ArgumentParser(description="Train YOLOv8 Segmentation Model using YAML config")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml", # Default config file name
        help="Path to the training configuration YAML file."
    )
    args = parser.parse_args()

    # --- Load Configuration from YAML ---
    config_path = args.config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded configuration from: {config_path}")
    except FileNotFoundError:
        logger.error(f"Error: Configuration file not found at '{config_path}'")
        exit(1) # Exit if config file is missing
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file '{config_path}': {e}", exc_info=True)
        exit(1) # Exit on YAML parsing error
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading config: {e}", exc_info=True)
        exit(1)

    # --- Run Training ---
    run_training(config)