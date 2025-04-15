import os
import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
from PIL import Image
import json

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def colorize_mask(mask: np.ndarray, num_classes: int = 9) -> tuple:
    """Colorizes a single-channel mask image and returns color palette."""
    if num_classes <= 10:
        palette = [
            (0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128),
        ]
    elif num_classes <= 20:
        palette = [
            (0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128),
            (255, 128, 0), (128, 255, 0), (0, 255, 128),
            (0, 128, 255), (128, 0, 255), (255, 0, 128),
            (64, 64, 64), (192, 192, 192), (220, 220, 220), (105, 105, 105)
        ]
    else:
        palette = [(0, 0, 0)] + [(np.random.randint(50, 256), np.random.randint(50, 256), np.random.randint(50, 256))
                                 for _ in
                                 range(num_classes - 1)]
        while len(palette) < num_classes:
            palette.append((np.random.randint(50, 256), np.random.randint(50, 256), np.random.randint(50, 256)))
    palette = palette[:num_classes]
    colorized_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    unique_mask_values = np.unique(mask)
    for i in unique_mask_values:
        if i < num_classes:
            colorized_mask[mask == i] = palette[i]
        else:
            logger.warning(
                f"Mask contains class index {i} which is out of bounds for the palette size {num_classes}. Assigning black.")
            colorized_mask[mask == i] = (0, 0, 0)
    return colorized_mask, palette


def load_yolo_gt_mask(label_path, image_shape, num_classes):
    """
    Load YOLO format segmentation ground truth label and convert to mask

    Args:
        label_path: Path to the YOLO format label file
        image_shape: Tuple of (height, width) for the output mask
        num_classes: Number of classes in the dataset

    Returns:
        A ground truth mask with class indices
    """
    height, width = image_shape
    gt_mask = np.zeros((height, width), dtype=np.uint8)

    if not Path(label_path).exists():
        logger.warning(f"Label file not found: {label_path}")
        return gt_mask

    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:  # Must have at least class and one point
                continue

            class_id = int(parts[0]) + 1  # Add 1 because 0 is background in our mask
            if class_id >= num_classes:
                logger.warning(f"Class ID {class_id} out of range for {num_classes} classes")
                continue

            # Convert polygon points from normalized to absolute
            points = []
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    x = float(parts[i]) * width
                    y = float(parts[i + 1]) * height
                    points.append([x, y])

            # Convert points to integer array for cv2
            if len(points) >= 3:  # Need at least 3 points for a polygon
                points_array = np.array(points, dtype=np.int32)
                cv2.fillPoly(gt_mask, [points_array], class_id)
    except Exception as e:
        logger.error(f"Error parsing label file {label_path}: {e}")

    return gt_mask


def calculate_iou(pred_mask, gt_mask, class_id):
    """Calculate IoU for a specific class"""
    pred_binary = (pred_mask == class_id).astype(np.uint8)
    gt_binary = (gt_mask == class_id).astype(np.uint8)

    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union


def calculate_dice(pred_mask, gt_mask, class_id):
    """Calculate Dice coefficient for a specific class"""
    pred_binary = (pred_mask == class_id).astype(np.uint8)
    gt_binary = (gt_mask == class_id).astype(np.uint8)

    intersection = np.logical_and(pred_binary, gt_binary).sum()
    pred_sum = pred_binary.sum()
    gt_sum = gt_binary.sum()

    if pred_sum + gt_sum == 0:
        return 1.0 if intersection == 0 else 0.0

    return (2.0 * intersection) / (pred_sum + gt_sum)


def calculate_pixel_accuracy(pred_mask, gt_mask):
    """Calculate overall pixel accuracy"""
    return np.mean(pred_mask == gt_mask)


def calculate_class_metrics(pred_mask, gt_mask, num_classes):
    """Calculate various metrics for each class"""
    metrics = {}

    # Process all classes (including background = 0)
    unique_classes = sorted(np.unique(np.concatenate([np.unique(pred_mask), np.unique(gt_mask)])))

    # Calculate per-class metrics
    class_ious = []
    class_dices = []
    class_precisions = []
    class_recalls = []
    class_f1s = []

    for class_id in unique_classes:
        # Skip background class (0) if desired
        if class_id == 0:
            continue

        # Calculate IoU and Dice for this class
        iou = calculate_iou(pred_mask, gt_mask, class_id)
        dice = calculate_dice(pred_mask, gt_mask, class_id)

        # Binary masks for this class
        pred_binary = (pred_mask == class_id).ravel()
        gt_binary = (gt_mask == class_id).ravel()

        # Handle empty ground truth
        if gt_binary.sum() == 0:
            precision = 1.0 if pred_binary.sum() == 0 else 0.0
            recall = 1.0  # Recall is 1 when there's nothing to recall
            f1 = 1.0 if pred_binary.sum() == 0 else 0.0
        else:
            precision = precision_score(gt_binary, pred_binary, zero_division=0)
            recall = recall_score(gt_binary, pred_binary, zero_division=0)
            f1 = f1_score(gt_binary, pred_binary, zero_division=0)

        # Store per-class metrics
        class_ious.append(iou)
        class_dices.append(dice)
        class_precisions.append(precision)
        class_recalls.append(recall)
        class_f1s.append(f1)

        # Add to metrics dictionary
        metrics[f"class_{class_id}"] = {
            "iou": iou,
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    # Calculate mean across classes
    metrics["mean"] = {
        "miou": np.mean(class_ious) if class_ious else 0.0,
        "mdice": np.mean(class_dices) if class_dices else 0.0,
        "mprecision": np.mean(class_precisions) if class_precisions else 0.0,
        "mrecall": np.mean(class_recalls) if class_recalls else 0.0,
        "mf1": np.mean(class_f1s) if class_f1s else 0.0
    }

    # Calculate pixel accuracy
    metrics["pixel_accuracy"] = calculate_pixel_accuracy(pred_mask, gt_mask)

    return metrics


def create_confusion_matrix(pred_mask, gt_mask, num_classes):
    """Create a confusion matrix for the predictions"""
    # Flatten masks for confusion matrix
    pred_flat = pred_mask.ravel()
    gt_flat = gt_mask.ravel()

    # Create confusion matrix
    cm = confusion_matrix(gt_flat, pred_flat, labels=range(num_classes))

    return cm


def plot_confusion_matrix(cm, class_names, output_path):
    """Plot the confusion matrix and save it to a file"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_metrics_per_class(metrics, class_names, output_dir):
    """Plot various metrics per class and save to files"""
    # Prepare data
    class_keys = [k for k in metrics.keys() if k.startswith('class_')]
    class_ids = [int(k.split('_')[1]) for k in class_keys]

    # Sort by class id
    sorted_indices = np.argsort(class_ids)
    class_ids = [class_ids[i] for i in sorted_indices]
    class_keys = [class_keys[i] for i in sorted_indices]

    # Get class names
    labels = [class_names.get(class_id - 1, f"Class {class_id}") for class_id in class_ids]

    # Metrics to plot
    metric_names = ["iou", "dice", "precision", "recall", "f1"]
    metric_titles = ["IoU", "Dice Coefficient", "Precision", "Recall", "F1 Score"]

    # Create plots
    for metric_name, metric_title in zip(metric_names, metric_titles):
        plt.figure(figsize=(12, 6))
        values = [metrics[class_key][metric_name] for class_key in class_keys]
        bars = plt.bar(labels, values)

        # Add mean line
        mean_value = metrics["mean"][f"m{metric_name}"]
        plt.axhline(y=mean_value, color='r', linestyle='-', label=f'Mean: {mean_value:.3f}')

        # Add value annotations
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.3f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom')

        plt.title(f'Per-Class {metric_title}')
        plt.xlabel('Class')
        plt.ylabel(metric_title)
        plt.ylim(0, 1.05)
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric_name}_per_class.png'))
        plt.close()


def visualize_prediction(image, gt_mask, pred_mask, class_names, num_classes, output_path):
    """
    Create a visualization with original image, ground truth, prediction and error map
    """
    # Get colorized masks
    gt_colored, palette = colorize_mask(gt_mask, num_classes)
    pred_colored, _ = colorize_mask(pred_mask, num_classes)

    # Create error map (red where prediction differs from ground truth)
    error_map = np.zeros_like(image)
    error_map[gt_mask != pred_mask] = [0, 0, 255]  # Red for errors

    # Create a 2x2 grid figure
    plt.figure(figsize=(12, 10))

    # Original image
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # Ground truth
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(gt_colored, cv2.COLOR_BGR2RGB))
    plt.title('Ground Truth')
    plt.axis('off')

    # Prediction
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(pred_colored, cv2.COLOR_BGR2RGB))
    plt.title('Prediction')
    plt.axis('off')

    # Error map
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(cv2.addWeighted(image, 0.7, error_map, 0.3, 0), cv2.COLOR_BGR2RGB))
    plt.title('Error Map')
    plt.axis('off')

    # Add a figure legend for classes
    unique_classes = sorted(np.unique(gt_mask))
    handles = []
    labels = []

    for class_id in unique_classes:
        if class_id == 0:
            continue  # Skip background
        color = palette[class_id]
        # Convert BGR to RGB for matplotlib
        color_rgb = (color[2] / 255, color[1] / 255, color[0] / 255)
        patch = plt.Rectangle((0, 0), 1, 1, fc=color_rgb)
        handles.append(patch)
        class_name = class_names.get(class_id - 1, f"Class {class_id}")
        labels.append(class_name)

    plt.figlegend(handles, labels, loc='lower center', ncol=min(5, len(labels)),
                  bbox_to_anchor=(0.5, 0.03), bbox_transform=plt.gcf().transFigure)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path)
    plt.close()


def run_prediction(model, image_path, conf_thres, iou_thres):
    """Run prediction on a single image and return the mask"""
    original_image = cv2.imread(str(image_path))
    if original_image is None:
        raise ValueError(f"Could not read image: {image_path}")

    results = model.predict(original_image, conf=conf_thres, iou=iou_thres, verbose=False)

    if not results or results[0].masks is None:
        logger.warning(f"Model prediction did not return masks for {image_path}.")
        return None, original_image

    # Process prediction masks
    result = results[0]
    masks_cpu = result.masks.data.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    class_ids_cpu = result.boxes.cls.cpu().numpy().astype(int)

    # Filter by confidence
    conf_mask = confidences >= conf_thres
    filtered_masks = masks_cpu[conf_mask]
    filtered_class_ids = class_ids_cpu[conf_mask]

    # If no masks remain after filtering
    if filtered_masks.shape[0] == 0:
        logger.warning(f"No masks remained after confidence filtering for {image_path}.")
        return None, original_image

    # Create combined mask with class indices
    orig_h, orig_w = original_image.shape[:2]
    combined_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

    for i, mask_instance in enumerate(filtered_masks):
        class_id = filtered_class_ids[i] + 1  # Add 1 as 0 is reserved for background
        mask_instance_resized = cv2.resize(mask_instance, (orig_w, orig_h),
                                           interpolation=cv2.INTER_NEAREST) if mask_instance.shape != (
            orig_h, orig_w) else mask_instance
        binary_mask = (mask_instance_resized > 0.5).astype(np.uint8)
        combined_mask = np.maximum(combined_mask, binary_mask * class_id)

    return combined_mask, original_image


def evaluate_model(model_path, image_dir, label_dir, output_dir, conf_thres, iou_thres, max_samples=None):
    """Evaluate a YOLO segmentation model on a test dataset"""
    model_path = Path(model_path)
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    # Load model
    try:
        model = YOLO(model_path)
        logger.info(f"Loaded model from: {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        return

    # Get class names from model
    class_names = {}
    if hasattr(model, 'names'):
        num_classes = len(model.names) + 1  # Add 1 for background class
        class_names = model.names
        logger.info(f"Model has {len(model.names)} classes: {model.names}")
    else:
        logger.error("Could not determine number of classes from model. Using default of 10 classes.")
        num_classes = 10

    # Find all images
    image_files = sorted([f for f in image_dir.glob('*') if f.is_file() and
                          f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')])

    if not image_files:
        logger.error(f"No supported images found in directory: {image_dir}")
        return

    # Limit number of samples if specified
    if max_samples and max_samples > 0:
        image_files = image_files[:max_samples]

    logger.info(f"Found {len(image_files)} images for evaluation")

    # Initialize metrics storage
    all_metrics = []
    confusion_matrices = []

    # Process each image
    for image_file in tqdm(image_files, desc="Evaluating images"):
        # Get corresponding label file
        label_file = label_dir / f"{image_file.stem}.txt"
        if not label_file.exists():
            logger.warning(f"Label file not found for {image_file.name}, skipping")
            continue

        try:
            # Run prediction
            pred_mask, original_image = run_prediction(model, image_file, conf_thres, iou_thres)
            if pred_mask is None:
                continue

            # Load ground truth mask from YOLO format
            gt_mask = load_yolo_gt_mask(label_file, original_image.shape[:2], num_classes)

            # Calculate metrics
            metrics = calculate_class_metrics(pred_mask, gt_mask, num_classes)
            all_metrics.append(metrics)

            # Create confusion matrix
            cm = create_confusion_matrix(pred_mask, gt_mask, num_classes)
            confusion_matrices.append(cm)

            # Create visualization
            viz_path = viz_dir / f"{image_file.stem}_eval.png"
            visualize_prediction(original_image, gt_mask, pred_mask, class_names, num_classes, viz_path)

        except Exception as e:
            logger.error(f"Error processing {image_file.name}: {e}", exc_info=True)

    if not all_metrics:
        logger.error("No images were successfully evaluated!")
        return

    # Aggregate metrics
    logger.info("Calculating aggregate metrics...")

    # Calculate mean metrics across all images
    mean_metrics = {
        "mean": {
            "miou": np.mean([m["mean"]["miou"] for m in all_metrics]),
            "mdice": np.mean([m["mean"]["mdice"] for m in all_metrics]),
            "mprecision": np.mean([m["mean"]["mprecision"] for m in all_metrics]),
            "mrecall": np.mean([m["mean"]["mrecall"] for m in all_metrics]),
            "mf1": np.mean([m["mean"]["mf1"] for m in all_metrics]),
        },
        "pixel_accuracy": np.mean([m["pixel_accuracy"] for m in all_metrics])
    }

    # Calculate per-class metrics across all images
    all_class_keys = set()
    for m in all_metrics:
        all_class_keys.update([k for k in m.keys() if k.startswith('class_')])

    for class_key in all_class_keys:
        class_metrics = [m.get(class_key, {"iou": 0, "dice": 0, "precision": 0, "recall": 0, "f1": 0})
                         for m in all_metrics]
        mean_metrics[class_key] = {
            "iou": np.mean([m["iou"] for m in class_metrics]),
            "dice": np.mean([m["dice"] for m in class_metrics]),
            "precision": np.mean([m["precision"] for m in class_metrics]),
            "recall": np.mean([m["recall"] for m in class_metrics]),
            "f1": np.mean([m["f1"] for m in class_metrics])
        }

    # Generate summary plots
    logger.info("Generating summary plots...")

    # Plot per-class metrics
    plot_metrics_per_class(mean_metrics, class_names, output_dir)

    # Plot combined confusion matrix
    combined_cm = sum(confusion_matrices)
    # Normalize confusion matrix
    row_sums = combined_cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(combined_cm, row_sums, out=np.zeros_like(combined_cm, dtype=float), where=row_sums != 0)

    # Class names for plot (including background)
    cm_class_names = ["Background"] + [class_names.get(i, f"Class {i + 1}") for i in range(num_classes - 1)]
    plot_confusion_matrix(cm_normalized, cm_class_names, os.path.join(output_dir, "confusion_matrix.png"))

    # Save metrics to JSON
    with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
        json.dump(mean_metrics, f, indent=2)

    # Print summary metrics
    logger.info("\n===== EVALUATION RESULTS =====")
    logger.info(f"Mean IoU: {mean_metrics['mean']['miou']:.4f}")
    logger.info(f"Mean Dice: {mean_metrics['mean']['mdice']:.4f}")
    logger.info(f"Mean Precision: {mean_metrics['mean']['mprecision']:.4f}")
    logger.info(f"Mean Recall: {mean_metrics['mean']['mrecall']:.4f}")
    logger.info(f"Mean F1: {mean_metrics['mean']['mf1']:.4f}")
    logger.info(f"Pixel Accuracy: {mean_metrics['pixel_accuracy']:.4f}")
    logger.info("=============================")

    return mean_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a YOLO segmentation model on a test dataset.")
    parser.add_argument("--model_path", type=Path, default="best.pt",
                        help="Path to the trained YOLO model (.pt file)")
    parser.add_argument("--image_dir", type=Path, default="final_code/yolo_test_set/images/test",
                        help="Directory containing test images")
    parser.add_argument("--label_dir", type=Path, default="final_code/yolo_test_set/labels/test",
                        help="Directory containing test labels in YOLO format")
    parser.add_argument("--output_dir", type=Path, default="evaluation_results_claude",
                        help="Directory to save evaluation results")
    parser.add_argument("--conf_thres", type=float, default=0.5,
                        help="Confidence threshold for filtering predictions (default: 0.5)")
    parser.add_argument("--iou_thres", type=float, default=0.7,
                        help="IoU threshold for NMS (default: 0.7)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate (default: all)")

    args = parser.parse_args()

    evaluate_model(
        args.model_path,
        args.image_dir,
        args.label_dir,
        args.output_dir,
        args.conf_thres,
        args.iou_thres,
        args.max_samples
    )