import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from ultralytics import YOLO
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from pathlib import Path
import argparse

def load_model(checkpoint_path):
    """Load the YOLOv8 model from checkpoint."""
    model = YOLO(checkpoint_path)
    return model


def parse_yolo_label(label_path, img_width, img_height, num_classes=9):
    """Parse YOLO label file and return segmentation mask."""
    mask = np.zeros((img_height, img_width, num_classes), dtype=np.uint8)

    if not os.path.exists(label_path):
        return mask

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        points = []

        # Extract polygon points
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                x = float(parts[i]) * img_width
                y = float(parts[i + 1]) * img_height
                points.append([x, y])

        # Convert to numpy array
        points = np.array(points, dtype=np.int32)

        # Create mask for this instance
        if len(points) > 2:  # Need at least 3 points for a polygon
            cv2.fillPoly(mask[:, :, class_id], [points], 1)

    return mask


def calculate_iou(pred_mask, gt_mask):
    """Calculate IoU for each class."""
    ious = []
    for c in range(pred_mask.shape[2]):
        pred = pred_mask[:, :, c]
        gt = gt_mask[:, :, c]

        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()

        iou = intersection / union if union > 0 else 0
        ious.append(iou)

    return ious


def calculate_dice(pred_mask, gt_mask):
    """Calculate Dice coefficient for each class."""
    dice_scores = []
    for c in range(pred_mask.shape[2]):
        pred = pred_mask[:, :, c]
        gt = gt_mask[:, :, c]

        intersection = np.logical_and(pred, gt).sum()
        total = pred.sum() + gt.sum()

        dice = 2 * intersection / total if total > 0 else 0
        dice_scores.append(dice)

    return dice_scores


def calculate_pixel_accuracy(pred_mask, gt_mask):
    """Calculate pixel accuracy for each class."""
    accuracies = []
    for c in range(pred_mask.shape[2]):
        pred = pred_mask[:, :, c]
        gt = gt_mask[:, :, c]

        correct = (pred == gt).sum()
        total = pred.shape[0] * pred.shape[1]

        accuracy = correct / total if total > 0 else 0
        accuracies.append(accuracy)

    return accuracies


def calculate_precision_recall(pred_masks, gt_masks, num_classes=9):
    """Calculate precision and recall for each class."""
    precisions = {c: [] for c in range(num_classes)}
    recalls = {c: [] for c in range(num_classes)}

    for c in range(num_classes):
        y_true = []
        y_scores = []

        for i in range(len(pred_masks)):
            pred = pred_masks[i][:, :, c].flatten()
            gt = gt_masks[i][:, :, c].flatten()

            y_true.extend(gt.tolist())
            y_scores.extend(pred.tolist())

        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        precisions[c] = precision
        recalls[c] = recall

    return precisions, recalls


def evaluate_model(model, test_img_dir, test_label_dir, num_classes=9, save_dir="results"):
    """Evaluate YOLOv8 segmentation model on test data."""
    os.makedirs(save_dir, exist_ok=True)

    # Get image and label files
    img_files = sorted([f for f in os.listdir(test_img_dir) if f.endswith('.png') or f.endswith('.jpg')])

    # Metrics containers
    all_ious = {i: [] for i in range(num_classes)}
    all_dice = {i: [] for i in range(num_classes)}
    all_accuracies = {i: [] for i in range(num_classes)}
    class_counts = {i: 0 for i in range(num_classes)}

    # Confusion matrix
    conf_matrix = np.zeros((num_classes, num_classes))

    # Save predictions and ground truths for PR curve calculation
    pred_masks_list = []
    gt_masks_list = []

    for img_file in tqdm(img_files, desc="Evaluating"):
        img_path = os.path.join(test_img_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(test_label_dir, label_file)

        # Read image
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]

        # Parse ground truth label
        gt_mask = parse_yolo_label(label_path, img_width, img_height, num_classes)

        # Run inference
        results = model.predict(img_path, save=False, verbose=False)[0]

        # Create prediction mask
        pred_mask = np.zeros((img_height, img_width, num_classes), dtype=np.uint8)

        for segment in results.masks.segments:
            cls = int(results.boxes.cls[results.masks.segments.index(segment)].item())
            if cls < num_classes:  # Ensure class is within range
                segment = np.array(segment, dtype=np.int32)
                segment[:, 0] = segment[:, 0] * img_width
                segment[:, 1] = segment[:, 1] * img_height
                cv2.fillPoly(pred_mask[:, :, cls], [segment.astype(int)], 1)

        # Calculate metrics
        ious = calculate_iou(pred_mask, gt_mask)
        dice_scores = calculate_dice(pred_mask, gt_mask)
        accuracies = calculate_pixel_accuracy(pred_mask, gt_mask)

        # Update metrics
        for c in range(num_classes):
            if gt_mask[:, :, c].sum() > 0:  # Only count classes present in ground truth
                all_ious[c].append(ious[c])
                all_dice[c].append(dice_scores[c])
                all_accuracies[c].append(accuracies[c])
                class_counts[c] += 1

        # Update confusion matrix
        for c in range(num_classes):
            gt = gt_mask[:, :, c].sum() > 0
            pred = pred_mask[:, :, c].sum() > 0
            if gt and pred:
                conf_matrix[c, c] += 1  # True positive
            elif gt and not pred:
                for other_c in range(num_classes):
                    if pred_mask[:, :, other_c].sum() > 0:
                        conf_matrix[c, other_c] += 1
            elif not gt and pred:
                for other_c in range(num_classes):
                    if gt_mask[:, :, other_c].sum() > 0:
                        conf_matrix[other_c, c] += 1

        # Save for PR curve
        pred_masks_list.append(pred_mask)
        gt_masks_list.append(gt_mask)

    # Calculate mean metrics per class
    mean_iou_per_class = {c: np.mean(all_ious[c]) if all_ious[c] else 0 for c in range(num_classes)}
    mean_dice_per_class = {c: np.mean(all_dice[c]) if all_dice[c] else 0 for c in range(num_classes)}
    mean_accuracy_per_class = {c: np.mean(all_accuracies[c]) if all_accuracies[c] else 0 for c in range(num_classes)}

    # Calculate overall metrics
    weighted_iou = sum(mean_iou_per_class[c] * class_counts[c] for c in range(num_classes)) / sum(class_counts.values())
    weighted_dice = sum(mean_dice_per_class[c] * class_counts[c] for c in range(num_classes)) / sum(
        class_counts.values())
    weighted_accuracy = sum(mean_accuracy_per_class[c] * class_counts[c] for c in range(num_classes)) / sum(
        class_counts.values())

    # Calculate PR curves
    precisions, recalls = calculate_precision_recall(pred_masks_list, gt_masks_list, num_classes)

    # Calculate average precision
    avg_precisions = {}
    for c in range(num_classes):
        y_true = []
        y_scores = []

        for i in range(len(pred_masks_list)):
            pred = pred_masks_list[i][:, :, c].flatten()
            gt = gt_masks_list[i][:, :, c].flatten()

            y_true.extend(gt.tolist())
            y_scores.extend(pred.tolist())

        avg_precisions[c] = average_precision_score(y_true, y_scores)

    # Save metrics as CSV
    metrics_df = pd.DataFrame({
        'Class': list(range(num_classes)),
        'Count': [class_counts[c] for c in range(num_classes)],
        'IoU': [mean_iou_per_class[c] for c in range(num_classes)],
        'Dice': [mean_dice_per_class[c] for c in range(num_classes)],
        'Accuracy': [mean_accuracy_per_class[c] for c in range(num_classes)],
        'AP': [avg_precisions[c] for c in range(num_classes)]
    })

    metrics_df.to_csv(os.path.join(save_dir, 'class_metrics.csv'), index=False)

    # Create summary row
    summary_df = pd.DataFrame({
        'Metric': ['Weighted IoU', 'Weighted Dice', 'Weighted Accuracy', 'mAP'],
        'Value': [
            weighted_iou,
            weighted_dice,
            weighted_accuracy,
            np.mean(list(avg_precisions.values()))
        ]
    })

    summary_df.to_csv(os.path.join(save_dir, 'summary_metrics.csv'), index=False)

    # Plot results
    # 1. Class distribution
    plt.figure(figsize=(12, 6))
    plt.bar(range(num_classes), [class_counts[c] for c in range(num_classes)])
    plt.title('Class Distribution in Test Data')
    plt.xlabel('Class ID')
    plt.ylabel('Count')
    plt.xticks(range(num_classes))
    plt.savefig(os.path.join(save_dir, 'class_distribution.png'))

    # 2. IoU per class
    plt.figure(figsize=(12, 6))
    plt.bar(range(num_classes), [mean_iou_per_class[c] for c in range(num_classes)])
    plt.title('Mean IoU per Class')
    plt.xlabel('Class ID')
    plt.ylabel('IoU')
    plt.xticks(range(num_classes))
    plt.savefig(os.path.join(save_dir, 'iou_per_class.png'))

    # 3. Dice per class
    plt.figure(figsize=(12, 6))
    plt.bar(range(num_classes), [mean_dice_per_class[c] for c in range(num_classes)])
    plt.title('Mean Dice Coefficient per Class')
    plt.xlabel('Class ID')
    plt.ylabel('Dice')
    plt.xticks(range(num_classes))
    plt.savefig(os.path.join(save_dir, 'dice_per_class.png'))

    # 4. Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='.1f', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))

    # 5. Precision-Recall curves
    plt.figure(figsize=(12, 10))
    for c in range(num_classes):
        if len(all_ious[c]) > 0:  # Only plot if class exists in test set
            plt.plot(recalls[c], precisions[c], label=f'Class {c} (AP={avg_precisions[c]:.3f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc='best')
    plt.savefig(os.path.join(save_dir, 'precision_recall_curves.png'))

    # 6. Plot metrics comparison
    plt.figure(figsize=(14, 7))
    metrics = ['IoU', 'Dice', 'Accuracy', 'AP']
    data = np.array([
        [mean_iou_per_class[c] for c in range(num_classes)],
        [mean_dice_per_class[c] for c in range(num_classes)],
        [mean_accuracy_per_class[c] for c in range(num_classes)],
        [avg_precisions[c] for c in range(num_classes)]
    ])

    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)
        plt.bar(range(num_classes), data[i])
        plt.title(f'{metric} per Class')
        plt.xlabel('Class ID')
        plt.ylabel(metric)
        plt.xticks(range(num_classes))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'))

    return {
        'class_metrics': metrics_df,
        'summary_metrics': summary_df,
        'class_counts': class_counts,
        'mean_iou_per_class': mean_iou_per_class,
        'mean_dice_per_class': mean_dice_per_class,
        'mean_accuracy_per_class': mean_accuracy_per_class,
        'avg_precisions': avg_precisions,
        'weighted_iou': weighted_iou,
        'weighted_dice': weighted_dice,
        'weighted_accuracy': weighted_accuracy,
        'mAP': np.mean(list(avg_precisions.values()))
    }


def main():
    """Main function to run evaluation with command line arguments."""


    parser = argparse.ArgumentParser(description='Evaluate YOLOv8 segmentation model on test data')
    parser.add_argument('--model', type=str, required=True, help='Path to YOLOv8 model checkpoint (.pt file)')
    parser.add_argument('--img-dir', type=str, required=True, help='Directory containing test images')
    parser.add_argument('--label-dir', type=str, required=True, help='Directory containing test labels (YOLO format)')
    parser.add_argument('--num-classes', type=int, default=9, help='Number of segmentation classes (default: 9)')
    parser.add_argument('--save-dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')

    args = parser.parse_args()

    # Load model
    print("Loading model...")
    model = load_model(args.model)

    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(model, args.img_dir, args.label_dir, args.num_classes, args.save_dir)

    # Print summary
    print("\nEvaluation Results Summary:")
    print(f"mAP: {results['mAP']:.4f}")
    print(f"Weighted IoU: {results['weighted_iou']:.4f}")
    print(f"Weighted Dice: {results['weighted_dice']:.4f}")
    print(f"Weighted Accuracy: {results['weighted_accuracy']:.4f}")

    print("\nPer-Class Results:")
    print(results['class_metrics'].to_string(index=False))

    print(f"\nResults saved to {args.save_dir}")


if __name__ == "__main__":
    main()