import streamlit as st
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
import time
from PIL import Image
import io
import base64
import logging

torch.classes.__path__ = []
# Configure page
st.set_page_config(
    page_title="AI Segmentation App",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to make the app more visually appealing
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: bold;
        border: none;
        transition-duration: 0.4s;
    }
    .stButton button:hover {
        background-color: #45a049;
        box-shadow: 0 12px 16px 0 rgba(0,0,0,0.24), 0 17px 50px 0 rgba(0,0,0,0.19);
    }
    .stSlider > div > div > div {
        background-color: #4CAF50;
    }
    .title-container {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 1rem;
        border-radius: 10px;
        background: linear-gradient(90deg, #4CAF50 0%, #2E8B57 100%);
        color: white;
        margin-bottom: 2rem;
    }
    .result-container {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        margin-top: 1rem;
    }
    .sidebar-content {
        padding: 1rem;
        background-color: #f1f1f1;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    # App title
    st.markdown('<div class="title-container"><h1>🔍 AI Segmentation Studio</h1></div>', unsafe_allow_html=True)

    # Sidebar configuration
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.subheader("Model Configuration")
        model_path = st.text_input("Model Path", "runs/segment/train/weights/best.pt",
                                   help="Path to your YOLO segmentation model file (.pt)")

        conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05,
                                   help="Minimum confidence score for detections")
        iou_threshold = st.slider("IoU Threshold", 0.1, 1.0, 0.7, 0.05,
                                  help="Intersection over Union threshold for NMS")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.subheader("Visualization Settings")
        overlay_opacity = st.slider("Mask Opacity", 0.1, 1.0, 0.5, 0.05,
                                    help="Opacity of the segmentation mask overlay")

        display_legend = st.checkbox("Display Color Legend", True,
                                     help="Show class color legend on the output")

        output_quality = st.slider("Output Quality", 50, 100, 90, 5,
                                   help="Quality of the output image")
        st.markdown('</div>', unsafe_allow_html=True)

    # Process image
    process_image(model_path, conf_threshold, iou_threshold, overlay_opacity, display_legend, output_quality)


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
        # Generate random colors but ensure they're visually distinct
        np.random.seed(42)  # For reproducible colors
        palette = [(0, 0, 0)]  # Background is black

        # Generate initial colors
        for i in range(num_classes - 1):
            color = (np.random.randint(50, 256), np.random.randint(50, 256), np.random.randint(50, 256))
            palette.append(color)

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


def overlay_mask(image: np.ndarray, colorized_mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlays a colorized mask on the original image."""
    if image.shape[:2] != colorized_mask.shape[:2]:
        logger.warning(
            f"Image shape {image.shape[:2]} and colorized mask shape {colorized_mask.shape[:2]} differ. Resizing mask.")
        colorized_mask = cv2.resize(colorized_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    foreground_mask = np.any(colorized_mask != [0, 0, 0], axis=-1)
    blended_image = image.copy()
    blended_image[foreground_mask] = cv2.addWeighted(
        image.astype(float)[foreground_mask], 1 - alpha,
        colorized_mask.astype(float)[foreground_mask], alpha,
        0
    ).astype(np.uint8)

    return blended_image


def add_color_legend(image, palette, class_names, unique_class_ids):
    """Adds a color legend to the upper right corner of the image."""
    # Define the size and position of the box
    box_width = 180
    box_height = 30 * min(len(unique_class_ids), 10)  # Limit legend height
    if box_height < 30:  # Ensure minimum height
        box_height = 30

    margin = 10

    # Create a white box in the upper right corner
    x1 = image.shape[1] - box_width - margin
    y1 = margin
    x2 = image.shape[1] - margin
    y2 = y1 + box_height

    # Draw white box with slight transparency
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), 1)  # Black border

    # Add the color legend with transparency
    alpha = 0.8
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Add color swatches and class names
    count = 0
    for i, class_id in enumerate(unique_class_ids):
        if class_id == 0 or count >= 10:  # Skip background class and limit to 10 classes
            continue

        # Draw color square
        color = palette[class_id]
        y_pos = y1 + 15 + (count * 30)
        cv2.rectangle(image, (x1 + 10, y_pos - 10), (x1 + 30, y_pos + 10), color, -1)
        cv2.rectangle(image, (x1 + 10, y_pos - 10), (x1 + 30, y_pos + 10), (0, 0, 0), 1)  # Black border

        # Add class name
        class_name = class_names.get(class_id - 1, f"Class {class_id}")
        cv2.putText(image, class_name, (x1 + 40, y_pos + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        count += 1

    return image


def process_frame(model, frame, conf_thres, iou_thres, overlay_alpha, display_legend, num_classes, class_names):
    """Process a single image frame and return the overlayed result."""
    try:
        results = model.predict(frame, conf=conf_thres, iou=iou_thres, verbose=False)
        if not results or results[0].masks is None:
            logger.warning("Model prediction did not return masks for this frame. Returning original frame.")
            return frame

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            logger.warning("No bounding boxes found for this frame. Returning original frame.")
            return frame

        if result.masks is None or len(result.masks) == 0:
            logger.warning("No masks found for this frame. Returning original frame.")
            return frame

        confidences = result.boxes.conf.cpu().numpy()
        conf_mask = confidences >= conf_thres
        if not np.any(conf_mask):
            logger.warning(f"No detections above confidence threshold {conf_thres}. Returning original frame.")
            return frame

        masks_cpu = result.masks.data.cpu().numpy()
        filtered_masks = masks_cpu[conf_mask]
        class_ids_cpu = result.boxes.cls.cpu().numpy().astype(int)
        filtered_class_ids = class_ids_cpu[conf_mask]

        if filtered_masks.shape[0] == 0:
            logger.warning("No masks remained after confidence filtering. Returning original frame.")
            return frame

        orig_h, orig_w = frame.shape[:2]
        combined_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        for i, mask_instance in enumerate(filtered_masks):
            class_id = filtered_class_ids[i] + 1
            mask_instance_resized = cv2.resize(mask_instance, (orig_w, orig_h),
                                               interpolation=cv2.INTER_NEAREST) if mask_instance.shape != (
                orig_h, orig_w) else mask_instance
            binary_mask = (mask_instance_resized > 0.5).astype(np.uint8)
            combined_mask = np.maximum(combined_mask, binary_mask * class_id)

        colorized_mask, palette = colorize_mask(combined_mask, num_classes=num_classes + 1)
        colorized_mask_resized = cv2.resize(colorized_mask, (orig_w, orig_h),
                                            interpolation=cv2.INTER_NEAREST) if colorized_mask.shape[:2] != (
            orig_h, orig_w) else colorized_mask

        overlayed_frame = overlay_mask(frame, colorized_mask_resized, alpha=overlay_alpha)

        if display_legend:
            # Get unique class IDs excluding background (0)
            unique_class_ids = np.unique(combined_mask)
            unique_class_ids = unique_class_ids[unique_class_ids > 0]  # Exclude background

            # Add color legend to overlayed frame
            final_frame = add_color_legend(overlayed_frame, palette, class_names, unique_class_ids)
        else:
            final_frame = overlayed_frame

        return final_frame

    except Exception as e:
        logger.error(f"Error processing frame: {e}", exc_info=True)
        return frame


def load_model(model_path):
    """Load the YOLO model with error handling."""
    progress_text = "Loading model..."
    progress_bar = st.progress(0)

    try:
        model_path = Path(model_path)

        if not model_path.is_file():
            script_dir = Path.cwd()
            potential_path = script_dir / model_path
            if potential_path.is_file():
                model_path = potential_path
                logger.info(f"Resolved model path to: {model_path}")
            else:
                st.error(f"⚠️ Model not found at: {model_path} or {potential_path}")
                st.stop()

        progress_bar.progress(25)

        # Load the model
        model = YOLO(model_path)

        progress_bar.progress(75)

        # Get class names from model
        class_names = {}
        if hasattr(model, 'names'):
            num_classes = len(model.names)
            class_names = model.names
            logger.info(f"Model has {num_classes} classes: {model.names}")
        else:
            st.error("⚠️ Could not determine number of classes from model.")
            st.stop()

        progress_bar.progress(100)
        time.sleep(0.5)  # Show completed progress briefly
        progress_bar.empty()

        return model, num_classes, class_names

    except Exception as e:
        progress_bar.empty()
        st.error(f"⚠️ Error loading model: {str(e)}")
        st.stop()


def get_download_link(img, filename, text, quality=90):
    """Generate a download link for an image."""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buffered = io.BytesIO()
    img_pil.save(buffered, format="JPEG", quality=quality)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'
    return href


def process_image(model_path, conf_threshold, iou_threshold, overlay_opacity, display_legend, output_quality):
    """Process and segment an image."""
    # File uploader for images
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png', 'bmp'])

    if uploaded_file is not None:
        # Create columns for before/after
        col1, col2 = st.columns(2)

        # Display the original image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with col1:
            st.markdown('<div class="result-container">', unsafe_allow_html=True)
            st.subheader("Original Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Load model
        model, num_classes, class_names = load_model(model_path)

        # Process the image
        with st.spinner("Processing image..."):
            processed_image = process_frame(
                model, image, conf_threshold, iou_threshold,
                overlay_opacity, display_legend, num_classes, class_names
            )

        # Display the processed image
        with col2:
            st.markdown('<div class="result-container">', unsafe_allow_html=True)
            st.subheader("Segmented Image")
            st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Download option
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.markdown("### Download Results")
        download_link = get_download_link(
            processed_image, "segmented_image.jpg",
            "📥 Download Segmented Image", output_quality
        )
        st.markdown(download_link, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Add instructions
    else:
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.markdown("""
        ### 📋 Instructions
        1. Upload an image (JPG, PNG, or BMP format)
        2. Adjust the confidence and IoU thresholds as needed
        3. Customize the visualization settings
        4. After processing, you can download the segmented image

        **Note:** Processing time depends on the image size and complexity.
        """)
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()