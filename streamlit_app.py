import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import io
import base64
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Surgical Tool Segmentation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #f0f0f0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .result-text {
        font-size: 1.2rem;
        font-weight: 500;
        color: #2E7D32;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #f0f0f0;
        color: #666;
        font-size: 0.8rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Default model path
DEFAULT_MODEL_PATH = "models/yolov8_segmentation.pt"  # Change this to your default model path

# Class names and colors
CLASS_NAMES = [
    'Tool shaft',
    'Tool clasper',
    'Tool wrist',
    'Thread',
    'Clamps',
    'Suturing needle',
    'Suction tool',
    'Catheter',
    'Needle Holder'
]


# Generate distinct colors for each class using HSV color space
@st.cache_data
def generate_colors(num_classes):
    colors = []
    for i in range(num_classes):
        # Use HSV color space for better distinctness
        hue = int(i * 255 / num_classes)
        # Full saturation and value for vibrant colors
        hsv_color = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        # Convert to BGR for OpenCV
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        # Convert to RGB format
        colors.append((int(bgr_color[2]), int(bgr_color[1]), int(bgr_color[0])))
    return colors


def draw_legend(image, class_names, colors):
    """Draw a legend with class names and their corresponding colors."""
    # Configuration
    start_x = 20
    start_y = 30
    box_size = 15
    text_offset = 5
    line_spacing = 5
    text_scale = 0.5
    text_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Get the width of the longest class name
    max_width = 0
    for name in class_names:
        (text_width, _), _ = cv2.getTextSize(name, font, text_scale, text_thickness)
        max_width = max(max_width, text_width)

    # Calculate background dimensions
    legend_height = len(class_names) * (box_size + line_spacing) + line_spacing
    legend_width = start_x + box_size + text_offset + max_width + 20

    # Create semi-transparent background
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (legend_width, legend_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image, 0)

    # Draw class boxes and names
    for i, name in enumerate(class_names):
        y = start_y + i * (box_size + line_spacing)

        # Draw color box
        cv2.rectangle(image, (start_x, y - box_size), (start_x + box_size, y), colors[i], -1)
        cv2.rectangle(image, (start_x, y - box_size), (start_x + box_size, y), (0, 0, 0), 1)

        # Draw class name
        cv2.putText(image, name, (start_x + box_size + text_offset, y - 2),
                    font, text_scale, (255, 255, 255), text_thickness)

    return image


def process_image(model, image, conf_threshold=0.3):
    """Process image with YOLOv8 segmentation model and overlay masks."""
    # Generate colors for classes
    colors = generate_colors(len(CLASS_NAMES))

    # Convert PIL Image to cv2 image
    image_cv = np.array(image)
    # Convert RGB to BGR (OpenCV format)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    height, width = image_cv.shape[:2]

    # Run inference
    results = model.predict(image_cv, conf=conf_threshold, verbose=False)[0]

    # Create a copy for drawing
    overlay_image = image_cv.copy()

    # Initialize a set to track detected classes
    detected_classes = set()

    # Process masks
    if hasattr(results, 'masks') and results.masks is not None:
        masks = results.masks.data.cpu().numpy()
        boxes = results.boxes.data.cpu().numpy()

        for i, mask in enumerate(masks):
            # Get class ID
            class_id = int(boxes[i][5])
            detected_classes.add(class_id)

            # Ensure class_id is within our range
            if class_id < len(colors):
                # Create binary mask
                mask = mask.reshape(height, width)
                binary_mask = (mask > 0.5).astype(np.uint8)

                # Create colored mask
                colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
                colored_mask[binary_mask == 1] = colors[class_id]

                # Blend with original image
                alpha = 0.5  # Transparency factor
                mask_area = (binary_mask == 1)
                overlay_image[mask_area] = cv2.addWeighted(
                    overlay_image[mask_area],
                    1 - alpha,
                    colored_mask[mask_area],
                    alpha,
                    0
                )

                # Add contour for better visibility
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay_image, contours, -1, colors[class_id], 2)

    # Add legend
    overlay_image = draw_legend(overlay_image, CLASS_NAMES, colors)

    # Convert back to RGB for displaying in Streamlit
    result_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)

    return result_image, detected_classes


def process_video_frame(frame, model, colors, conf_threshold=0.3):
    """Process a single video frame."""
    height, width = frame.shape[:2]

    # Run inference
    results = model.predict(frame, conf=conf_threshold, verbose=False)[0]

    # Create a copy for drawing
    overlay_frame = frame.copy()

    # Process each detected segment
    if hasattr(results, 'masks') and results.masks is not None:
        masks = results.masks.data.cpu().numpy()
        boxes = results.boxes.data.cpu().numpy()

        for i, mask in enumerate(masks):
            # Get class ID
            class_id = int(boxes[i][5])

            # Ensure the class_id is within our range
            if class_id < len(colors):
                # Create binary mask
                mask = mask.reshape(height, width)
                binary_mask = (mask > 0.5).astype(np.uint8)

                # Create colored mask
                colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
                colored_mask[binary_mask == 1] = colors[class_id]

                # Blend with original frame
                alpha = 0.5  # Transparency factor
                mask_area = (binary_mask == 1)
                overlay_frame[mask_area] = cv2.addWeighted(
                    overlay_frame[mask_area],
                    1 - alpha,
                    colored_mask[mask_area],
                    alpha,
                    0
                )

                # Add contour for better visibility
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay_frame, contours, -1, colors[class_id], 2)

    # Add legend
    overlay_frame = draw_legend(overlay_frame, CLASS_NAMES, colors)

    return overlay_frame


def get_download_link(file_path, file_name):
    """Generate a download link for a file."""
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/download;base64,{b64}" download="{file_name}">Download {file_name}</a>'
    return href


# Sidebar content
st.sidebar.markdown("<h1 style='text-align: center;'>Settings</h1>", unsafe_allow_html=True)

# Confidence threshold
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.3,
    step=0.05,
    help="Minimum confidence score for detection"
)

# Inference mode
inference_mode = st.sidebar.radio(
    "Inference Mode",
    options=["Image", "Video"],
    help="Select whether to process an image or video"
)

# Main content
st.markdown("<h1 class='main-header'>Surgical Tool Segmentation</h1>", unsafe_allow_html=True)

# Display information about the application
with st.expander("ℹ️ About this app", expanded=False):
    st.markdown("""
    <div class='info-box'>
        <p>This application uses a YOLOv8 segmentation model to detect and segment surgical tools in images and videos.</p>
        <p>The model can detect the following 9 classes of surgical tools:</p>
        <ul>
            <li>Tool shaft</li>
            <li>Tool clasper</li>
            <li>Tool wrist</li>
            <li>Thread</li>
            <li>Clamps</li>
            <li>Suturing needle</li>
            <li>Suction tool</li>
            <li>Catheter</li>
            <li>Needle Holder</li>
        </ul>
        <p>To get started, select your inference mode (Image or Video) in the sidebar, then upload a file for processing.</p>
    </div>
    """, unsafe_allow_html=True)

# Try to load the model from the default path
try:
    # Load the model
    with st.spinner("Loading YOLOv8 model..."):
        model = YOLO(DEFAULT_MODEL_PATH)
        st.sidebar.success("✅ Model loaded successfully!")

    # Generate colors for classes
    colors = generate_colors(len(CLASS_NAMES))

    # Image inference
    if inference_mode == "Image":
        st.markdown("<h2 class='sub-header'>Image Segmentation</h2>", unsafe_allow_html=True)

        # Upload image
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            # Read image
            image = Image.open(uploaded_image)

            # Create columns for before/after display
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("<p style='text-align: center;'>Original Image</p>", unsafe_allow_html=True)
                st.image(image, use_column_width=True)

            # Process image
            with st.spinner("Processing image..."):
                start_time = time.time()
                result_image, detected_classes = process_image(model, image, conf_threshold)
                processing_time = time.time() - start_time

            with col2:
                st.markdown("<p style='text-align: center;'>Segmented Image</p>", unsafe_allow_html=True)
                st.image(result_image, use_column_width=True)

            # Show metrics
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                st.metric("Processing Time", f"{processing_time:.3f} seconds")

            with metric_col2:
                detected_classes_list = [CLASS_NAMES[i] for i in detected_classes]
                st.metric("Detected Classes", len(detected_classes))

            with metric_col3:
                st.metric("Confidence Threshold", f"{conf_threshold:.2f}")

            st.markdown("</div>", unsafe_allow_html=True)

            # Display detected classes
            if detected_classes:
                st.markdown("<p class='result-text'>Detected Surgical Tools:</p>", unsafe_allow_html=True)
                detected_classes_str = ", ".join([CLASS_NAMES[i] for i in detected_classes])
                st.info(detected_classes_str)
            else:
                st.warning("No surgical tools detected in the image.")

            # Save result
            result_pil = Image.fromarray(result_image)
            output_buffer = io.BytesIO()
            result_pil.save(output_buffer, format="PNG")

            # Download button
            st.download_button(
                label="Download Segmented Image",
                data=output_buffer.getvalue(),
                file_name=f"segmented_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
            )

    # Video inference
    else:
        st.markdown("<h2 class='sub-header'>Video Segmentation</h2>", unsafe_allow_html=True)

        # Upload video
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

        if uploaded_video is not None:
            # Save the uploaded video to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
                tmp_video.write(uploaded_video.getvalue())
                video_path = tmp_video.name

            # Display the original video
            st.markdown("<p style='text-align: center;'>Original Video</p>", unsafe_allow_html=True)
            st.video(uploaded_video)

            # Process video button
            if st.button("Process Video"):
                # Get video properties
                cap = cv2.VideoCapture(video_path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Create a temporary file for the output video
                output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

                # Create video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Process frames
                frame_idx = 0
                start_time = time.time()

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Process frame
                    processed_frame = process_video_frame(frame, model, colors, conf_threshold)

                    # Write frame to output video
                    out.write(processed_frame)

                    # Update progress
                    frame_idx += 1
                    progress = frame_idx / frame_count
                    progress_bar.progress(progress)
                    status_text.text(
                        f"Processing: {frame_idx}/{frame_count} frames ({progress * 100:.1f}%) - ETA: {((time.time() - start_time) / frame_idx) * (frame_count - frame_idx):.1f}s")

                # Release resources
                cap.release()
                out.release()

                # Processing time
                processing_time = time.time() - start_time

                # Display results
                st.markdown("<p style='text-align: center;'>Processed Video</p>", unsafe_allow_html=True)
                st.video(output_path)

                # Show metrics
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                metric_col1, metric_col2, metric_col3 = st.columns(3)

                with metric_col1:
                    st.metric("Processing Time", f"{processing_time:.2f} seconds")

                with metric_col2:
                    st.metric("Processed Frames", frame_idx)

                with metric_col3:
                    st.metric("Average FPS", f"{frame_idx / processing_time:.2f}")

                st.markdown("</div>", unsafe_allow_html=True)

                # Download link
                output_filename = f"segmented_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                st.markdown(get_download_link(output_path, output_filename), unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error loading or using the model: {e}")
    st.info(f"Please make sure the model exists at: {DEFAULT_MODEL_PATH}")
    st.info("The app expects a YOLOv8 segmentation model trained on surgical tool data.")

# Footer
st.markdown("""
<div class='footer'>
    <p>YOLOv8 Surgical Tool Segmentation App</p>
    <p>Powered by Ultralytics and Streamlit</p>
</div>
""", unsafe_allow_html=True)