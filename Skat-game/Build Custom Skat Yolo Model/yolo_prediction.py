import gradio as gr
from gradio_webrtc import WebRTC
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import sys
import os
import threading

# Global model variable
model = None
model_lock = threading.Lock()

def load_yolo_model(model_path):
    global model
    try:
        model = YOLO(model_path)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def predict_image(image):
    try:
        if model is None:
            return None, "Model failed to load. Please check the model path and requirements."
        
        if image is None:
            return None, "Please provide an image"
            
        # Convert image to BGR (OpenCV format) if it's RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Make prediction
        results = model(image)
        
        # Get the plot of results
        result_plot = results[0].plot()
        
        # Convert back to RGB for display
        result_plot_rgb = cv2.cvtColor(result_plot, cv2.COLOR_BGR2RGB)
        
        # Get prediction details
        predictions = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]
                predictions.append(f"{class_name}: {conf:.2f}")
        
        return Image.fromarray(result_plot_rgb), "\n".join(predictions)
    except Exception as e:
        return None, f"Error during prediction: {str(e)}"

def process_frame(frame_data):
    """Process each frame from the webcam stream"""
    global model
    try:
        if frame_data is None or model is None:
            return frame_data

        print("Processing frame...")  # Debug print
        
        # Convert frame to BGR for YOLO processing
        frame_bgr = cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR)
        
        # Run YOLO detection with confidence threshold
        results = model(frame_bgr, conf=0.3)  # You can adjust confidence threshold here
        
        # Draw the results on the frame
        annotated_frame = results[0].plot()
        
        # Convert back to RGB for display
        processed_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        print("Frame processed successfully")  # Debug print
        
        return processed_frame

    except Exception as e:
        print(f"Error in process_frame: {str(e)}")
        return frame_data

# Load model at startup
print("Loading model...")
model = load_yolo_model("best.pt")

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# YOLO Object Detection (Upload & Live Stream)")
    
    with gr.Tabs():
        # Tab for image upload
        with gr.TabItem("Image Upload"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="numpy", label="Upload Image")
                    detect_button = gr.Button("Detect Objects")
                with gr.Column():
                    output_image = gr.Image(type="pil", label="Detected Objects")
                    output_text = gr.Textbox(label="Predictions")
            
            detect_button.click(
                fn=predict_image,
                inputs=[image_input],
                outputs=[output_image, output_text]
            )
        
        # Tab for live detection
        with gr.TabItem("Live Detection"):
            webrtc = WebRTC()
            # Stream setup
            webrtc.stream(
                fn=process_frame,
                inputs=[webrtc],  # Pass webrtc as input
                outputs=[webrtc]  # Pass webrtc as output
            )

if __name__ == "__main__":
    if model is None:
        print("Failed to load model. Please check the model path and requirements.")
        sys.exit(1)
    print("Starting Gradio interface...")
    demo.launch(share=True)