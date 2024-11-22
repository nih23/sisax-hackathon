import requests
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

def call_lsa(image_path, text_prompt, sam_type = "sam2.1_hiera_small", box_threshold = 0.5, text_threshold = 0.5):
    #image_path = "/home/ubuntu/lang-segment-anything/assets/car.jpeg"

    # Prepare the multipart form-data payload
    with open(image_path, "rb") as image_file:
                files = {"image": image_file}  # Server expects the key 'image_file'
                data = {
                       "sam_type": sam_type,
                       "box_threshold": box_threshold,
                       "text_threshold": text_threshold,
                       "text_prompt": text_prompt,
                       }

                    # Send the POST request to the predict endpoint
                object_detection_host = os.getenv("OBJECT_DETECTION_HOST")
                response = requests.post(f"{object_detection_host}/predict", files=files, data=data)

                # Handle the response
                if response.status_code == 200:
                     #output_image = Image.open(BytesIO(response.content)).convert("RGB")
                     boxes = np.load(BytesIO(response.content))
                else:
                     print(f"Error: {response.status_code} - {response.text}")

    return boxes