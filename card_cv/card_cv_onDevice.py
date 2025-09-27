from roboflow import Roboflow
from dotenv import load_dotenv
import os
import json
import serial
import time
from PIL import Image, ImageDraw, ImageFont

# --- Configuration ---
load_dotenv()
API_KEY = os.getenv("API_KEY")
PROJECT_ID = "playing-cards-ow27d"
MODEL_VERSION = 4
SERIAL_PORT = '/dev/ttyUSB0'   # Change to your COM port if using ESP32
BAUD_RATE = 921600
SAVE_DIR = "esp32_images"

# --- Image Source ---
# This script uses a hardcoded image path.
# To use an image from your ESP32, uncomment the section below.
#
# print("Attempting to receive image from ESP32...")
# if not os.path.exists(SAVE_DIR):
#     os.makedirs(SAVE_DIR)
# try:
#     ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=5)
#     # A simple function to request image data would be needed here.
#     # For example:
#     # ser.write(b'GET_IMAGE\n')
#     # img_data = ser.read_until(b'END_IMAGE') # Assuming a delimiter
#     # if img_data:
#     #     IMAGE_PATH = os.path.join(SAVE_DIR, f"photo.jpg")
#     #     with open(IMAGE_PATH, "wb") as f:
#     #         f.write(img_data)
#     #     print(f"Saved {IMAGE_PATH} ({len(img_data)} bytes)")
#     # else:
#     #     raise ValueError("No image data received from ESP32.")
#     # ser.close()
# except (serial.SerialException, ValueError) as e:
#     print(f"Could not get image from ESP32: {e}")
#     print("Falling back to local image.")
#     IMAGE_PATH = "/home/poker/Documents/pocketpal/card_cv/esp32_images/photo.jpg" # Fallback path
#
IMAGE_PATH = "/home/poker/Downloads/1200px-AcetoFive.jpeg"
ANNOTATED_IMAGE_PATH = "annotated_image.jpg"

# --- Main Script ---
try:
    # Initialize the Roboflow object with your API key
    rf = Roboflow(api_key=API_KEY)
    
    # Get a specific project
    project = rf.workspace().project(PROJECT_ID)
    
    # Get a specific model version
    model = project.version(MODEL_VERSION).model
    
    # Run inference on the image
    print(f"Running inference on model version {MODEL_VERSION} using {IMAGE_PATH}...")
    prediction_data = model.predict(IMAGE_PATH, confidence=20, overlap=30).json()
    
    # Print the prediction results in a nicely formatted way
    print("\n--- Inference Results ---")
    print(json.dumps(prediction_data, indent=4))
    print("-------------------------")

    # --- Annotate and Save Image ---
    print("\nAnnotating image with predictions...")
    
    # Load the original image
    image = Image.open(IMAGE_PATH)
    draw = ImageDraw.Draw(image)
    
    font = ImageFont.load_default()

    # Iterate over each prediction and draw a bounding box
    for pred in prediction_data['predictions']:
        # Extract coordinates and dimensions from the prediction
        x = pred['x']
        y = pred['y']
        width = pred['width']
        height = pred['height']
        
        # Roboflow provides center coordinates, calculate top-left and bottom-right corners
        x1 = x - width / 2
        y1 = y - height / 2
        x2 = x + width / 2
        y2 = y + height / 2
        
        # Extract class name and confidence
        class_name = pred['class']
        confidence = pred['confidence']
        
        # Draw the bounding box
        # Using a bright color and a thicker line for better visibility
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
        
        # Create the label text
        label = f"{class_name} ({confidence:.2f})"
        
        # Get the size of the text to create a background rectangle
        text_bbox = font.getbbox(label)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Create a background for the label for better readability
        label_bg_coords = [x1, y1 - text_height - 5, x1 + text_width + 4, y1]
        draw.rectangle(label_bg_coords, fill="lime")
        
        # Draw the text label itself
        draw.text((x1 + 2, y1 - text_height - 3), label, fill="black", font=font)

    # Save the annotated image to a new file
    image.save(ANNOTATED_IMAGE_PATH)
    print(f"Successfully saved annotated image to: {ANNOTATED_IMAGE_PATH}")

except FileNotFoundError:
    print(f"Error: The image file was not found at {IMAGE_PATH}")
except Exception as e:
    print(f"An error occurred: {e}")
