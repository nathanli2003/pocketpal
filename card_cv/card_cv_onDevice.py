
from roboflow import Roboflow
from dotenv import load_dotenv
import test_image_receive
import os
import json

import serial
import time

load_dotenv()
API_KEY = os.getenv("API_KEY")
PROJECT_ID = "playing-cards-ow27d"
MODEL_VERSION = 4
# IMAGE_URL = "/home/poker/Downloads/download.jpeg"
SERIAL_PORT = '/dev/ttyUSB0'   # change to your COM port
BAUD_RATE = 921600
SAVE_DIR = "esp32_images"
CHUNK_SIZE = 4096       # 4 KB per read

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=5)
img_data = test_image_receive.request_image(ser)

if img_data:
    filename = os.path.join(SAVE_DIR, f"photo.jpg")
    with open(filename, "wb") as f:
        f.write(img_data)
    print(f"Saved {filename} ({len(img_data)} bytes)")
IMAGE_URL = filename

# --- Main Script ---
try:
    # Initialize the Roboflow object with your API key
    rf = Roboflow(api_key=API_KEY)
    
    # Get a specific project
    project = rf.workspace().project(PROJECT_ID)
    
    # Get a specific model version
    model = project.version(MODEL_VERSION).model
    
    # Run inference on the image from the URL
    print(f"Running inference on model version {MODEL_VERSION}...")
    prediction = model.predict(IMAGE_URL, confidence=20, overlap=30).json()
    
    # Print the prediction results in a nicely formatted way
    print("\n--- Inference Results ---")
    print(json.dumps(prediction, indent=4))
    print("-------------------------")

except Exception as e:
    print(f"An error occurred: {e}")