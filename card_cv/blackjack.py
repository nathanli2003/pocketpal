from inference import get_model
from dotenv import load_dotenv
import os
import json
import supervision as sv # Using supervision for easier annotation
from PIL import Image, ImageDraw, ImageFont

# --- Configuration ---
load_dotenv()
API_KEY = os.getenv("API_KEY") # Still needed to download the model the first time
PROJECT_ID = "playing-cards-ow27d"
MODEL_VERSION = 4

# --- Image Source ---
IMAGE_PATH = "/home/poker/Documents/pocketpal/card_cv/9D.png"
ANNOTATED_IMAGE_PATH = "poker_cropped_2.jpg"

# --- Main Script ---
try:
    # Initialize the local model
    # This will download the model on the first run, then use the local cache.
    # After the first run, you can run this script completely offline.
    print("Loading local model...")
    model = get_model(model_id=f"{PROJECT_ID}/{MODEL_VERSION}", api_key=API_KEY)
    
    # Run local inference on the image
    print(f"Running local inference on {IMAGE_PATH}...")
    prediction_data = model.infer(IMAGE_PATH, confidence=0.20, overlap_threshold=0.30)
    
    # The output is a list of prediction objects, not a single JSON dict
    # We will access the 'predictions' attribute from the first result
    predictions = prediction_data[0].predictions
    
    # Print the prediction results in a nicely formatted way
    print("\n--- Inference Results ---")
    # Convert predictions to a list of dicts for printing
    predictions_as_dict = [p.dict() for p in predictions]
    print(json.dumps(predictions_as_dict, indent=4))
    print("-------------------------")

    # --- Annotate and Save Image ---
    print("\nAnnotating image with predictions...")
    
    # Load the original image
    image = Image.open(IMAGE_PATH).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    try:
        # Try to load a specific font, fallback to default
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    # Iterate over each prediction and draw a bounding box
    for pred in predictions:
        # Extract coordinates
        x1 = pred.x - pred.width / 2
        y1 = pred.y - pred.height / 2
        x2 = pred.x + pred.width / 2
        y2 = pred.y + pred.height / 2
        
        # Extract class name and confidence
        class_name = pred.class_name
        confidence = pred.confidence
        
        # Draw the bounding box
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
        
        # Create the label text
        label = f"{class_name} ({confidence:.2f})"
        
        # Get text size for background
        text_bbox = font.getbbox(label)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Draw background and text
        label_bg_coords = [x1, y1 - text_height - 5, x1 + text_width + 4, y1]
        draw.rectangle(label_bg_coords, fill="lime")
        draw.text((x1 + 2, y1 - text_height - 3), label, fill="black", font=font)

    # Save the annotated image
    image.save(ANNOTATED_IMAGE_PATH)
    print(f"Successfully saved annotated image to: {ANNOTATED_IMAGE_PATH}")

except FileNotFoundError:
    print(f"Error: The image file was not found at {IMAGE_PATH}")
except Exception as e:
    print(f"An error occurred: {e}")