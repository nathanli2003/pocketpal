from inference import get_model
from dotenv import load_dotenv
import os
from PIL import Image
import numpy as np

# --- Configuration ---
load_dotenv()
API_KEY = os.getenv("API_KEY") # Still needed to download the model the first time
PROJECT_ID = "poker-cards-cxcvz"
MODEL_VERSION = 1

def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """
    Performs Non-Max Suppression to filter overlapping bounding boxes.

    Args:
        boxes (np.ndarray): Array of bounding boxes in format [x1, y1, x2, y2].
        scores (np.ndarray): Array of confidence scores for each box.
        iou_threshold (float): The IoU threshold to use for suppression.

    Returns:
        np.ndarray: An array of indices of the boxes to keep.
    """
    # Sort the indices of the boxes by confidence score in descending order
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        # The index of the box with the highest confidence
        i = order[0]
        keep.append(i)
        
        # Get the coordinates of the remaining boxes
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        # Calculate the width and height of the intersection
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = w * h

        # Calculate the Intersection over Union (IoU)
        box_i_area = (boxes[i, 2] - boxes[i, 0] + 1) * (boxes[i, 3] - boxes[i, 1] + 1)
        other_boxes_area = (boxes[order[1:], 2] - boxes[order[1:], 0] + 1) * (boxes[order[1:], 3] - boxes[order[1:], 1] + 1)
        union = box_i_area + other_boxes_area - intersection
        iou = intersection / union

        # Find the indices of boxes to remove (those with IoU > threshold)
        indices_to_remove = np.where(iou > iou_threshold)[0]
        
        # Remove the current box and the overlapping boxes from the order list
        order = np.delete(order, np.concatenate(([0], indices_to_remove + 1)))

    return np.array(keep)


def NMS(
    image_path: str,
    output_dir: str = "cropped_cards_manual",
    confidence_threshold: float = 0.20,
    overlap_threshold: float = 0.50
):
    """
    Detects playing cards, uses a manual NMS function to filter them,
    and saves each unique card as a cropped image.
    """
    try:
        # --- 1. Load Model and Run Inference ---
        print("Loading local model...")
        model = get_model(model_id=f"{PROJECT_ID}/{MODEL_VERSION}", api_key=API_KEY)

        print(f"Running local inference on {image_path}...")
        prediction_data = model.infer(image_path, confidence=confidence_threshold)
        predictions = prediction_data[0].predictions

        if not predictions:
            print("No cards were detected in the image.")
            return

        print(f"Initial raw detections found: {len(predictions)}")

        # --- 2. Prepare Data for NMS ---
        # Extract boxes and scores from the model's prediction objects
        boxes_list = []
        scores_list = []
        for pred in predictions:
            x1 = pred.x - pred.width / 2
            y1 = pred.y - pred.height / 2
            x2 = pred.x + pred.width / 2
            y2 = pred.y + pred.height / 2
            boxes_list.append([x1, y1, x2, y2])
            scores_list.append(pred.confidence)
        
        boxes_np = np.array(boxes_list)
        scores_np = np.array(scores_list)

        # --- 3. Apply Manual Non-Max Suppression ---
        print(f"Applying Manual Non-Max Suppression with overlap threshold: {overlap_threshold}")
        kept_indices = non_max_suppression(boxes_np, scores_np, iou_threshold=overlap_threshold)
        
        # Filter the original boxes using the indices returned by NMS
        filtered_boxes = boxes_np[kept_indices]
        print(f"Detections after filtering: {len(filtered_boxes)}")

        # --- 4. Crop and Save Images ---
        print(f"\nCropping and saving unique cards to '{output_dir}' directory...")
        os.makedirs(output_dir, exist_ok=True)
        original_image = Image.open(image_path).convert("RGB")

        # --- MODIFICATION START ---
        # Get original image dimensions and define the padding
        img_width, img_height = original_image.size
        padding = 50

        for box in filtered_boxes:
            x1, y1, x2, y2 = box

            # Apply padding to the bounding box coordinates
            # Use max/min to ensure the new coordinates do not go outside the image bounds
            padded_x1 = max(0, x1 - padding)
            padded_y1 = max(0, y1 - padding)
            padded_x2 = min(img_width, x2 + padding)
            padded_y2 = min(img_height, y2 + padding)

            # Crop the image using the new, padded coordinates
            cropped_image = original_image.crop((padded_x1, padded_y1, padded_x2, padded_y2))
            
            # Filename is based on original coordinates for reference
            filename = f"x{int(x1)}y{int(y1)}.png"
            output_path = os.path.join(output_dir, filename)
            cropped_image.save(output_path)
            print(f"  -> Saved {output_path}")
        # --- MODIFICATION END ---

        print("\nProcessing complete.")

    except FileNotFoundError:
        print(f"Error: The image file was not found at {image_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    IMAGE_SOURCE_PATH = "iphoneimage.jpg"
    OUTPUT_FOLDER = "card_images"
    
    NMS(
        image_path=IMAGE_SOURCE_PATH,
        output_dir=OUTPUT_FOLDER,
        confidence_threshold=0.10,
        overlap_threshold=0.9
    )