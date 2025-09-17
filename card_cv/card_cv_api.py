from inference_sdk import InferenceHTTPClient
import json

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="p4h8F5NBZiDPVoFaH0XZ"
)

# Replace this with your actual image path
image_path = "/home/nathan/Downloads/rsz_1200px-jack_playing_cards.jpg"

try:
    # Perform inference on the local image file
    result = CLIENT.infer(image_path, model_id="playing-cards-ow27d/4")
    
    # Check if predictions were returned
    if result and 'predictions' in result and result['predictions']:
        print("--- Detected Cards ---")
        
        # Iterate through each detected prediction
        for i, prediction in enumerate(result['predictions']):
            # Extract key information from the prediction dictionary
            card_class = prediction.get('class')
            confidence = prediction.get('confidence')
            x = prediction.get('x')
            y = prediction.get('y')
            width = prediction.get('width')
            height = prediction.get('height')
            
            # Print the formatted output
            print(f"\nCard {i+1}:")
            print(f"  Class: {card_class}")
            print(f"  Confidence: {confidence:.2%}") # Format confidence as a percentage
            print(f"  Bounding Box:")
            print(f"    - Center: ({x:.1f}, {y:.1f})")
            print(f"    - Dimensions: {width:.1f}x{height:.1f} (width x height)")
            
    else:
        print("No cards were detected in the image.")

except Exception as e:
    print(f"An error occurred: {e}")