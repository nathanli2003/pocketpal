import serial
import time
import os

SERIAL_PORT = 'COM10'   # change to your COM port
BAUD_RATE = 921600
SAVE_DIR = "esp32_images"
CHUNK_SIZE = 4096       # 4 KB per read

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def request_image(ser):
    """Request a single image from ESP32 and save it."""
    # Send request byte
    ser.write(b'R')
    
    # Read 4-byte length
    length_bytes = ser.read(4)
    if len(length_bytes) < 4:
        print("Failed to read image length")
        return None
    length = int.from_bytes(length_bytes, 'little')
    
    # Read JPEG data in chunks
    img_data = bytearray()
    bytes_remaining = length
    while bytes_remaining > 0:
        chunk = ser.read(min(CHUNK_SIZE, bytes_remaining))
        if not chunk:
            print("Serial read timed out")
            break
        img_data.extend(chunk)
        bytes_remaining -= len(chunk)
    
    if len(img_data) != length:
        print(f"Warning: expected {length} bytes, got {len(img_data)} bytes")
    
    return img_data

def main():
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=5)
    time.sleep(2)  # give ESP32 time to reset
    img_count = 0

    try:
        while True:
            input("Press Enter to request an image...")
            img_data = request_image(ser)
            if img_data:
                img_count += 1
                filename = os.path.join(SAVE_DIR, f"photo_{img_count}.jpg")
                with open(filename, "wb") as f:
                    f.write(img_data)
                print(f"Saved {filename} ({len(img_data)} bytes)")
            else:
                print("Failed to receive image.")
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        ser.close()

if __name__ == "__main__":
    main()
