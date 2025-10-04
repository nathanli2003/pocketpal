from ultralytics import YOLO
import cv2
import os

if __name__ == '__main__':

    model = YOLO("yolo11l.pt")

    train_results = model.train(
        data="dataset/data.yaml",
        epochs=200,
        imgsz=640,
        batch=16, 
        device=0,
        degrees=30,
        translate=0.2,
        scale=0.5,
        shear=25.0,
        perspective=0.3,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,   
        mixup=0.0,  
        patience=50
    )


    results_dir = train_results.save_dir
    convergence_plot_path = os.path.join(results_dir, "results.png")

    print(f"Convergence plot and other results saved in: {results_dir}")

    # Display the convergence plot
    if os.path.exists(convergence_plot_path):
        convergence_img = cv2.imread(convergence_plot_path)
        cv2.imshow("Convergence Plot", convergence_img)
        cv2.waitKey(0) # Wait for a key press to close the image window
        cv2.destroyAllWindows()
    else:
        print("Could not find the convergence plot.")

    print("\nRunning validation...")
    metrics = model.val()
    print("Validation metrics:", metrics)

    # Export the model to ONNX format for deployment
    print("\nExporting model to ONNX format...")
    path = model.export(format="onnx")
    print(f"Model exported to: {path}") 