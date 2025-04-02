from ultralytics import YOLO
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def visualize_predictions(image_path, results):
    """Visualize detection results on an image"""
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get the detection results
    boxes = results[0].boxes
    
    # Draw bounding boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        class_name = results[0].names[cls]
        label = f"{class_name} {conf:.2f}"
        
        # Define colors for different classes
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255)]
        color = colors[cls % len(colors)]
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label box
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.rectangle(image, (x1, y1 - 20), (x1 + text_size[0], y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Display the image
    plt.figure(figsize=(12, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Defects detected: {len(boxes)}")
    plt.show()
    
    return image

if __name__ == "__main__":
    # Load your trained model - using the best model from training
    model_path = os.path.join("runs", "train2", "weights", "best.pt") 
    model = YOLO(model_path)
    
    # Directory with validation images to test on
    val_dir = os.path.join("data", "PCB", "images", "val")
    
    # Create output directory for results
    output_dir = os.path.join("runs", "detect", "predictions")
    os.makedirs(output_dir, exist_ok=True)
    
    # Select a few images for visualization
    image_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith(('.jpg', '.png'))][:5]
    
    for image_path in image_files:
        print(f"Predicting on {os.path.basename(image_path)}")
        
        # Run prediction
        results = model.predict(source=image_path, conf=0.25)
        
        # Visualize and save results
        output_image = visualize_predictions(image_path, results)
        
        # Save the image with detections
        output_name = os.path.basename(image_path)
        output_path = os.path.join(output_dir, output_name)
        cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
        print(f"Saved results to {output_path}")

    # You can also run prediction on the entire validation set
    # Uncomment the following lines to do this
    # all_results = model.predict(
    #     source=val_dir,
    #     conf=0.25,
    #     save=True,
    #     project="runs",
    #     name="detect/all_predictions"
    # )
    
    print("Predictions completed!")