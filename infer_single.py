import cv2
import numpy as np
import argparse
from ultralytics import YOLO
import matplotlib.pyplot as plt

def run_inference(model_path, image_path, conf_threshold=0.25):
    """Run inference on a single image and display results"""
    # Load model
    model = YOLO(model_path)
    
    # Run inference
    results = model.predict(source=image_path, conf=conf_threshold)
    
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get detections
    boxes = results[0].boxes
    
    # Draw results
    for box in boxes:
        # Get box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Get confidence and class
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        class_name = results[0].names[cls]
        
        # Create label
        label = f"{class_name} {conf:.2f}"
        
        # Set color based on class
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255)]
        color = colors[cls % len(colors)]
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.rectangle(img, (x1, y1 - 20), (x1 + text_size[0], y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Display image
    plt.figure(figsize=(12, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Defects detected: {len(boxes)}")
    plt.show()
    
    # Save the annotated image
    output_path = image_path.replace('.jpg', '_result.jpg').replace('.png', '_result.png')
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Result saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PCB defect detection on a single image")
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--model', type=str, default='runs/train/weights/best.pt', help='Path to model weights')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    
    args = parser.parse_args()
    
    run_inference(args.model, args.image, args.conf)