from ultralytics import YOLO
import os

if __name__ == '__main__':
    # Make sure the directory structure exists
    os.makedirs('runs/detect', exist_ok=True)
    
    # Initialize model
    model = YOLO('yolov8n.pt')  # Use the nano model for faster training
    
    # Start training
    results = model.train(
        data=os.path.join(os.getcwd(), 'data', 'PCB.yaml'),  # Use absolute path
        imgsz=640,
        epochs=100,
        batch=8,  # Reduce if you have memory issues
        workers=4,  # Adjust based on your CPU
        patience=20,  # Early stopping
        project='runs',
        name='train'
    )
    
    # Export model to ONNX format
    model.export(format="onnx")