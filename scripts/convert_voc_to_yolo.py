import os
import xml.etree.ElementTree as ET
import random
import shutil

# box [xmin,ymin,xmax,ymax]
def convert(size, box):
    x_center = (box[2] + box[0]) / 2.0
    y_center = (box[3] + box[1]) / 2.0
    # Normalize
    x = x_center / size[0]
    y = y_center / size[1]
    # Calculate width and height and normalize
    w = (box[2] - box[0]) / size[0]
    h = (box[3] - box[1]) / size[1]
    return (x, y, w, h)

# Convert xml to yolo format
def convert_annotation(xml_path, yolo_txt_path, classes):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    # Get width and height values from xml
    w = int(size.find("width").text)
    h = int(size.find("height").text)
    
    with open(yolo_txt_path, 'w') as f:
        for obj in root.iter("object"):
            difficult = obj.find("difficult")
            if difficult is not None:
                difficult = int(difficult.text)
            else:
                difficult = 0
                
            # Class name
            cls = obj.find("name").text
            if cls not in classes or difficult == 1:
                continue
                
            # Convert to label id
            cls_id = classes.index(cls)
            xml_box = obj.find("bndbox")
            box = (float(xml_box.find("xmin").text), float(xml_box.find("ymin").text),
                   float(xml_box.find("xmax").text), float(xml_box.find("ymax").text))
            boxex = convert((w, h), box)
            # YOLO format: class_id x_center y_center width height
            f.write(str(cls_id) + " " + " ".join([str(s) for s in boxex]) + '\n')

def process_dataset(annotations_dir, images_dir, output_base_dir, classes, val_split=0.2):
    """
    Process VOC dataset and create YOLO format dataset
    """
    # Create output directories
    train_images_dir = os.path.join(output_base_dir, 'images', 'train')
    val_images_dir = os.path.join(output_base_dir, 'images', 'val')
    train_labels_dir = os.path.join(output_base_dir, 'labels', 'train')
    val_labels_dir = os.path.join(output_base_dir, 'labels', 'val')
    
    for dir_path in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Get list of xml files
    xml_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]
    random.shuffle(xml_files)
    
    # Split into train and val
    val_count = int(len(xml_files) * val_split)
    val_files = xml_files[:val_count]
    train_files = xml_files[val_count:]
    
    print(f"Total files: {len(xml_files)}")
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    
    # Process train files
    for xml_file in train_files:
        base_name = os.path.splitext(xml_file)[0]
        xml_path = os.path.join(annotations_dir, xml_file)
        img_path = os.path.join(images_dir, f"{base_name}.jpg")
        
        if not os.path.exists(img_path):
            img_path = os.path.join(images_dir, f"{base_name}.png")
            if not os.path.exists(img_path):
                print(f"Warning: Image not found for {xml_file}")
                continue
        
        # Copy image
        shutil.copy(img_path, os.path.join(train_images_dir, os.path.basename(img_path)))
        
        # Convert annotation
        yolo_txt_path = os.path.join(train_labels_dir, f"{base_name}.txt")
        convert_annotation(xml_path, yolo_txt_path, classes)
    
    # Process validation files
    for xml_file in val_files:
        base_name = os.path.splitext(xml_file)[0]
        xml_path = os.path.join(annotations_dir, xml_file)
        img_path = os.path.join(images_dir, f"{base_name}.jpg")
        
        if not os.path.exists(img_path):
            img_path = os.path.join(images_dir, f"{base_name}.png")
            if not os.path.exists(img_path):
                print(f"Warning: Image not found for {xml_file}")
                continue
        
        # Copy image
        shutil.copy(img_path, os.path.join(val_images_dir, os.path.basename(img_path)))
        
        # Convert annotation
        yolo_txt_path = os.path.join(val_labels_dir, f"{base_name}.txt")
        convert_annotation(xml_path, yolo_txt_path, classes)

if __name__ == "__main__":
    # PCB dataset class names
    classes = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']
    
    # Input directories - Updated to match your folder structure
    annotations_dir = "VOC2007/Annotations"  # Relative path from where script is run
    images_dir = "VOC2007/JPEGImages"  # Relative path from where script is run
    
    # Output directory - Updated to match your folder structure
    output_base_dir = "data/PCB"  # This will put converted data in the data/PCB folder
    
    # Process the dataset
    process_dataset(annotations_dir, images_dir, output_base_dir, classes)
    
    print("Dataset conversion completed!")