import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
from tqdm import tqdm

class VOCtoYOLOConverter:
    def __init__(self, voc_dir, output_dir):
        self.voc_dir = Path(voc_dir)
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / 'images'
        self.labels_dir = self.output_dir / 'labels'
        
        # Create output directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
    def convert_bbox(self, bbox, img_width, img_height):
        """Convert VOC bbox to YOLO format"""
        x_min, y_min, x_max, y_max = bbox
        
        # Calculate center, width, height
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min
        
        # Normalize
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height
        
        return [x_center, y_center, width, height]
    
    def parse_voc_xml(self, xml_path):
        """Parse VOC XML file and return image info and bboxes"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image dimensions
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
        
        # Get all text region bounding boxes
        bboxes = []
        for obj in root.findall('object'):
            # Only process text regions
            if obj.find('name').text == 'text':
                bbox = obj.find('bndbox')
                x_min = float(bbox.find('xmin').text)
                y_min = float(bbox.find('ymin').text)
                x_max = float(bbox.find('xmax').text)
                y_max = float(bbox.find('ymax').text)
                bboxes.append([x_min, y_min, x_max, y_max])
        
        return img_width, img_height, bboxes
    
    def convert_dataset(self):
        """Convert entire VOC dataset to YOLO format"""
        xml_files = list(self.voc_dir.glob('*.xml'))
        
        for xml_file in tqdm(xml_files, desc="Converting VOC to YOLO"):
            # Get corresponding image file
            img_file = xml_file.with_suffix('.jpg')
            if not img_file.exists():
                continue
                
            # Parse XML
            img_width, img_height, bboxes = self.parse_voc_xml(xml_file)
            
            # Convert bboxes to YOLO format
            yolo_bboxes = [self.convert_bbox(bbox, img_width, img_height) 
                          for bbox in bboxes]
            
            # Write YOLO format labels
            label_file = self.labels_dir / f"{xml_file.stem}.txt"
            with open(label_file, 'w') as f:
                for bbox in yolo_bboxes:
                    f.write(f"0 {' '.join(map(str, bbox))}\n")
            
            # Copy image to output directory
            shutil.copy2(img_file, self.images_dir / img_file.name)

def main():
    # Convert training set
    train_converter = VOCtoYOLOConverter('train', 'dataset/train')
    train_converter.convert_dataset()
    
    # Convert test set
    test_converter = VOCtoYOLOConverter('test', 'dataset/test')
    test_converter.convert_dataset()

if __name__ == "__main__":
    main() 