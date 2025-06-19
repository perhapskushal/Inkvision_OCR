import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_validation.log'),
        logging.StreamHandler()
    ]
)

class DatasetValidator:
    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        self.train_dir = self.dataset_dir / 'train'
        self.test_dir = self.dataset_dir / 'test'
        
    def extract_letter_from_filename(self, filename):
        """Extract the letter from the image filename"""
        # Assuming filename format contains the letter somewhere
        # Modify this based on your actual filename format
        return filename.split('_')[0]  # Adjust this based on your naming convention
    
    def validate_dataset(self):
        """Validate the dataset by comparing image names with their labels"""
        logging.info("Starting dataset validation...")
        
        # Process both train and test sets
        for dataset_type in ['train', 'test']:
            dataset_path = self.dataset_dir / dataset_type
            images_dir = dataset_path / 'images'
            labels_dir = dataset_path / 'labels'
            
            if not images_dir.exists() or not labels_dir.exists():
                logging.error(f"Missing required directories for {dataset_type} set")
                continue
                
            logging.info(f"\nProcessing {dataset_type} set...")
            
            # Get all image files
            image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
            
            for img_path in tqdm(image_files, desc=f"Validating {dataset_type} images"):
                # Get corresponding label file
                label_path = labels_dir / f"{img_path.stem}.txt"
                
                if not label_path.exists():
                    logging.warning(f"Missing label file for {img_path.name}")
                    continue
                
                # Extract letter from filename
                expected_letter = self.extract_letter_from_filename(img_path.stem)
                
                # Read the image to verify it's valid
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        logging.error(f"Failed to read image: {img_path}")
                        continue
                except Exception as e:
                    logging.error(f"Error reading image {img_path}: {str(e)}")
                    continue
                
                # Read the label file
                try:
                    with open(label_path, 'r') as f:
                        label_content = f.read().strip()
                        if not label_content:
                            logging.warning(f"Empty label file for {img_path.name}")
                            continue
                except Exception as e:
                    logging.error(f"Error reading label file {label_path}: {str(e)}")
                    continue
                
                # Log the comparison
                logging.info(f"Image: {img_path.name}")
                logging.info(f"Expected letter: {expected_letter}")
                logging.info(f"Label content: {label_content}")
                logging.info("-" * 50)

def main():
    validator = DatasetValidator('dataset')
    validator.validate_dataset()

if __name__ == "__main__":
    main() 