import cv2
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import os
from PIL import Image, ImageDraw, ImageFont
import json
import requests
import pytesseract
import torch
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO
import base64
from typing import Dict, Any
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tesseract_ocr_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class NepaliTesseractOCR:
    def __init__(self, debug_dir: str = 'debug_output', ollama_url: str = 'http://localhost:11434/api/generate'):
        """Initialize the OCR pipeline with debug output directory"""
        self.debug_dir = Path(debug_dir)
        self.regions_dir = self.debug_dir / 'regions'
        self.preprocessed_dir = self.debug_dir / 'preprocessed'
        self.visualizations_dir = self.debug_dir / 'visualizations'
        
        # Set Ollama URL for Gemma
        self.ollama_url = ollama_url
        
        # Configure session with connection pooling and retries
        self.session = requests.Session()
        retry_strategy = requests.adapters.Retry(
            total=3,  # number of retries
            backoff_factor=0.5,  # wait 0.5, 1, 2... seconds between retries
            status_forcelist=[500, 502, 503, 504]  # HTTP status codes to retry on
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Create debug directories
        for directory in [self.debug_dir, self.regions_dir, self.preprocessed_dir, self.visualizations_dir]:
            try:
                if directory.exists():
                    # Try to remove existing files
                    for file in directory.glob('*'):
                        try:
                            file.unlink()
                        except PermissionError:
                            logging.warning(f"Could not delete file {file} - it may be in use")
                        except Exception as e:
                            logging.warning(f"Error deleting file {file}: {str(e)}")
                    # Try to remove directory
                    try:
                        directory.rmdir()
                    except PermissionError:
                        logging.warning(f"Could not remove directory {directory} - it may be in use")
                    except Exception as e:
                        logging.warning(f"Error removing directory {directory}: {str(e)}")
                
                # Create directory
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logging.error(f"Error setting up directory {directory}: {str(e)}")
        
        # Configure Tesseract path
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Initialize YOLO model for text detection
        try:
            self.yolo_model = YOLO('runs/detect/train4/weights/best.pt')
            logging.info("Custom YOLO model loaded from runs/detect/train4/weights/best.pt")
        except Exception as e:
            logging.error(f"Failed to load YOLO model: {str(e)}")
            self.yolo_model = None

        # Initialize PyTorch model for character recognition
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = models.resnet18(weights=None)
            # Modify first layer for grayscale
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # Modify final layer for character classification
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 46)  # 46 classes for Nepali characters
            )
            
            # Load trained weights
            model_path = 'models/model_latest.pth'
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                logging.info(f"PyTorch model loaded from {model_path}")
            else:
                logging.warning(f"PyTorch model not found at {model_path}, will use Tesseract only")
                self.model = None
        except Exception as e:
            logging.error(f"Failed to initialize PyTorch model: {str(e)}")
            self.model = None
        
        logging.info("OCR pipeline initialized")

    def __del__(self):
        """Cleanup when the object is destroyed"""
        try:
            # Close the session
            if hasattr(self, 'session'):
                self.session.close()
            
            # Close any open files
            for directory in [self.regions_dir, self.preprocessed_dir, self.visualizations_dir]:
                if directory.exists():
                    for file in directory.glob('*'):
                        try:
                            file.unlink()
                        except Exception:
                            pass  # Ignore errors during cleanup
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")

    def preprocess_image(self, image, language='nep'):
        """Preprocess image for better OCR results"""
        if language == 'nep':
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
            # Convert to binary (INVERSE)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(binary, (3, 3), 0)
            # Apply dilation
            kernel = np.ones((2, 2), np.uint8)
            dilated = cv2.dilate(blurred, kernel, iterations=1)
            # Normalize to [0, 255]
            norm_img = cv2.normalize(dilated, None, 0, 255, cv2.NORM_MINMAX)
            return norm_img
        else:
            # For English, no preprocessing
            return image

    def clean_text(self, text):
        """Clean up the extracted text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters and unwanted symbols
        text = re.sub(r'[|_\-=]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove empty lines
        text = '\n'.join(line for line in text.splitlines() if line.strip())
        
        # Fix common OCR mistakes
        text = text.replace('|', 'I')  # Replace vertical bars with I
        text = text.replace('l', 'I')  # Replace lowercase L with I
        
        return text.strip()

    def detect_text_regions(self, image: np.ndarray) -> list:
        """Detect text regions using YOLO model"""
        try:
            if self.yolo_model is None:
                return []

            # Run YOLO detection
            results = self.yolo_model(image)
            
            # Extract bounding boxes
            regions = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    if confidence > 0.5:  # Confidence threshold
                        regions.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence)
                        })
            
            if not regions:
                return []

            # Calculate average height of regions
            heights = [r['bbox'][3] - r['bbox'][1] for r in regions]
            avg_height = sum(heights) / len(heights)
            
            # Group regions into lines using a more robust approach
            # First, sort all regions by y-coordinate
            regions.sort(key=lambda r: r['bbox'][1])
            
            # Initialize lines
            lines = []
            current_line = []
            current_y = None
            line_threshold = avg_height * 0.5  # Use half of average height as threshold
            
            for region in regions:
                y1 = region['bbox'][1]
                
                # If this is the first region or if it's far enough from current line
                if current_y is None or abs(y1 - current_y) > line_threshold:
                    # If we have a current line, add it to lines
                    if current_line:
                        # Sort the line by x-coordinate before adding
                        current_line.sort(key=lambda r: r['bbox'][0])
                        lines.append(current_line)
                    # Start a new line
                    current_line = [region]
                    current_y = y1
                else:
                    # Add to current line
                    current_line.append(region)
                    # Update current_y to be the average of all y-coordinates in the line
                    current_y = sum(r['bbox'][1] for r in current_line) / len(current_line)
            
            # Add the last line if it exists
            if current_line:
                current_line.sort(key=lambda r: r['bbox'][0])
                lines.append(current_line)
            
            # Flatten the lines into a single list
            sorted_regions = [region for line in lines for region in line]
            
            # Log the sorting information
            logging.info(f"Sorted {len(sorted_regions)} regions into {len(lines)} lines")
            for i, line in enumerate(lines):
                if line:  # Only log non-empty lines
                    line_text = [f"({r['bbox'][0]}, {r['bbox'][1]})" for r in line]
                    logging.info(f"Line {i+1}: {', '.join(line_text)}")
            
            return sorted_regions
        except Exception as e:
            logging.error(f"Error in text region detection: {str(e)}")
            return []

    def recognize_character(self, image: np.ndarray) -> str:
        """Recognize character using PyTorch model"""
        try:
            if self.model is None:
                return ""

            # Preprocess image for model
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (32, 32))
            image = torch.from_numpy(image).float() / 255.0
            image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            image = image.to(self.device)

            # Get prediction
            with torch.no_grad():
                output = self.model(image)
                _, predicted = torch.max(output, 1)
                predicted_class = predicted.item()

            # Map class index to character
            # TODO: Implement proper character mapping
            return str(predicted_class)

        except Exception as e:
            logging.error(f"Error in character recognition: {str(e)}")
            return ""

    def extract_text_with_tesseract(self, image: np.ndarray) -> str:
        """Extract text from image using Tesseract OCR"""
        try:
            # Convert to PIL Image for Tesseract
            pil_image = Image.fromarray(image)
            
            # Try with Nepali language first (better for handwriting)
            custom_config = r'--oem 3 --psm 6 -l nep'
            text = pytesseract.image_to_string(pil_image, config=custom_config)
            
            # If no text found, try Hindi
            if not text.strip():
                custom_config = r'--oem 3 --psm 6 -l hin'
                text = pytesseract.image_to_string(pil_image, config=custom_config)
            
            # If still no text found, try English
            if not text.strip():
                custom_config = r'--oem 3 --psm 6 -l eng'
                text = pytesseract.image_to_string(pil_image, config=custom_config)
            
            return text.strip()

        except Exception as e:
            logging.error(f"Error in Tesseract text extraction: {str(e)}")
            return ""

    def visualize_regions(self, image: np.ndarray, regions: list) -> np.ndarray:
        """Visualize detected text regions on the image"""
        try:
            # Create a copy of the image for visualization
            vis_image = image.copy()
            
            # Draw bounding boxes and line numbers
            for i, region in enumerate(regions):
                x1, y1, x2, y2 = region['bbox']
                # Draw rectangle
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Add region number
                cv2.putText(vis_image, str(i+1), (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            return vis_image
        except Exception as e:
            logging.error(f"Error in visualization: {str(e)}")
            return image

    def refine_text_with_gemma(self, prompt: str, image_base64: str = None) -> str:
        """Refine extracted text using Gemma model"""
        try:
            # Check if Ollama is running
            try:
                response = requests.get('http://localhost:11434/api/tags')
                if response.status_code != 200:
                    logging.error("Ollama service is not running")
                    return prompt
                # Check if Gemma3:latest model is available
                models = response.json().get('models', [])
                if not any(model['name'] == 'gemma3:latest' for model in models):
                    logging.error("Gemma model is not available in Ollama")
                    return prompt
            except Exception as e:
                logging.error(f"Error checking Ollama service: {str(e)}")
                return prompt

            # Construct the refinement prompt
            refinement_prompt = f"""You are a Nepali text refinement expert. Your task is to refine the following Nepali text while maintaining its original meaning. Follow these rules strictly:

1. Fix any spelling mistakes and incomplete words
2. Add proper punctuation (पूर्ण विराम, अल्प विराम, प्रश्न चिन्ह, etc.)
3. Ensure proper grammar and sentence structure
4. If a word is incomplete, predict and complete it based on context
5. Maintain the original paragraph structure
6. Do not summarize or change the meaning of the text
7. If the text is not in Nepali, return it as is without modification
8. Remove any special characters and symbols (like = $ # etc) and English numbers (like 1, 2, 3, etc)

Text to refine:
{prompt}

Refined text:"""

            # Call Ollama API
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'gemma3:latest',
                    'prompt': refinement_prompt,
                    'stream': False
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                refined_text = result.get('response', '').strip()
                # If the model indicates it's not Nepali text, return original
                if "not in Nepali" in refined_text.lower():
                    return prompt
                return refined_text
            else:
                logging.error(f"Error from Ollama API: {response.text}")
                return prompt
        except Exception as e:
            logging.error(f"Error in text refinement: {str(e)}")
            return prompt

    def remove_lines(self, image):
        """Remove horizontal and vertical lines from the image"""
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Create a copy of the grayscale image
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Remove horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=3)
        horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=3)
        
        # Remove vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=3)
        vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=3)
        
        # Combine horizontal and vertical lines
        lines = cv2.add(horizontal_lines, vertical_lines)
        
        # Create a mask for the lines
        mask = cv2.bitwise_not(lines)
        
        # Apply the mask to remove lines
        result = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Inpaint the removed lines
        result = cv2.inpaint(result, lines, 3, cv2.INPAINT_TELEA)
        
        return result

    def process_image(self, image_path: str, language: str = 'nep') -> Dict[str, Any]:
        """Process an image through the OCR pipeline"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not read image")
            
            # Convert to RGB for visualization
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if language == 'eng':
                # Preprocess the image for English
                processed = self.preprocess_image(image, language='eng')
                # For English, use direct Tesseract OCR with PSM 6
                pil_image = Image.fromarray(processed)
                custom_config = r'--oem 3 --psm 6 -l eng'
                text = pytesseract.image_to_string(pil_image, config=custom_config)
                text = self.clean_text(text)
                return {
                    'text': text,
                    'regions': 1  # Single region for English
                }
            # For Nepali, use the existing pipeline with YOLO and region detection
            regions = self.detect_text_regions(image)
            logging.info(f"Detected and sorted {len(regions)} text regions")
            vis_image = image.copy()
            current_line = 0
            current_y = None
            line_threshold = None
            extracted_text = []
            for i, region in enumerate(regions):
                x1, y1, x2, y2 = region['bbox']
                if line_threshold is None:
                    heights = [r['bbox'][3] - r['bbox'][1] for r in regions]
                    avg_height = sum(heights) / len(heights)
                    line_threshold = 0.5 * avg_height
                if current_y is None or abs(y1 - current_y) > line_threshold:
                    current_line += 1
                    current_y = y1
                    if extracted_text:
                        extracted_text.append("\n")
                region_image = image[y1:y2, x1:x2]
                preprocessed = self.preprocess_image(region_image, language='nep')
                preprocessed_path = str(self.preprocessed_dir / f"preprocessed_{i+1}_{Path(image_path).stem}.jpg")
                cv2.imwrite(preprocessed_path, preprocessed)
                text = self.extract_text_with_tesseract(preprocessed)
                if text.strip():
                    extracted_text.append(text.strip())
            final_text = " ".join(extracted_text).strip()
            refined_text = self.refine_text_with_gemma(final_text)
            return {
                'text': refined_text,
                'regions': len(regions)
            }
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {str(e)}")
            return {
                'text': '',
                'regions': 0,
                'error': str(e)
            }

def main():
    """Main function to test the OCR pipeline"""
    try:
        # Initialize OCR pipeline
        ocr = NepaliTesseractOCR()
        
        # Process test image
        test_image = "test_image.jpg"  # Replace with your test image path
        if os.path.exists(test_image):
            result = ocr.process_image(test_image)
            print("\nExtracted Text:")
            print(result['text'])
            print(f"\nProcessed {result['regions']} regions")
        else:
            print(f"Test image not found: {test_image}")
            
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main() 