import cv2
import numpy as np
from tesseract_ocr_pipeline import NepaliTesseractOCR
import logging
from PIL import Image
import pytesseract

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def preprocess_image(image):
    """Enhanced preprocessing for better OCR accuracy"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply bilateral filter for noise reduction while preserving edges
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        filtered,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    
    # Noise removal
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    return opening

def main():
    # Initialize OCR pipeline
    pipeline = NepaliTesseractOCR()
    
    # Path to the test image
    image_path = "dataset/test/images/88.jpg"
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # Preprocess the image
    preprocessed = preprocess_image(image)
    
    # Save preprocessed image for debugging
    cv2.imwrite("preprocessed_88.jpg", preprocessed)
    
    # Process the image with the pipeline
    result = pipeline.process_image(image_path)
    
    # Print the extracted text
    print("\nExtracted Text:")
    print(result['full_text'])
    
    # Print debug information
    print("\nDebug Information:")
    print(f"Number of regions: {len(result['regions'])}")
    
    # Print individual region results
    print("\nRegion Results:")
    for region in result['regions']:
        print(f"Region {region['region_id']}:")
        print(f"Text: {region['text']}")
        print(f"Method: {region['method']}")
        print("---")
    
    # Try direct Tesseract OCR with Nepali language
    print("\nDirect Tesseract OCR Result:")
    custom_config = r'--oem 3 --psm 6 -l nep'
    text = pytesseract.image_to_string(preprocessed, config=custom_config)
    print(text)

if __name__ == "__main__":
    main() 