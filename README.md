# Nepali OCR Web Application

A web-based OCR application for Nepali text recognition using Tesseract OCR.

## Features

- Upload and process images containing Nepali text
- Real-time text extraction with visualization
- Support for both original and preprocessed images
- Download results as PDF
- Modern web interface

## Requirements

- Python 3.8 or higher
- Tesseract OCR with Nepali language support

## Installation

1. Install Tesseract OCR:

   For Windows (using chocolatey):
   ```bash
   choco install tesseract
   choco install tesseract-lang
   ```

   For Linux:
   ```bash
   sudo apt-get update
   sudo apt-get install tesseract-ocr
   sudo apt-get install tesseract-ocr-nep
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the Flask application:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Upload an image containing Nepali text
4. View the extracted text and visualization
5. Download the results as PDF if needed

## Project Structure

```
.
├── app.py                 # Flask application
├── tesseract_ocr_pipeline.py  # OCR processing pipeline
├── requirements.txt       # Python dependencies
├── templates/            # HTML templates
│   ├── home.html
│   ├── about.html
│   ├── features.html
│   └── ocr.html
├── static/              # Static files (CSS, JS)
├── uploads/            # Temporary upload directory
└── debug_output/       # Debug visualization output
```

## Notes

- The application uses Tesseract OCR with Nepali language support
- Images are preprocessed to improve OCR accuracy
- Results are saved in the debug_output directory for reference
- Maximum file size is limited to 16MB

## License

This project is licensed under the MIT License - see the LICENSE file for details. 