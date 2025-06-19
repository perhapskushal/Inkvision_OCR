from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
from pathlib import Path
from tesseract_ocr_pipeline import NepaliTesseractOCR
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import tempfile
import json
import cv2
import numpy as np
from PIL import Image
import io
import logging
import pytesseract
import base64
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

# Register fonts
font_path = os.path.join(os.path.dirname(__file__), 'fonts', 'NotoSansDevanagari-Regular.ttf')
if not os.path.exists(font_path):
    os.makedirs(os.path.dirname(font_path), exist_ok=True)
    # Download the font if it doesn't exist
    import urllib.request
    urllib.request.urlretrieve(
        'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf',
        font_path
    )
pdfmetrics.registerFont(TTFont('NotoSansDevanagari', font_path))
# Helvetica is a built-in font in ReportLab, no need to register it

# Initialize OCR pipeline
pipeline = NepaliTesseractOCR() 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Check for finetuned models
finetuned_models = {
    'char': Path('out/nep_char.traineddata'),
    'word': Path('out/nep_word.traineddata')
}

for model_type, model_path in finetuned_models.items():
    if model_path.exists():
        logging.info(f"Found finetuned {model_type}-level model at {model_path}")
    else:
        logging.warning(f"Finetuned {model_type}-level model not found at {model_path}")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def create_pdf(text, output_path, language='nep'):
    """Create a PDF with proper text wrapping and formatting"""
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    
    # Set font based on language
    if language == 'nep':
        font_name = 'NotoSansDevanagari'
    else:  # English
        font_name = 'Helvetica'  # Built-in font
    
    c.setFont(font_name, 12)
    
    # Define margins and line spacing
    margin = 50
    line_spacing = 15
    max_width = width - (2 * margin)
    
    # Split text into paragraphs
    paragraphs = text.split('\n')
    
    # Start position
    y = height - margin
    
    for paragraph in paragraphs:
        # Skip empty paragraphs
        if not paragraph.strip():
            y -= line_spacing
            continue
            
        # Split paragraph into words
        words = paragraph.split()
        current_line = []
        current_width = 0
        
        for word in words:
            # Get width of word with a space
            word_width = c.stringWidth(word + ' ', font_name, 12)
            
            # If adding this word would exceed line width, write current line and start new one
            if current_width + word_width > max_width:
                # Write the current line
                c.drawString(margin, y, ' '.join(current_line))
                y -= line_spacing
                
                # Check if we need a new page
                if y < margin:
                    c.showPage()
                    c.setFont(font_name, 12)  # Reset font for new page
                    y = height - margin
                
                # Start new line
                current_line = [word]
                current_width = word_width
            else:
                current_line.append(word)
                current_width += word_width
        
        # Write any remaining words in the last line
        if current_line:
            c.drawString(margin, y, ' '.join(current_line))
            y -= line_spacing
        
        # Add extra space between paragraphs
        y -= line_spacing
        
        # Check if we need a new page
        if y < margin:
            c.showPage()
            c.setFont(font_name, 12)  # Reset font for new page
            y = height - margin
    
    c.save()

def preprocess_image(image):
    """Enhanced preprocessing for better OCR accuracy, especially for handwriting"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding for better handling of varying lighting conditions
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 11, 2)
    
    # Noise removal with bilateral filter to preserve edges
    denoised = cv2.bilateralFilter(thresh, 9, 75, 75)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Combine the enhanced image with the thresholded image
    combined = cv2.bitwise_and(enhanced, denoised)
    
    # Dilation to connect broken characters
    kernel = np.ones((2,2), np.uint8)
    dilated = cv2.dilate(combined, kernel, iterations=1)
    
    # Erosion to remove small noise
    eroded = cv2.erode(dilated, kernel, iterations=1)
    
    return eroded

def extract_text_from_image(image, preprocess=True, language='nep'):
    """Extract text from image with enhanced OCR processing"""
    try:
        # Convert to PIL Image for Tesseract
        pil_image = Image.fromarray(image)
        
        try:
            # Try with specified language first
            custom_config = f'--oem 3 --psm 6 -l {language}'
            text = pytesseract.image_to_string(pil_image, config=custom_config)
        except pytesseract.TesseractError as e:
            # Fallback to English if specified language fails
            if language != 'eng':
                custom_config = r'--oem 3 --psm 6 -l eng'
                text = pytesseract.image_to_string(pil_image, config=custom_config)
            else:
                raise e

        return text
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

def visualize_ocr(image, boxes, texts):
    # Create a copy of the image for visualization
    vis_image = image.copy()
    
    # Draw bounding boxes and text
    for box, text in zip(boxes, texts):
        x, y, w, h = box
        # Draw rectangle
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Add text
        cv2.putText(vis_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return vis_image

def cleanup_debug_output():
    """Clean up debug output directory"""
    debug_dir = Path('debug_output')
    if debug_dir.exists():
        for file in debug_dir.glob('*'):
            try:
                file.unlink()
            except Exception as e:
                logging.error(f"Error deleting file {file}: {str(e)}")

@app.route('/')
def home():
    """Render the home page"""
    cleanup_debug_output()  # Clean up debug output when refreshing
    return render_template('home.html')

@app.route('/about')
def about():
    """Render the about page"""
    cleanup_debug_output()  # Clean up debug output when refreshing
    return render_template('about.html')

@app.route('/features')
def features():
    """Render the features page"""
    cleanup_debug_output()  # Clean up debug output when refreshing
    return render_template('features.html')

@app.route('/ocr')
def ocr():
    """Render the OCR page"""
    cleanup_debug_output()  # Clean up debug output when refreshing
    return render_template('ocr.html')

@app.route('/process', methods=['POST'])
def process_image():
    """Handle image processing and OCR"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Get language from form data, default to 'nep'
        language = request.form.get('language', 'nep')
        
        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Process image through OCR pipeline with language parameter
                result = pipeline.process_image(filepath, language=language)
                
                return jsonify({
                    'text': result['text'],
                    'regions': result['regions']
                })
            except Exception as e:
                logging.error(f"Error processing image: {str(e)}")
                return jsonify({'error': f'Error processing image: {str(e)}'}), 500
            finally:
                # Clean up uploaded file
                try:
                    os.remove(filepath)
                except Exception as e:
                    logging.warning(f"Error removing temporary file {filepath}: {str(e)}")
        else:
            return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        logging.error(f"Error in process_image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/update-pdf', methods=['POST'])
def update_pdf():
    """Update the PDF with edited text"""
    try:
        data = request.get_json()
        if not data or 'text' not in data or 'filename' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
        
        text = data['text']
        filename = secure_filename(data['filename'])
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Create new PDF with updated text
        create_pdf(text, pdf_path)
        
        return jsonify({
            'success': True,
            'pdf_url': url_for('download_pdf', filename=filename)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_pdf(filename):
    """Download the generated PDF file"""
    return send_file(
        os.path.join(app.config['UPLOAD_FOLDER'], filename),
        as_attachment=True,
        download_name=filename
    )

@app.route('/process_image', methods=['POST'])
def process_image_ocr():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Get language from form data, default to 'nep'
        language = request.form.get('language', 'nep')
        
        if not allowed_file(image_file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save uploaded file
        filename = secure_filename(image_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(filepath)
        
        try:
            # Process image through OCR pipeline with language parameter
            result = pipeline.process_image(filepath, language=language)
            
            return jsonify({
                'text': result['text']
            })
            
        except Exception as e:
            logging.error(f"Error processing image: {str(e)}")
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
            
        finally:
            # Clean up uploaded image
            if os.path.exists(filepath):
                os.remove(filepath)
                
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/ocr', methods=['POST'])
def ocr_post():
    """Handle OCR processing"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Get language from form data, default to 'nep'
        language = request.form.get('language', 'nep')
        
        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Process image through OCR pipeline with language parameter
                result = pipeline.process_image(filepath, language=language)
                
                return jsonify({
                    'text': result['text']
                })
                
            except Exception as e:
                logging.error(f"Error processing image: {str(e)}")
                return jsonify({'error': f'Error processing image: {str(e)}'}), 500
            
            finally:
                # Clean up uploaded image
                if os.path.exists(filepath):
                    os.remove(filepath)
                    
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        logging.error(f"Error in OCR processing: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/debug/<path:filename>')
def serve_debug_file(filename):
    """Serve files from the debug output directory"""
    return send_from_directory('debug_output', filename)

@app.route('/debug/preprocessed/<path:filename>')
def serve_preprocessed(filename):
    """Serve preprocessed images"""
    return send_from_directory('debug_output/preprocessed', filename)

@app.route('/debug/regions/<path:filename>')
def serve_regions(filename):
    """Serve region images"""
    return send_from_directory('debug_output/regions', filename)

@app.route('/debug/visualizations/<path:filename>')
def serve_visualizations(filename):
    """Serve visualization images"""
    return send_from_directory('debug_output/visualizations', filename)

@app.route('/generate-pdf', methods=['POST'])
def generate_pdf():
    """Generate PDF from OCR results"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and allowed_file(file.filename):
            language = request.form.get('language', 'nep')
            edited_text = request.form.get('edited_text', None)
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            try:
                result = pipeline.process_image(filepath, language=language)
                # Use edited text if provided, else use refined text
                text_for_pdf = edited_text if edited_text is not None and edited_text.strip() != '' else result['text']
                pdf_filename = f"{os.path.splitext(filename)[0]}.pdf"
                pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
                create_pdf(text_for_pdf, pdf_path, language=language)
                return send_file(
                    pdf_path,
                    as_attachment=True,
                    download_name=pdf_filename
                )
            except Exception as e:
                logging.error(f"Error processing image: {str(e)}")
                return jsonify({'error': f'Error processing image: {str(e)}'}), 500
            finally:
                try:
                    os.remove(filepath)
                except Exception as e:
                    logging.warning(f"Error removing temporary file {filepath}: {str(e)}")
        else:
            return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        logging.error(f"Error in generate_pdf: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/pdf/<path:filename>')
def serve_pdf(filename):
    """Serve PDF files from the pdf directory"""
    return send_from_directory('pdf', filename)

if __name__ == '__main__':
    app.run(debug=True) 