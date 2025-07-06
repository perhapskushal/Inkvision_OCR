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

# Configure Tesseract path for Render/Linux
# Render uses Linux, so we use the standard Linux path
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

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
        # Convert box to integer coordinates
        box = np.array(box).astype(np.int32)
        
        # Draw bounding box
        cv2.polylines(vis_image, [box], True, (0, 255, 0), 2)
        
        # Add text label
        x, y = box[0]
        cv2.putText(vis_image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return vis_image

def cleanup_debug_output():
    """Clean up debug output directory"""
    debug_dir = Path('debug_output')
    if debug_dir.exists():
        import shutil
        shutil.rmtree(debug_dir)
    debug_dir.mkdir(exist_ok=True)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/ocr')
def ocr():
    return render_template('ocr.html')

@app.route('/process', methods=['POST'])
def process_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Read and process image
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Preprocess image
        preprocessed = preprocess_image(image)
        
        # Extract text using Tesseract
        extracted_text = extract_text_from_image(preprocessed, language='nep')
        
        # Generate unique filename for PDF
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pdf_filename = f'extracted_text_{timestamp}.pdf'
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
        
        # Create PDF
        create_pdf(extracted_text, pdf_path)
        
        return jsonify({
            'success': True,
            'text': extracted_text,
            'pdf_filename': pdf_filename
        })
        
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/update-pdf', methods=['POST'])
def update_pdf():
    try:
        data = request.get_json()
        text = data.get('text', '')
        pdf_filename = data.get('pdf_filename', '')
        
        if not pdf_filename:
            return jsonify({'error': 'PDF filename not provided'}), 400
        
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
        create_pdf(text, pdf_path)
        
        return jsonify({'success': True, 'message': 'PDF updated successfully'})
        
    except Exception as e:
        logging.error(f"Error updating PDF: {str(e)}")
        return jsonify({'error': f'PDF update error: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_pdf(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)
    except Exception as e:
        return jsonify({'error': f'File not found: {str(e)}'}), 404

@app.route('/process_image', methods=['POST'])
def process_image_ocr():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Read image
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Process with OCR pipeline
        try:
            # Use the pipeline for advanced processing
            result = pipeline.process_image(image)
            extracted_text = result.get('text', '')
            
            # If pipeline fails, fallback to basic OCR
            if not extracted_text:
                preprocessed = preprocess_image(image)
                extracted_text = extract_text_from_image(preprocessed, language='nep')
                
        except Exception as pipeline_error:
            logging.warning(f"Pipeline failed, using fallback: {pipeline_error}")
            preprocessed = preprocess_image(image)
            extracted_text = extract_text_from_image(preprocessed, language='nep')
        
        return jsonify({
            'success': True,
            'text': extracted_text
        })
        
    except Exception as e:
        logging.error(f"Error in process_image_ocr: {str(e)}")
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/ocr', methods=['POST'])
def ocr_post():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Read image
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Clean up debug output
        cleanup_debug_output()
        
        # Process with OCR pipeline
        try:
            result = pipeline.process_image(image)
            extracted_text = result.get('text', '')
            
            # Generate debug visualizations if available
            debug_info = {}
            if 'debug_info' in result:
                debug_info = result['debug_info']
            
        except Exception as pipeline_error:
            logging.warning(f"Pipeline failed, using fallback: {pipeline_error}")
            preprocessed = preprocess_image(image)
            extracted_text = extract_text_from_image(preprocessed, language='nep')
            debug_info = {}
        
        return jsonify({
            'success': True,
            'text': extracted_text,
            'debug_info': debug_info
        })
        
    except Exception as e:
        logging.error(f"Error in ocr_post: {str(e)}")
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/debug/<path:filename>')
def serve_debug_file(filename):
    return send_from_directory('debug_output', filename)

@app.route('/debug/preprocessed/<path:filename>')
def serve_preprocessed(filename):
    return send_from_directory('debug_output/preprocessed', filename)

@app.route('/debug/regions/<path:filename>')
def serve_regions(filename):
    return send_from_directory('debug_output/regions', filename)

@app.route('/debug/visualizations/<path:filename>')
def serve_visualizations(filename):
    return send_from_directory('debug_output/visualizations', filename)

@app.route('/generate-pdf', methods=['POST'])
def generate_pdf():
    try:
        data = request.get_json()
        text = data.get('text', '')
        language = data.get('language', 'nep')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pdf_filename = f'generated_text_{timestamp}.pdf'
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
        
        # Create PDF
        create_pdf(text, pdf_path, language)
        
        return jsonify({
            'success': True,
            'pdf_filename': pdf_filename,
            'message': 'PDF generated successfully'
        })
        
    except Exception as e:
        logging.error(f"Error generating PDF: {str(e)}")
        return jsonify({'error': f'PDF generation error: {str(e)}'}), 500

@app.route('/pdf/<path:filename>')
def serve_pdf(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        return jsonify({'error': f'File not found: {str(e)}'}), 404

@app.route('/health')
def health_check():
    """Health check endpoint for Render"""
    return jsonify({'status': 'healthy', 'message': 'Flask app is running with Tesseract'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port) 