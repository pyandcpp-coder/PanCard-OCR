import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.block import C2f, SPPF
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect
import pytesseract
from PIL import Image
import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64
from io import BytesIO
import re
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add safe globals for YOLO model loading
torch.serialization.add_safe_globals([
    DetectionModel,
    C2f,
    SPPF,
    Conv,
    Detect
])

# Pydantic models for request bodies
class Base64ImageRequest(BaseModel):
    image_base64: str

class ImagePathRequest(BaseModel):
    image_path: str
    save_detections: bool = True

class DocumentProcessor:
    def __init__(self, model_path='model/best_pancard.pt', output_dir='detections'):
        """
        Initialize the document processor
        """
        self.model = YOLO(model_path)
        self.output_dir = output_dir
        self.create_output_directories()
        
        # OCR configuration for better accuracy
        # self.ocr_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
# Field-specific OCR configs
        self.ocr_configs = {
            "pan": r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            "name": r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ',
            "father": r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ',
            "dob": r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789/-'
        }


    def create_output_directories(self):
        """Create necessary output directories"""
        directories = [
            self.output_dir,
            f"{self.output_dir}/rotated",
            f"{self.output_dir}/cropped",
            f"{self.output_dir}/processed"
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    def detect_rotation_angle(self, image):
        """
        Detect rotation angle using text line detection
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply morphological operations to find text lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        morphed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        angles = []
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            angle = rect[2]

            # Normalize angle to [-45, 45]
            if angle < -45:
                angle += 90

            # Only keep angles that are small deviations (not -90 false cases)
            if -15 <= angle <= 15:
                angles.append(angle)

        if angles:
            return -float(np.median(angles))  # cast to Python float
        return 0.0
    
    def rotate_image(self, image, angle):
        """
        Rotate image by given angle
        """
        if abs(angle) < 1:  # Skip rotation if angle is very small
            return image
            
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new image dimensions
        cos = abs(rotation_matrix[0, 0])
        sin = abs(rotation_matrix[0, 1])
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))
        
        # Adjust rotation matrix for new center
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Perform rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
    
    def preprocess_image_for_ocr(self, image):
        """
        Strong preprocessing for better OCR
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize up to help Tesseract
        scale = 2
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Apply bilateral filter (denoise while keeping edges)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10
        )

        # Morphological opening (remove noise dots)
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        return processed

    
    def extract_text_with_filters(self, image, field_type):
        """
        Extract text with field-specific filters
        """
        # Preprocess image
        processed = self.preprocess_image_for_ocr(image)

        # Pick config based on field_type (fallback to generic if missing)
        config = self.ocr_configs.get(
            field_type,
            r'--oem 3 --psm 6'
        )

        # Extract text using OCR
        text = pytesseract.image_to_string(processed, config=config).strip()

        # Apply field-specific filters
        if field_type == 'pan':
            pan_match = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]', text.upper())
            return pan_match.group() if pan_match else text.upper()

        elif field_type in ['name', 'father']:
            cleaned = re.sub(r'[^a-zA-Z\s]', '', text)
            return ' '.join(cleaned.split()).title()

        elif field_type == 'dob':
            date_patterns = [
                r'\d{2}[/-]\d{2}[/-]\d{4}',
                r'\d{2}\.\d{2}\.\d{4}',
                r'\d{1,2}[/-]\d{1,2}[/-]\d{4}'
            ]
            for pattern in date_patterns:
                date_match = re.search(pattern, text)
                if date_match:
                    return date_match.group()
            return text.strip()

        return text.strip()

    
    def process_document(self, image_path, save_detections=True):
        """
        Main processing pipeline
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Step 1: Detect and correct rotation
            rotation_angle = self.detect_rotation_angle(image)
            logger.info(f"Detected rotation angle: {rotation_angle:.2f} degrees")
            
            rotated_image = self.rotate_image(image, rotation_angle)
            
            if save_detections:
                rotated_path = f"{self.output_dir}/rotated/rotated_image.jpg"
                cv2.imwrite(rotated_path, rotated_image)
            
            # Step 2: Run YOLO detection
            results = self.model.predict(
                source=rotated_image,
                conf=0.25,
                iou=0.45,
                save=False
            )
            
            # Step 3: Extract detected regions and perform OCR
            extracted_data = {}
            names = self.model.names
            
            for i, r in enumerate(results):
                for j, box in enumerate(r.boxes):
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    class_name = names[class_id]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Crop detected region
                    cropped_region = rotated_image[y1:y2, x1:x2]
                    
                    if save_detections:
                        crop_path = f"{self.output_dir}/cropped/{class_name}_{j}_{confidence:.2f}.jpg"
                        cv2.imwrite(crop_path, cropped_region)
                    
                    # Extract text with field-specific filters
                    extracted_text = self.extract_text_with_filters(cropped_region, class_name)
                    
                    # Store the best detection for each class (highest confidence)
                    if class_name not in extracted_data or confidence > extracted_data[class_name]['confidence']:
                        extracted_data[class_name] = {
                            'text': extracted_text,
                            'confidence': confidence,
                            'bbox': [x1, y1, x2, y2]
                        }
                    
                    logger.info(f"Detected {class_name}: {extracted_text} (confidence: {confidence:.3f})")
            
            final_output = {}
            for field in ['pan', 'name', 'father', 'dob']:
                if field in extracted_data:
                    final_output[field] = extracted_data[field]['text']
                else:
                    final_output[field] = ""

            # Add metadata
            final_output['processing_info'] = {
                'rotation_corrected': bool(abs(rotation_angle) > 1),  # cast to Python bool
                'rotation_angle': float(round(float(rotation_angle), 2)),  # ensure float
                'detections_count': int(len(extracted_data)),  # ensure int
                'timestamp': datetime.now().isoformat()
            }

            return final_output

        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return {'error': str(e)}

# Initialize FastAPI app
app = FastAPI(title="Document Processing API", version="1.0.0")
processor = DocumentProcessor()

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Document processing API is running"}

@app.post("/process_document")
async def process_document_file(file: UploadFile = File(...)):
    """
    Process document from file upload
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            raise HTTPException(status_code=400, detail="Invalid file format")
        
        # Save uploaded file temporarily
        temp_path = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process document
        result = processor.process_document(temp_path)
        
        # Clean up temporary file
        os.remove(temp_path)
        
        return result
    
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/process_document_base64")
async def process_document_base64(request: Base64ImageRequest):
    """
    Process document from base64 encoded image
    """
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)
        
        # Save base64 image temporarily
        temp_path = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        with open(temp_path, 'wb') as f:
            f.write(image_data)
        
        # Process document
        result = processor.process_document(temp_path)
        
        # Clean up temporary file
        os.remove(temp_path)
        
        return result
    
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/process_document_path")
async def process_document_path(request: ImagePathRequest):
    """
    Process document from file path
    """
    try:
        result = processor.process_document(request.image_path, request.save_detections)
        return result
    
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    
    # Ensure model file exists
    if not os.path.exists('model/best_pancard.pt'):
        print("Warning: Model file 'best_pancard.pt' not found!")
        print("Please make sure the model file is in the current directory")
    
    # Start the API server with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)