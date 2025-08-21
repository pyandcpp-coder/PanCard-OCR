import os
import cv2
import numpy as np
import torch
import pytesseract
from PIL import Image
import json
import re
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import base64
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Base64ImageRequest(BaseModel):
    image_base64: str

class ImagePathRequest(BaseModel):
    image_path: str

class CleanPANProcessor:
    def __init__(self, model_path='model/best_pancard.pt'):
        """Initialize the PAN processor with clean and simple extraction"""
        
        # Dynamic device selection
        if torch.cuda.is_available():
            self.device = "cuda"
            logger.info(f"✅ CUDA is available. Model will run on GPU ({torch.cuda.get_device_name(0)}).")
        else:
            self.device = "cpu"
            logger.info("⚠️ CUDA not available. Model will fall back to CPU.")
        
        # Load YOLO model
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.model = YOLO(model_path)
        logger.info(f"✅ YOLO model loaded successfully from {model_path}")
        
        self._check_tesseract()
    
    def _check_tesseract(self):
        """Check if Tesseract is properly installed"""
        try:
            pytesseract.get_tesseract_version()
            logger.info("✅ Tesseract OCR is available")
        except pytesseract.TesseractNotFoundError:
            logger.critical("❌ Tesseract not found. Please install Tesseract OCR.")
            raise RuntimeError("Tesseract not found")
    
    def _try_detection_with_rotation(self, image: np.ndarray) -> Tuple[Any, np.ndarray, int]:
        """
        Try YOLO detection with automatic rotation correction
        """
        # First try without rotation
        results = self.model.predict(
            source=image,
            conf=0.25,
            iou=0.45,
            save=False,
            device=self.device,
            verbose=False
        )
        
        # Check detection quality
        detection_count = 0
        avg_confidence = 0
        
        if results and len(results) > 0:
            for box in results[0].boxes:
                class_name = self.model.names[int(box.cls)]
                if class_name in ['pan', 'name', 'father', 'dob']:
                    detection_count += 1
                    avg_confidence += float(box.conf)
            
            if detection_count > 0:
                avg_confidence /= detection_count
        
        # If good detections, use them
        if detection_count >= 3 and avg_confidence > 0.6:
            logger.info(f"✅ Good detections without rotation: {detection_count} fields, avg conf: {avg_confidence:.3f}")
            return results, image, 0
        
        logger.info(f"⚠️ Initial detections: {detection_count} fields, trying rotations...")
        
        # Try different rotations
        best_results = results
        best_image = image
        best_rotation = 0
        best_score = detection_count * avg_confidence
        
        for angle in [90, 180, 270]:
            if angle == 90:
                rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rotated = cv2.rotate(image, cv2.ROTATE_180)
            elif angle == 270:
                rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            test_results = self.model.predict(
                source=rotated,
                conf=0.25,
                iou=0.45,
                save=False,
                device=self.device,
                verbose=False
            )
            
            test_count = 0
            test_conf = 0
            
            if test_results and len(test_results) > 0:
                for box in test_results[0].boxes:
                    class_name = self.model.names[int(box.cls)]
                    if class_name in ['pan', 'name', 'father', 'dob']:
                        test_count += 1
                        test_conf += float(box.conf)
                
                if test_count > 0:
                    test_conf /= test_count
            
            test_score = test_count * test_conf
            
            if test_score > best_score:
                best_score = test_score
                best_results = test_results
                best_image = rotated
                best_rotation = angle
        
        if best_rotation != 0:
            logger.info(f"✅ Using {best_rotation}° rotation")
        
        return best_results, best_image, best_rotation
    
    def _simple_preprocess(self, image: np.ndarray, enhance_contrast: bool = True) -> np.ndarray:
        """Simple and effective preprocessing"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Scale up if too small
        h, w = gray.shape[:2]
        if h < 50:
            scale = 100 / h
            new_w, new_h = int(w * scale), int(h * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Enhance contrast if needed
        if enhance_contrast:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
        
        return gray
    
    def _extract_pan(self, image: np.ndarray) -> str:
        """Extract PAN number with strict validation"""
        # Try different preprocessing
        images_to_try = [
            self._simple_preprocess(image, enhance_contrast=False),
            self._simple_preprocess(image, enhance_contrast=True),
        ]
        
        for img in images_to_try:
            # Try different PSM modes
            for psm in [8, 7, 13]:
                try:
                    # With whitelist
                    config = f'--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    text = pytesseract.image_to_string(img, config=config).strip()
                    
                    # Extract valid PAN
                    clean = text.upper().replace(' ', '').replace('\n', '')
                    pan_match = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]', clean)
                    if pan_match:
                        return pan_match.group()
                    
                    # Without whitelist
                    text = pytesseract.image_to_string(img, config=f'--psm {psm}').strip()
                    clean = text.upper().replace(' ', '').replace('\n', '')
                    pan_match = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]', clean)
                    if pan_match:
                        return pan_match.group()
                        
                except:
                    continue
        
        return ""
    
    def _extract_name(self, image: np.ndarray) -> str:
        """Extract name with better configuration"""
        best_name = ""
        
        # Preprocess
        gray = self._simple_preprocess(image, enhance_contrast=True)
        
        # Try different PSM modes for names
        for psm in [7, 6, 8, 11]:
            try:
                # OCR without restrictive whitelist
                config = f'--psm {psm} -l eng'
                text = pytesseract.image_to_string(gray, config=config).strip()
                
                # Clean the text
                # Remove non-alphabetic characters except spaces
                cleaned = re.sub(r'[^a-zA-Z\s]', '', text)
                
                # Remove single letters and very short words (likely noise)
                words = []
                for word in cleaned.split():
                    if len(word) > 2:  # Keep words with more than 2 characters
                        words.append(word)
                    elif word.upper() in ['JR', 'SR', 'II', 'III']:  # Keep valid short words
                        words.append(word)
                
                if words:
                    name = ' '.join(words)
                    # Check if this looks like a valid name
                    if len(words) <= 5 and len(name) > len(best_name):  # Names usually have 1-5 words
                        best_name = name
                        
            except:
                continue
        
        # Format the name properly
        if best_name:
            # Title case and replace spaces with underscores
            return '_'.join(word.title() for word in best_name.split())
        
        return ""
    
    def _extract_dob(self, image: np.ndarray) -> str:
        """Extract date of birth"""
        # Preprocess
        gray = self._simple_preprocess(image, enhance_contrast=True)
        
        for psm in [7, 8, 6]:
            try:
                # Try with whitelist
                config = f'--psm {psm} -c tessedit_char_whitelist=0123456789/-'
                text = pytesseract.image_to_string(gray, config=config).strip()
                
                # Look for date pattern
                date_match = re.search(r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})', text)
                if date_match:
                    day, month, year = date_match.groups()
                    day, month, year = int(day), int(month), int(year)
                    if 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2010:
                        # Return with underscores
                        return f"{day:02d}_{month:02d}_{year}"
                
                # Try without whitelist
                text = pytesseract.image_to_string(gray, config=f'--psm {psm}').strip()
                date_match = re.search(r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})', text)
                if date_match:
                    day, month, year = date_match.groups()
                    day, month, year = int(day), int(month), int(year)
                    if 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2010:
                        return f"{day:02d}_{month:02d}_{year}"
                        
            except:
                continue
        
        return ""
    
    def process_pan_card(self, image_path: str) -> Dict[str, str]:
        """Main processing pipeline for PAN card"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            logger.info(f"Processing PAN card from: {image_path}")
            
            # Try detection with automatic rotation
            results, corrected_image, rotation_used = self._try_detection_with_rotation(image)
            
            if rotation_used != 0:
                logger.info(f"🔄 Image rotated by {rotation_used}°")
            
            # Initialize output
            output = {
                "pan": "",
                "name": "",
                "father": "",
                "dob": ""
            }
            
            # Process detections
            best_detections = {}
            
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    class_name = self.model.names[class_id]
                    
                    if class_name not in ['pan', 'name', 'father', 'dob']:
                        continue
                    
                    if class_name not in best_detections or confidence > best_detections[class_name]['confidence']:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        
                        # Add padding
                        padding = 10
                        y1 = max(0, y1 - padding)
                        y2 = min(corrected_image.shape[0], y2 + padding)
                        x1 = max(0, x1 - padding)
                        x2 = min(corrected_image.shape[1], x2 + padding)
                        
                        best_detections[class_name] = {
                            'bbox': (x1, y1, x2, y2),
                            'confidence': confidence
                        }
            
            # Extract text from each field
            for field_name, detection in best_detections.items():
                x1, y1, x2, y2 = detection['bbox']
                cropped = corrected_image[y1:y2, x1:x2].copy()
                
                # Use appropriate extraction method
                if field_name == 'pan':
                    extracted_text = self._extract_pan(cropped)
                elif field_name in ['name', 'father']:
                    extracted_text = self._extract_name(cropped)
                elif field_name == 'dob':
                    extracted_text = self._extract_dob(cropped)
                else:
                    extracted_text = ""
                
                output[field_name] = extracted_text
                logger.info(f"✅ {field_name}: {extracted_text if extracted_text else 'Not found'} (conf: {detection['confidence']:.3f})")
            output["status"] = any(v is not None for k, v in output.items() if k != "status")

            return output
            
        except Exception as e:
            logger.error(f"Error processing PAN card: {str(e)}")
            return {"pan": "", "name": "", "father": "", "dob": "","status":False}

# FastAPI app setup
app = FastAPI(title="Clean PAN Card Processing API", version="6.0.0")
processor = None

@app.on_event("startup")
async def startup_event():
    global processor
    try:
        processor = CleanPANProcessor()
        logger.info("✅ PAN processor initialized successfully")
    except Exception as e:
        logger.critical(f"❌ Failed to initialize processor: {e}")
        raise

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": processor.device if processor else "not_initialized",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available()
    }

@app.post("/process_pan")
async def process_pan_file(file: UploadFile = File(...)):
    """Process PAN card from file upload"""
    try:
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            raise HTTPException(status_code=400, detail="Invalid file format")
        
        temp_path = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        result = processor.process_pan_card(temp_path)
        
        os.remove(temp_path)
        
        return result
    
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/process_pan_base64")
async def process_pan_base64(request: Base64ImageRequest):
    """Process PAN card from base64 encoded image"""
    try:
        image_data = base64.b64decode(request.image_base64)
        
        temp_path = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        with open(temp_path, 'wb') as f:
            f.write(image_data)
        
        result = processor.process_pan_card(temp_path)
        
        os.remove(temp_path)
        
        return result
    
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/process_pan_path")
async def process_pan_path(request: ImagePathRequest):
    """Process PAN card from file path or URL"""
    try:
        import requests
        if request.image_path.startswith("http://") or request.image_path.startswith("https://"):
            response = requests.get(request.image_path, timeout=10)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to fetch image from URL")
            
            temp_path = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            result = processor.process_pan_card(temp_path)
            os.remove(temp_path)
        else:
            result = processor.process_pan_card(request.image_path)

        return result
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

if __name__ == '__main__':
    if not os.path.exists('model/best_pancard.pt'):
        print("❌ Warning: Model file 'model/best_pancard.pt' not found!")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)