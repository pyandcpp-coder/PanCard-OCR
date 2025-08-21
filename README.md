# PAN Card OCR API using YOLOv8 & Tesseract

This project provides a robust and efficient FastAPI-based API for extracting key information from Indian PAN (Permanent Account Number) cards. It leverages a custom-trained **YOLOv8** model for object detection and **Tesseract** for Optical Character Recognition (OCR).



## ✨ Features

-   **High Accuracy Detection**: Utilizes a YOLOv8 model trained to detect four key fields: **Name**, **Father's Name**, **Date of Birth (DOB)**, and **PAN Number**.
-   **Robust OCR**: Employs Tesseract OCR with field-specific preprocessing and configurations for accurate text extraction.
-   **Automatic Rotation Correction**: Intelligently detects the orientation of the PAN card (0°, 90°, 180°, 270°) and corrects it for optimal detection.
-   **Multiple Input Methods**: Offers flexible API endpoints to process images from:
    -   File Upload
    -   Base64 Encoded String
    -   Local File Path or Remote URL
-   **Dynamic Hardware Support**: Automatically uses an NVIDIA GPU (CUDA) if available for faster processing, otherwise falls back to the CPU.
-   **Clean & Validated Output**: Post-processes and validates extracted data (e.g., using Regex for PAN and DOB formats) to provide a clean JSON response.
-   **Asynchronous API**: Built with FastAPI for high performance and scalability.

## 🛠️ Installation & Setup

Follow these steps to set up and run the project locally.

### 1. Prerequisites

-   **Python 3.8+**
-   **Tesseract OCR Engine**: You must install Tesseract on your system and ensure it's available in your system's PATH.
    -   **Windows**: Download and run the installer from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki). Remember to add the installation path to your `PATH` environment variable.
    -   **macOS**: `brew install tesseract`
    -   **Linux (Ubuntu/Debian)**: `sudo apt-get update && sudo apt-get install tesseract-ocr`

### 2. Clone the Repository

```bash
git clone [https://github.com/pyandcpp-coder/PanCard-OCR.git](https://github.com/pyandcpp-coder/PanCard-OCR.git)
cd PanCard-OCR
```
### 3. Set Up Virtual Environment


# Create a virtual environment
python3 -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate


### Install Depedencies
```bash
pip install -r requirement.txt
```
# 📡 API Endpoints

The application provides the following endpoints for PAN card processing.

---

## 1. Health Check

**Endpoint:** `GET /health`

**Description:** Checks the status of the API and provides information about the environment.

**cURL Example:**

```bash
curl -X GET "http://localhost:8000/health"
```

**Success Response (200 OK):**

```json
{
  "status": "healthy",
  "device": "cpu",
  "torch_version": "2.1.0",
  "cuda_available": false
}
```

---

## 2. Process via File Upload

**Endpoint:** `POST /process_pan`

**Description:** Upload a PAN card image file for processing.

**Request:** `multipart/form-data`

**cURL Example:**

```bash
curl -X POST "http://localhost:8000/process_pan" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/pancard.jpg"
```

---

## 3. Process via Base64 String

**Endpoint:** `POST /process_pan_base64`

**Description:** Send a base64 encoded image string in a JSON payload.

**Request Body:**

```json
{
  "image_base64": "..."
}
```

**cURL Example:**

```bash
curl -X POST "http://localhost:8000/process_pan_base64" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"image_base64": "/9j/4AAQSkZJRgABAQ..."}'
```

---

## 4. Process via Path or URL

**Endpoint:** `POST /process_pan_path`

**Description:** Provide a local file path or a public URL to an image.

**Request Body:**

```json
{
  "image_path": "https://example.com/pan_card.png"
}
```

**cURL Example:**

```bash
curl -X POST "http://localhost:8000/process_pan_path" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"image_path": "/path/to/local/image.jpg"}'
```

---

## 📌 Sample Success Response

For all processing endpoints, a successful request will return a `200 OK` status with the extracted data in JSON format:

```json
{
  "pan": "ABCDE1234F",
  "name": "First_Last",
  "father": "Fathers_First_Last",
  "dob": "01_01_1990",
  "status": true
}
```

If a field cannot be extracted, its value will be **null**. The `status` field will be `true` if at least one field was successfully extracted, otherwise `false`.
