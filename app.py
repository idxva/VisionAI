import os
import cv2
import numpy as np
import base64
import time
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Global variables for model status
model = None
model_loaded = False
model_error = None

# Attempt to load YOLOv8
try:
    from ultralytics import YOLO
    # This will load or download the tiny YOLOv8 model (~6MB)
    print("Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")
    model_loaded = True
    print("YOLOv8 model loaded successfully.")
except Exception as e:
    model_error = str(e)
    print(f"Error loading YOLOv8: {e}")
    print("Cateye will run in fallback mock/simulation mode until YOLOv8 is configured.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({
        "status": "online" if model_loaded else "fallback_mode",
        "model": "YOLOv8-nano" if model_loaded else "mock_detector",
        "error": model_error
    })

def process_image(img):
    """
    Runs inference on the OpenCV image (BGR).
    Returns list of detections with classes, confidence, and bounding boxes.
    """
    if not model_loaded:
        # Fallback Mock Detections (Simulation for testing if model fails to load)
        h, w = img.shape[:2]
        # Simulate simple object detection (e.g. a center bounding box representing 'object')
        # This prevents the app from crashing and keeps it usable.
        time.sleep(0.05)  # Simulate latency
        return [
            {
                "class": "simulated_lens",
                "confidence": 0.95,
                "box": [int(w*0.25), int(h*0.25), int(w*0.75), int(h*0.75)]
            }
        ], 50.0

    start_time = time.time()
    results = model(img, verbose=False)
    inference_time = (time.time() - start_time) * 1000.0  # in ms

    detections = []
    if len(results) > 0:
        result = results[0]
        boxes = result.boxes
        for box in boxes:
            # Get coordinates [x1, y1, x2, y2]
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            detections.append({
                "class": cls_name,
                "confidence": conf,
                "box": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
            })

    return detections, inference_time

@app.route('/detect', methods=['POST'])
def detect():
    try:
        img_bytes = None
        
        # 1. Handle file upload
        if 'image' in request.files:
            file = request.files['image']
            img_bytes = file.read()
        
        # 2. Handle Base64 payload (Webcam data)
        elif request.json and 'image' in request.json:
            b64_data = request.json['image']
            if ',' in b64_data:
                # Remove data url prefix (e.g., "data:image/jpeg;base64,")
                b64_data = b64_data.split(',')[1]
            img_bytes = base64.b64decode(b64_data)
            
        else:
            return jsonify({"status": "error", "message": "No image data provided"}), 400

        # Decode image using OpenCV
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"status": "error", "message": "Failed to decode image"}), 400

        h, w = img.shape[:2]
        detections, inference_time = process_image(img)

        return jsonify({
            "status": "success",
            "detections": detections,
            "width": w,
            "height": h,
            "inference_time_ms": round(inference_time, 2)
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Run server on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
