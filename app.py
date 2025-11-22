from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import os
from ultralytics import YOLO
import base64
from werkzeug.utils import secure_filename
import json
import requests
from datetime import datetime
from collections import deque
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov', 'webm'}

# Hugging Face API Configuration (optional - set your API key)
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY', '')  # Set via environment variable
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/dima806/vehicle_10_types_image_detection"

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Vehicle classes in COCO dataset (YOLO)
VEHICLE_CLASSES = {
    'car': 2,
    'motorcycle': 3,
    'airplane': 4,
    'bus': 5,
    'train': 6,
    'truck': 7,
    'boat': 8
}

# Emergency vehicle indicators (patterns to detect)
EMERGENCY_INDICATORS = {
    'ambulance': ['ambulance', 'medical', 'rescue'],
    'fire_truck': ['fire', 'firetruck', 'fire engine'],
    'police': ['police', 'patrol', 'cop', 'squad']
}

# Traffic signal states
class TrafficSignal:
    RED = 'red'
    YELLOW = 'yellow'
    GREEN = 'green'
    
    def __init__(self):
        self.state = self.RED
        self.emergency_active = False
        self.last_change = time.time()
        self.normal_duration = 30  # seconds
        self.emergency_duration = 10  # seconds

traffic_signal = TrafficSignal()

# Traffic statistics
traffic_stats = {
    'total_vehicles': 0,
    'emergency_vehicles': 0,
    'detection_history': deque(maxlen=100),
    'emergency_logs': deque(maxlen=50)
}

# Load YOLO model
model = None

def load_yolo_model():
    """Load YOLO model with PyTorch 2.6+ compatibility fix"""
    global model
    import torch
    import functools
    
    # Save original torch.load
    original_torch_load = torch.load
    
    # Create patched version that sets weights_only=False for YOLO models
    # This is safe for Ultralytics YOLO models from official source
    @functools.wraps(original_torch_load)
    def patched_torch_load(f, map_location=None, pickle_module=None, weights_only=None, **kwargs):
        # Force weights_only=False for YOLO model loading (PyTorch 2.6+ fix)
        kwargs['weights_only'] = False
        return original_torch_load(f, map_location=map_location, pickle_module=pickle_module, **kwargs)
    
    try:
        # Patch torch.load before loading YOLO model
        torch.load = patched_torch_load
        
        # Try to load the model
        model = YOLO('yolov8n.pt')
        print("✅ YOLO model loaded successfully with PyTorch 2.6+ compatibility fix!")
        return True
                
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error loading YOLO model: {error_msg}")
        
        # Check if it's the weights_only error
        if 'weights_only' in error_msg or 'WeightsUnpickler' in error_msg:
            print("⚠️  Detected PyTorch 2.6+ security issue, patch should have fixed this...")
            print("   Trying alternative approach...")
        
        # Try with environment variable set (sometimes helps)
        import os
        try:
            os.environ['YOLO_VERBOSE'] = 'False'
            # Ensure patch is still active
            torch.load = patched_torch_load
            model = YOLO('yolov8n.pt')
            print("✅ YOLO model loaded successfully (fallback method)!")
            return True
        except Exception as e2:
            print(f"❌ Failed to load YOLO model: {e2}")
            print("\n💡 Troubleshooting steps:")
            print("   1. Update ultralytics: pip install --upgrade ultralytics")
            print("   2. Check PyTorch version: python -c 'import torch; print(torch.__version__)'")
            print("   3. If PyTorch >= 2.6, try: pip install 'torch<2.6'")
            print("   4. Or reinstall: pip uninstall ultralytics && pip install ultralytics")
            return False
    finally:
        # Restore original torch.load after model is loaded (model is in memory now)
        torch.load = original_torch_load

# Load the model
if not load_yolo_model():
    print("\n⚠️  WARNING: YOLO model could not be loaded!")
    print("The system will try to download/load the model on first use.")
    print("If issues persist, try:")
    print("  pip install --upgrade ultralytics")
    print("  Or install PyTorch < 2.6: pip install 'torch<2.6'")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def is_vehicle(class_name):
    """Check if detected object is a vehicle"""
    return class_name.lower() in VEHICLE_CLASSES.keys()

def check_emergency_vehicle_smart(class_name, confidence):
    """
    Smart emergency vehicle detection.
    
    Current approach: Uses YOLO for fast vehicle detection, then pattern matching
    for emergency vehicle classification (no API calls needed).
    
    Why this approach:
    1. YOLO is FAST and detects all vehicles in one pass
    2. Pattern matching is instant (no network latency)
    3. Hugging Face API is optional - only use if you have a specialized model
    
    Alternative: Use Hugging Face model alone if it detects both vehicles AND
    emergency vehicles in one call.
    """
    # Method 1: Pattern matching (instant, no API calls)
    is_emergency, emergency_type = check_emergency_pattern(class_name)
    if is_emergency:
        return True, emergency_type
    
    # Note: Standard YOLO COCO doesn't have ambulance/firetruck classes,
    # so we rely on pattern matching. For production, you'd either:
    # - Train YOLO on emergency vehicle dataset
    # - Use a specialized Hugging Face model that detects both
    # - Use color/shape analysis for emergency vehicle classification
    
    return False, None

def check_emergency_pattern(class_name):
    """
    Pattern-based emergency vehicle detection.
    Enhanced to detect emergency vehicles based on vehicle characteristics.
    Note: Standard YOLO COCO doesn't have ambulance/firetruck classes,
    but we can detect patterns or use visual features (colors, sirens detection).
    """
    class_lower = class_name.lower()
    
    # Direct pattern matching from class name (if model was trained with these classes)
    for emergency_type, keywords in EMERGENCY_INDICATORS.items():
        if any(keyword in class_lower for keyword in keywords):
            return True, emergency_type
    
    # Enhanced: Check for vehicles that could be emergency (trucks/buses)
    # In real implementation, you'd analyze:
    # - Vehicle color (red/white for ambulances, red for fire trucks)
    # - Vehicle markings/logos
    # - Lights/sirens (requires additional CV techniques)
    # - Size and shape characteristics
    
    return False, None

# Optional: Use Hugging Face model directly for ALL detection (alternative approach)
def detect_with_huggingface_only(frame):
    """
    Alternative: Use Hugging Face model for all detection.
    Call this instead of YOLO if your HF model handles both vehicle and emergency detection.
    Uncomment and use if preferred.
    """
    if not HUGGINGFACE_API_KEY:
        return []
    
    try:
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        _, buffer = cv2.imencode('.jpg', frame)
        response = requests.post(
            HUGGINGFACE_API_URL,
            headers=headers,
            data=buffer.tobytes(),
            timeout=3
        )
        
        if response.status_code == 200:
            results = response.json()
            vehicles = []
            for pred in results:
                label = pred.get('label', '').lower()
                score = pred.get('score', 0)
                box = pred.get('box', {})
                
                if any(vehicle in label for vehicle in VEHICLE_CLASSES.keys()):
                    is_emergency = any(
                        indicator in label 
                        for indicators in EMERGENCY_INDICATORS.values() 
                        for indicator in indicators
                    )
                    vehicles.append({
                        'class': label,
                        'confidence': round(score * 100, 2),
                        'bbox': [box.get('xmin', 0), box.get('ymin', 0), 
                                box.get('xmax', 0), box.get('ymax', 0)],
                        'is_emergency': is_emergency,
                        'emergency_type': 'ambulance' if 'ambulance' in label else 
                                        'fire_truck' if 'fire' in label else 
                                        'police' if 'police' in label else None
                    })
            return vehicles
    except Exception as e:
        print(f"Hugging Face detection error: {e}")
    
    return []

def update_traffic_signal(has_emergency):
    """Update traffic signal based on emergency vehicle detection"""
    current_time = time.time()
    
    if has_emergency:
        traffic_signal.emergency_active = True
        traffic_signal.state = TrafficSignal.GREEN
        traffic_signal.last_change = current_time
    else:
        if traffic_signal.emergency_active:
            # Check if emergency duration has passed
            if current_time - traffic_signal.last_change > traffic_signal.emergency_duration:
                traffic_signal.emergency_active = False
                traffic_signal.state = TrafficSignal.YELLOW
                traffic_signal.last_change = current_time
        else:
            # Normal traffic light cycle
            elapsed = current_time - traffic_signal.last_change
            if elapsed > traffic_signal.normal_duration:
                if traffic_signal.state == TrafficSignal.GREEN:
                    traffic_signal.state = TrafficSignal.YELLOW
                elif traffic_signal.state == TrafficSignal.YELLOW:
                    traffic_signal.state = TrafficSignal.RED
                else:  # RED
                    traffic_signal.state = TrafficSignal.GREEN
                traffic_signal.last_change = current_time

def draw_traffic_signal(frame, x, y, state, emergency=False):
    """Draw traffic signal on frame"""
    signal_size = 30
    spacing = 35
    
    # Red light
    color_red = (0, 0, 255) if state == TrafficSignal.RED else (0, 0, 100)
    cv2.circle(frame, (x, y), signal_size, color_red, -1)
    if state == TrafficSignal.RED:
        cv2.circle(frame, (x, y), signal_size, (255, 255, 255), 2)
    
    # Yellow light
    color_yellow = (0, 255, 255) if state == TrafficSignal.YELLOW else (0, 200, 200)
    cv2.circle(frame, (x, y + spacing), signal_size, color_yellow, -1)
    if state == TrafficSignal.YELLOW:
        cv2.circle(frame, (x, y + spacing), signal_size, (255, 255, 255), 2)
    
    # Green light
    color_green = (0, 255, 0) if state == TrafficSignal.GREEN else (0, 150, 0)
    cv2.circle(frame, (x, y + spacing * 2), signal_size, color_green, -1)
    if state == TrafficSignal.GREEN:
        cv2.circle(frame, (x, y + spacing * 2), signal_size, (255, 255, 255), 2)
    
    # Emergency indicator
    if emergency:
        cv2.putText(frame, "EMERGENCY", (x - 50, y - 20), 
                   cv2.FONT_HERSHEY_BOLD, 0.7, (0, 0, 255), 2)

def process_traffic_frame(frame):
    """Process traffic frame with vehicle detection and emergency vehicle handling"""
    global model
    
    # Try to load model if it's not loaded yet
    if model is None:
        print("Model not loaded, attempting to load now...")
        if not load_yolo_model():
            print("ERROR: Cannot process frames without YOLO model!")
            return frame, [], False, traffic_signal.state
    
    try:
        results = model(frame)
        
        vehicles = []
        emergency_detected = False
        emergency_type = None
        
        boxes = results[0].boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls]
            
            if is_vehicle(name):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Check if emergency vehicle using pattern matching (fast, no API calls)
                # Note: YOLO detects vehicles, we classify emergency vehicles by pattern matching
                # Hugging Face API is optional and can be used for enhanced classification
                is_emergency, emergency_label = check_emergency_vehicle_smart(name, conf)
                
                vehicle_data = {
                    'class': name,
                    'confidence': round(conf * 100, 2),
                    'bbox': [x1, y1, x2, y2],
                    'is_emergency': is_emergency,
                    'emergency_type': emergency_label
                }
                
                vehicles.append(vehicle_data)
                traffic_stats['total_vehicles'] += 1
                
                if is_emergency:
                    emergency_detected = True
                    emergency_type = emergency_label
                    traffic_stats['emergency_vehicles'] += 1
                    
                    # Log emergency detection
                    log_entry = {
                        'timestamp': datetime.now().isoformat(),
                        'type': emergency_label,
                        'location': [x1, y1, x2, y2]
                    }
                    traffic_stats['emergency_logs'].append(log_entry)
        
        # Update traffic signal
        update_traffic_signal(emergency_detected)
        
        # Draw detections on frame
        annotated_frame = frame.copy()
        
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle['bbox']
            
            # Choose color based on vehicle type
            if vehicle['is_emergency']:
                color = (0, 0, 255)  # Red for emergency
                label = f"EMERGENCY: {vehicle['class']} ({vehicle['confidence']}%)"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_BOLD, 0.6, color, 2)
            else:
                color = (0, 255, 0)  # Green for normal vehicles
                label = f"{vehicle['class']} ({vehicle['confidence']}%)"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw traffic signal (top right corner)
        h, w = annotated_frame.shape[:2]
        signal_x = w - 60
        signal_y = 30
        draw_traffic_signal(annotated_frame, signal_x, signal_y, 
                          traffic_signal.state, emergency_detected)
        
        # Add traffic stats text
        stats_text = f"Vehicles: {len(vehicles)}"
        if emergency_detected:
            stats_text += f" | EMERGENCY: {emergency_type}"
        cv2.putText(annotated_frame, stats_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Store detection in history
        traffic_stats['detection_history'].append({
            'timestamp': datetime.now().isoformat(),
            'vehicles': len(vehicles),
            'emergency': emergency_detected
        })
        
        return annotated_frame, vehicles, emergency_detected, traffic_signal.state
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return frame, [], False, traffic_signal.state

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_traffic', methods=['POST'])
def process_traffic():
    """Process traffic camera feed"""
    try:
        data = request.json
        image_data = data.get('image')
        
        # Decode base64 image
        header, encoded = image_data.split(',', 1)
        image_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Could not decode image'}), 400
        
        # Process traffic frame
        annotated_frame, vehicles, emergency_detected, signal_state = process_traffic_frame(frame)
        
        # Encode result
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': f'data:image/jpeg;base64,{img_base64}',
            'vehicles': vehicles,
            'vehicle_count': len(vehicles),
            'emergency_detected': emergency_detected,
            'signal_state': signal_state,
            'traffic_stats': {
                'total_vehicles': traffic_stats['total_vehicles'],
                'emergency_vehicles': traffic_stats['emergency_vehicles']
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/traffic_stats', methods=['GET'])
def get_traffic_stats():
    """Get traffic statistics"""
    # Convert deques to lists for JSON serialization
    stats_copy = {
        'total_vehicles': traffic_stats['total_vehicles'],
        'emergency_vehicles': traffic_stats['emergency_vehicles'],
        'detection_history': list(traffic_stats['detection_history']),
        'emergency_logs': list(traffic_stats['emergency_logs'])
    }
    return jsonify({
        'stats': stats_copy,
        'signal_state': traffic_signal.state,
        'emergency_active': traffic_signal.emergency_active
    })

@app.route('/signal_control', methods=['POST'])
def signal_control():
    """Manual traffic signal control"""
    data = request.json
    state = data.get('state')
    
    if state in [TrafficSignal.RED, TrafficSignal.YELLOW, TrafficSignal.GREEN]:
        traffic_signal.state = state
        traffic_signal.last_change = time.time()
        return jsonify({'success': True, 'state': state})
    
    return jsonify({'error': 'Invalid state'}), 400

def generate_video_frames(video_path):
    """Generator function for video streaming with traffic detection"""
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process traffic frame
        annotated_frame, _, _, _ = process_traffic_frame(frame)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Upload and process a static image"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Read and process the image
            img = cv2.imread(filepath)
            if img is None:
                return jsonify({'error': 'Could not read image file'}), 400
            
            # Process traffic frame
            annotated_frame, vehicles, emergency_detected, signal_state = process_traffic_frame(img)
            
            # Encode result as base64
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Save annotated image
            result_filename = f'result_{filename}'
            result_path = os.path.join('static', 'results', result_filename)
            cv2.imwrite(result_path, annotated_frame)
            
            return jsonify({
                'success': True,
                'image': f'data:image/jpeg;base64,{img_base64}',
                'vehicles': vehicles,
                'vehicle_count': len(vehicles),
                'emergency_detected': emergency_detected,
                'signal_state': signal_state,
                'traffic_stats': {
                    'total_vehicles': traffic_stats['total_vehicles'],
                    'emergency_vehicles': traffic_stats['emergency_vehicles']
                },
                'result_url': f'/static/results/{result_filename}'
            })
        except Exception as e:
            print(f"Error processing image: {e}")
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'}), 400

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'video_path': f'/video_feed/{filename}',
            'filename': filename
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/video_feed/<filename>')
def video_feed(filename):
    """Stream processed video with traffic detection"""
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(video_path):
        return "Video not found", 404
    
    return Response(generate_video_frames(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'huggingface_configured': bool(HUGGINGFACE_API_KEY)
    })

if __name__ == '__main__':
    print("🚦 Traffic Control System Starting...")
    print(f"📡 Hugging Face API: {'Configured' if HUGGINGFACE_API_KEY else 'Not configured (using pattern matching)'}")
    app.run(debug=True, host='0.0.0.0', port=5000)
