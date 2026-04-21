import cv2
import time
import sys
import os
import base64
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS

# Make sure final_model.py can be imported from the same directory
sys.path.insert(0, os.path.dirname(__file__))
from final_model import PhoneFaceDetector

app = Flask(__name__)
CORS(app) # Enable CORS for cloud deployment

# ── Initialize detector once at server boot ────────────────────────────────
# Use 'n' (nano) for fastest performance on free cloud tiers
detector = PhoneFaceDetector(model_size='n')
detector.pomodoro_active = True

DISPLAY_SIZE = (640, 480)

# ── Routes ─────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the Shadow Teacher website."""
    return render_template('index.html')


@app.route('/api/detect', methods=['POST'])
def api_detect():
    """
    Receives a base64 frame from the browser, runs YOLO, and returns 
    distraction status + detection metadata.
    """
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"status": "error", "message": "No image data"}), 400

        # Decode base64 image
        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"status": "error", "message": "Invalid image"}), 400

        # Optional: Resize for speed
        frame = cv2.resize(frame, DISPLAY_SIZE)

        # Run YOLO26
        # Using imgsz=416 for a good balance of speed and accuracy on cloud CPUs
        results = detector.model(frame, verbose=False, conf=0.08, iou=0.45, imgsz=416)
        
        # This call updates detector's internal counters (distraction_count, etc.)
        annotated, interactions = detector.draw_detections(frame, results)
        
        # We don't draw alerts on the frame here because the frontend handles banners
        # but we return the interactions so the frontend can draw boxes if it wants.
        
        return jsonify({
            "status": "success",
            "data": {
                "is_distracted": bool(detector.show_focus_alert or detector.show_absence_alert),
                "distraction_type": "phone" if detector.show_focus_alert else ("absence" if detector.show_absence_alert else None),
                "stats": {
                    "distraction_count":    detector.distraction_count,
                    "absence_count":        detector.absence_count,
                    "total_distraction_time": int(detector.total_paused_time + detector.total_absence_time),
                    "phone_held":           detector.stats['phone_held'],
                    "phone_near_face":      detector.stats['phone_near_face'],
                    "total_interactions":   detector.stats['interactions_detected']
                }
            }
        })
    except Exception as e:
        print(f"Detection error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/stats')
def api_stats():
    """JSON snapshot of current distraction state."""
    return jsonify({
        "status": "success",
        "data": {
            "is_distracted":        detector.show_focus_alert or detector.show_absence_alert,
            "distraction_count":    detector.distraction_count,
            "absence_count":        detector.absence_count,
            "current_phone_time":   int(detector.continuous_phone_time),
            "current_absence_time": int(detector.continuous_absence_time),
            "total_distraction_time": int(detector.total_paused_time + detector.total_absence_time),
            "total_phone_time":     int(detector.total_paused_time),
            "total_absence_time":   int(detector.total_absence_time),
            "total_interactions":   detector.stats['interactions_detected'],
            "phone_held":           detector.stats['phone_held'],
            "phone_near_face":      detector.stats['phone_near_face']
        }
    })


@app.route('/api/reset', methods=['POST'])
def api_reset():
    """Reset session statistics."""
    global detector
    detector = PhoneFaceDetector(model_size='n')
    detector.pomodoro_active = True
    return jsonify({"status": "success"})


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("\n🌿 Shadow Teacher — Cloud AI Backend")
    print("=" * 40)
    print(f"🌐 Running on port: {port}")
    print("=" * 40 + "\n")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
