"""
app.py — Flask API connecting Shadow Teacher website with the YOLO26 detection model.
"""

import cv2
import time
import sys
import os
import threading

from flask import Flask, Response, jsonify, render_template

# Make sure final_model.py can be imported from the same directory
sys.path.insert(0, os.path.dirname(__file__))
from final_model import PhoneFaceDetector

app = Flask(__name__)

# ── Constants ─────────────────────────────────────────────────────────────
DISPLAY_SIZE = (640, 480)
SKIP_FRAMES  = 2          # Run YOLO every N frames (speeds up stream)

# ── Initialize detector once at server boot ────────────────────────────────
detector = PhoneFaceDetector(model_size='n')
detector.pomodoro_active = True   # Always monitoring


# ── Threaded camera — always keeps the freshest frame ready instantly ──────
class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DISPLAY_SIZE[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_SIZE[1])
        self.ret, self.frame = self.cap.read()
        self.lock    = threading.Lock()
        self.running = True
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.ret else (False, None)

    def stop(self):
        self.running = False
        self.cap.release()


# Start the camera in the background immediately so the first frame is instant
camera = CameraStream()

_skip_counter  = 0
_last_annotated = None


# ── Video stream generator ─────────────────────────────────────────────────
def generate_frames():
    """Reads from threaded camera, runs YOLO every few frames, streams JPEG."""
    global _skip_counter, _last_annotated

    while True:
        success, frame = camera.read()
        if not success or frame is None:
            time.sleep(0.03)
            continue

        _skip_counter += 1
        if _skip_counter >= SKIP_FRAMES:
            _skip_counter = 0
            # Run YOLO26 with high accuracy
            results  = detector.model(frame, verbose=False, conf=0.08, iou=0.45, imgsz=640)
            annotated, _ = detector.draw_detections(frame, results)
            detector._draw_distractions(annotated, DISPLAY_SIZE)
            _last_annotated = annotated
        else:
            annotated = _last_annotated if _last_annotated is not None else frame

        ret, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the Shadow Teacher website."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """
    Live MJPEG stream with YOLO bounding boxes.
    Use as: <img src="/video_feed">
    """
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/stats')
def api_stats():
    """JSON snapshot of current distraction state — poll every second from JS."""
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


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n🌿 Shadow Teacher — AI Focus Monitor")
    print("=" * 40)
    print("🌐 Open your browser at: http://localhost:5000")
    print("=" * 40 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
