# Extracted from final model notebook (project_modelyolo26_deepsort_fast.ipynb)

"""
Real-time YOLO26 Phone-Face Interaction Detection with Pomodoro Timer
- DeepSORT tracking for persistent person/phone IDs
- Temporal phone buffer: phone confirmed over N frames to reduce false negatives
- Linger frames: detected phone stays "alive" for a few frames after disappearing
Optimized for Jupyter Notebook with cross-platform sound alarms.
"""

import cv2
import numpy as np
from ultralytics import YOLO
try:
    from IPython.display import display, clear_output, Image
except ImportError:
    display = clear_output = Image = None  # Not running in Jupyter — that's fine
from datetime import datetime, timedelta
import math
import time
import sys
import threading
from PIL import Image as PILImage
import io
from collections import defaultdict, deque
# MediaPipe removed (feature deleted)


# ═══════════════════════════════════════════════════════
#  ⚡ SPEED CONFIG  (tweak these without touching the model)
# ═══════════════════════════════════════════════════════
IMGSZ        = 416   # YOLO inference size: 320 (fastest) | 416 | 640 (best accuracy)
SKIP_FRAMES  = 2     # run DeepSORT only every N frames (1 = every frame, 2 = every other, etc.)
JPEG_QUALITY = 70    # Jupyter JPEG quality (50-85); lower = faster display
# ═══════════════════════════════════════════════════════


# DeepSORT
from deep_sort_realtime.deepsort_tracker import DeepSort


# ═══════════════════════════════════════════════════════
#  SOUND ALARM MODULE  (cross-platform)
# ═══════════════════════════════════════════════════════

def _beep_windows(frequency=1000, duration_ms=300):
    try:
        import winsound
        winsound.Beep(frequency, duration_ms)
    except Exception:
        pass


def _beep_macos(frequency=1000, duration_ms=300):
    import subprocess
    try:
        subprocess.run(
            ['sox', '-n', '-d', 'synth', str(duration_ms / 1000), 'sine', str(frequency)],
            capture_output=True, timeout=2)
    except Exception:
        try:
            subprocess.run(['osascript', '-e', 'beep'], capture_output=True, timeout=1)
        except Exception:
            pass


def _beep_linux(frequency=1000, duration_ms=300):
    import subprocess
    try:
        subprocess.run(
            ['beep', '-f', str(frequency), '-l', str(duration_ms)],
            capture_output=True, timeout=2)
    except Exception:
        try:
            sys.stdout.write('\a')
            sys.stdout.flush()
        except Exception:
            pass


# Alarm patterns: (frequency_hz, duration_ms)
ALARM_PATTERNS = {
    'phone':    [(880, 200), (1100, 200), (1320, 350)],   # rising urgent
    'absence':  [(440, 300), (370, 500)],                  # falling warning
    'complete': [(523, 180), (659, 180), (784, 360)],      # C-E-G celebration
    'break':    [(659, 140), (523, 280)],                   # E-C end-of-break
}


def play_alarm(alarm_type='phone', blocking=False):
    """
    Play a sound alarm (non-blocking by default, runs in daemon thread).

    alarm_type options:
      'phone'    - rising 3-tone: phone detected too long
      'absence'  - falling 2-tone: person absent too long
      'complete' - C-E-G ascending: work session done
      'break'    - E-C descending: break session ended
    """
    pattern = ALARM_PATTERNS.get(alarm_type, ALARM_PATTERNS['phone'])

    def _play():
        for freq, dur in pattern:
            if sys.platform.startswith('win'):
                _beep_windows(freq, dur)
            elif sys.platform.startswith('darwin'):
                _beep_macos(freq, dur)
            else:
                _beep_linux(freq, dur)
            time.sleep(0.06)

    if blocking:
        _play()
    else:
        threading.Thread(target=_play, daemon=True).start()


# ═══════════════════════════════════════════════════════
#  DEEPSORT TRACKER WRAPPER
# ═══════════════════════════════════════════════════════

class PhoneTracker:
    """
    Wraps DeepSORT for persons and phones with two extra features:
      1. Confidence buffer  — phone box must appear in CONFIRM_FRAMES consecutive
                              frames before it is considered a real detection.
      2. Linger frames      — once confirmed, a phone track stays "active" for
                              LINGER_FRAMES after it was last seen (handles brief
                              occlusion, hand movement, YOLO miss-frame).
    """

    # ── Tunable parameters ────────────────────────────────────────────
    CONFIRM_FRAMES = 2     # frames a new phone track must survive before being used
    LINGER_FRAMES  = 5     # frames a phone track is kept "alive" after last detection
    # ─────────────────────────────────────────────────────────────────

    def __init__(self):
        # Separate DeepSORT instances for persons and phones
        self.person_tracker = DeepSort(
            max_age=30,
            n_init=2,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3,
            nn_budget=None,
            embedder="mobilenet",
            half=False,
            bgr=True,
        )
        self.phone_tracker = DeepSort(
            max_age=self.LINGER_FRAMES,   # max_age acts as linger for phones
            n_init=self.CONFIRM_FRAMES,   # n_init acts as confirm threshold
            nms_max_overlap=1.0,
            max_cosine_distance=0.4,
            nn_budget=None,
            embedder="mobilenet",
            half=False,
            bgr=True,
        )

        # track_id → last confirmed xyxy (for linger display)
        self._phone_last_box: dict[int, np.ndarray] = {}
        # track_id → frames since last seen
        self._phone_linger:   dict[int, int]        = {}

    def _yolo_to_deepsort(self, boxes_conf, frame):
        """Convert [(xyxy, conf), ...] → DeepSORT detection list."""
        detections = []
        h, w = frame.shape[:2]
        for xyxy, conf in boxes_conf:
            x1, y1, x2, y2 = xyxy
            # DeepSORT wants [left, top, w, h]
            bw, bh = float(x2 - x1), float(y2 - y1)
            det = ([float(x1), float(y1), bw, bh], float(conf), 'object')
            detections.append(det)
        return detections

    def update(self, person_boxes_conf, phone_boxes_conf, frame):
        """
        Update both trackers.

        Returns:
            person_tracks : list of (track_id, xyxy, conf)  — confirmed persons
            phone_tracks  : list of (track_id, xyxy, conf)  — confirmed + lingering phones
        """
        # ── Persons ──────────────────────────────────────────────────
        p_dets   = self._yolo_to_deepsort(person_boxes_conf, frame)
        p_tracks = self.person_tracker.update_tracks(p_dets, frame=frame)

        person_tracks = []
        for t in p_tracks:
            if not t.is_confirmed():
                continue
            ltwh = t.to_ltwh()
            x1 = ltwh[0]; y1 = ltwh[1]
            x2 = x1 + ltwh[2]; y2 = y1 + ltwh[3]
            conf = t.det_conf if t.det_conf is not None else 0.5
            person_tracks.append((t.track_id, np.array([x1, y1, x2, y2]), conf))

        # ── Phones ───────────────────────────────────────────────────
        ph_dets   = self._yolo_to_deepsort(phone_boxes_conf, frame)
        ph_tracks = self.phone_tracker.update_tracks(ph_dets, frame=frame)

        # Update linger counters
        active_ids = set()
        phone_tracks_raw = []
        for t in ph_tracks:
            if not t.is_confirmed():
                continue
            ltwh = t.to_ltwh()
            x1 = ltwh[0]; y1 = ltwh[1]
            x2 = x1 + ltwh[2]; y2 = y1 + ltwh[3]
            box = np.array([x1, y1, x2, y2])
            conf = t.det_conf if t.det_conf is not None else 0.5
            active_ids.add(t.track_id)
            self._phone_last_box[t.track_id] = box
            self._phone_linger[t.track_id]   = 0
            phone_tracks_raw.append((t.track_id, box, conf))

        # Age non-active tracks; keep them if within linger window
        linger_tracks = []
        to_delete = []
        for tid, age in list(self._phone_linger.items()):
            if tid in active_ids:
                continue
            age += 1
            if age <= self.LINGER_FRAMES:
                self._phone_linger[tid] = age
                linger_tracks.append((tid, self._phone_last_box[tid], 0.15))
            else:
                to_delete.append(tid)
        for tid in to_delete:
            self._phone_linger.pop(tid, None)
            self._phone_last_box.pop(tid, None)

        phone_tracks = phone_tracks_raw + linger_tracks
        return person_tracks, phone_tracks


# ═══════════════════════════════════════════════════════
#  MAIN DETECTOR CLASS
# ═══════════════════════════════════════════════════════

class PhoneFaceDetector:
    def __init__(self, model_size='n'):
        """
        Initialize with YOLO26.
        model_size: 'n' (nano-fastest), 's', 'm', 'l', 'x'

        YOLO26 improvements over YOLO11:
          - End-to-end NMS-free inference (faster, simpler deployment)
          - DFL removed (broader hardware support)
          - STAL: Small-Target-Aware Label Assignment (better phone detection)
          - ProgLoss: Progressive loss balancing (more stable training)
          - MuSGD optimizer
          - ~43% faster CPU inference for nano model
        """
        print(f'Loading YOLO26{model_size} model...')
        self.model = YOLO(f'yolo26{model_size}.pt')   # <-- YOLO26 model string
        print(f'✅ YOLO26{model_size} loaded!')

        self.mp_face_mesh = None


        # COCO class IDs (unchanged from YOLO11)
        self.PERSON_CLASS     = 0
        self.CELL_PHONE_CLASS = 67

        # Interaction parameters
        self.PROXIMITY_THRESHOLD = 200

        # ── Confidence thresholds ────────────────────────────────────
        # YOLO26 STAL improves phone recall; we lower the threshold further
        # and rely on the temporal buffer in PhoneTracker to suppress noise.
        self.PERSON_CONFIDENCE = 0.25
        self.PHONE_CONFIDENCE  = 0.08   # 🔥 Max sensitivity for interaction detection

        # DeepSORT tracker
        self.tracker = PhoneTracker()

        # Rolling vote buffer — phone interaction must appear in ≥ VOTE_MIN of the
        # last VOTE_WINDOW frames before a distraction is recorded.  This kills
        # single-frame false positives without adding perceptible latency.
        self.VOTE_WINDOW   = 4
        self.VOTE_MIN      = 2
        self._interaction_votes: deque = deque(maxlen=self.VOTE_WINDOW)

        # Statistics
        self.stats = {
            'interactions_detected': 0,
            'total_distractions': 0,
            'total_frames': 0,
            'phone_held': 0,
            'phone_near_face': 0
        }


        # Pomodoro state
        self.pomodoro_active     = True
        self.pomodoro_duration   = 0
        self.pomodoro_start_time = None
        self.pomodoro_end_time   = None
        self.break_duration      = 0
        self.session_count       = 0
        self.distraction_count   = 0
        self.distraction_time    = 0
        self.last_distraction    = None
        self.on_break            = False
        self.break_start_time    = None
        self.break_end_time      = None

        # Phone tracking
        self.phone_usage_start     = None
        self.continuous_phone_time = 0
        self.phone_usage_threshold = 30
        self.show_focus_alert      = False
        self.alert_display_time    = None
        self.alert_duration        = 5

        # Timer pause (phone)
        self.timer_paused      = False
        self.pause_start_time  = None
        self.total_paused_time = 0
        self.frozen_end_time   = None

        # Absence tracking
        self.person_absent_start     = None
        self.continuous_absence_time = 0
        self.absence_threshold       = 120
        self.show_absence_alert      = False
        self.absence_alert_time      = None
        self.timer_paused_absence    = False
        self.absence_pause_start     = None
        self.total_absence_time      = 0
        self.absence_count           = 0

        self._last_phone_alarm   = 0.0
        self._last_absence_alarm = 0.0
        self.ALARM_COOLDOWN      = 2


    # ── Pomodoro helpers ─────────────────────

    def start_pomodoro_session(self, duration_minutes, break_minutes):
        self.pomodoro_duration   = duration_minutes
        self.break_duration      = break_minutes
        self.pomodoro_start_time = datetime.now()
        self.pomodoro_end_time   = self.pomodoro_start_time + timedelta(minutes=duration_minutes)
        self.pomodoro_active     = True
        self.on_break            = False
        self.session_count      += 1
        self.distraction_count   = 0
        self.distraction_time    = 0
        self.timer_paused        = False
        self.total_paused_time   = 0
        print(f'\n🍅 Session {self.session_count} started! Work {duration_minutes}min / Break {break_minutes}min')
        print(f'Ends at: {self.pomodoro_end_time.strftime("%H:%M:%S")}')
        print('⏸️  Timer pauses if phone > 30s or absent > 2min\n')
    def start_break(self):
        self.on_break         = True
        self.pomodoro_active  = False
        self.break_start_time = datetime.now()
        self.break_end_time   = self.break_start_time + timedelta(minutes=self.break_duration)
        print(f'\n💤 Break! {self.break_duration} min — ends {self.break_end_time.strftime("%H:%M:%S")}\n')
        play_alarm('complete')   # 🔔 celebratory beep

    def check_pomodoro_status(self):
        if self.on_break:
            if datetime.now() >= self.break_end_time:
                self.on_break = False
                return 'BREAK_COMPLETED'
            return 'ON_BREAK'
        if not self.pomodoro_active:
            return 'INACTIVE'
        if self.timer_paused or self.timer_paused_absence:

            return 'ACTIVE'
        if datetime.now() >= self.pomodoro_end_time:
            self.pomodoro_active = False
            return 'COMPLETED'
        return 'ACTIVE'




    def get_remaining_time(self):
        if self.on_break:
            return max(0, (self.break_end_time - datetime.now()).total_seconds())
        if self.pomodoro_active:
            if (self.timer_paused or self.timer_paused_absence) and self.frozen_end_time:

                return max(0, (self.frozen_end_time - datetime.now()).total_seconds())
            return max(0, (self.pomodoro_end_time - datetime.now()).total_seconds())
        return 0

    def format_time(self, seconds):
        return f'{int(seconds // 60):02d}:{int(seconds % 60):02d}'

    # ── Absence tracking ─────────────────────

    def check_person_presence(self, persons_detected):
        if not self.pomodoro_active:
            return
        now = datetime.now()

        if not persons_detected:
            if self.person_absent_start is None:
                self.person_absent_start     = now
                self.continuous_absence_time = 0
            else:
                self.continuous_absence_time = (now - self.person_absent_start).total_seconds()
                if self.continuous_absence_time >= self.absence_threshold:
                    if not self.timer_paused_absence:
                        self.timer_paused_absence = True
                        self.absence_pause_start  = now
                        self.absence_count       += 1
                        print(f'\n⚠️  ALERT — absent {self.continuous_absence_time:.0f}s')
                    self.show_absence_alert = True
                    self.absence_alert_time = now

                    # 🔔 Sound alarm with cooldown
                    ts = time.time()
                    if ts - self._last_absence_alarm >= self.ALARM_COOLDOWN:
                        play_alarm('absence')
                        self._last_absence_alarm = ts
            self.stats['total_distractions'] += 1

        else:
            if self.timer_paused_absence:
                dur = (now - self.absence_pause_start).total_seconds()
                self.total_absence_time  += dur
                self.timer_paused_absence = False
                self.absence_pause_start  = None
                self.show_absence_alert  = False
                print(f'\n▶️  RESUMED — back after {dur:.0f}s')
            self.person_absent_start     = None
            self.continuous_absence_time = 0

    # ── Phone distraction tracking ───────────

    def record_distraction(self, interaction_type):
        if not self.pomodoro_active:
            return
        now = datetime.now()

        if interaction_type != 'NO_INTERACTION':
            if self.phone_usage_start is None:
                self.phone_usage_start     = now
                self.continuous_phone_time = 0
            else:
                self.continuous_phone_time = (now - self.phone_usage_start).total_seconds()
                if self.continuous_phone_time >= self.phone_usage_threshold:
                    if not self.timer_paused:
                        self.timer_paused     = True
                        self.pause_start_time = now
                        print(f'\n⚠️  ALERT — phone {self.phone_usage_threshold}s exceeded!')
                    if (self.last_distraction is None or
                            (now - self.last_distraction).total_seconds() > 10):
                        self.distraction_count += 1
                        self.last_distraction   = now
                        print(f'⚠️  Distraction #{self.distraction_count} — phone {self.continuous_phone_time:.0f}s')
                    self.show_focus_alert   = True
                    self.alert_display_time = now

                    # 🔔 Sound alarm with cooldown
                    ts = time.time()
                    if ts - self._last_phone_alarm >= self.ALARM_COOLDOWN:
                        play_alarm('phone')
                        self._last_phone_alarm = ts

            self.distraction_time += 1
        else:
            if self.timer_paused:
                dur = (now - self.pause_start_time).total_seconds()
                self.total_paused_time += dur
                self.timer_paused       = False
                self.pause_start_time   = None
                self.show_focus_alert   = False
                print(f'\n▶️  RESUMED — phone down (tracked {dur:.0f}s)')
            self.phone_usage_start     = None
            self.continuous_phone_time = 0

    # ── Geometry helpers ─────────────────────

    def calculate_distance(self, box1, box2):
        c1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
        c2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
        return math.hypot(c1[0] - c2[0], c1[1] - c2[1])

    def get_box_center(self, box):
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    def detect_face_in_person(self, person_box):
        x1, y1, x2, y2 = person_box
        return [x1, y1, x2, y1 + (y2 - y1) * 0.3]

    def _iou(self, box_a, box_b):
        """Intersection-over-Union of two xyxy boxes."""
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b
        ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
        ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(1.0, (xa2 - xa1) * (ya2 - ya1))
        area_b = max(1.0, (xb2 - xb1) * (yb2 - yb1))
        return inter / (area_a + area_b - inter)

    def analyze_interaction(self, person_box, phone_box):
        face_box     = self.detect_face_in_person(person_box)
        distance     = self.calculate_distance(face_box, phone_box)
        phone_center = self.get_box_center(phone_box)

        # Strong signal: IoU overlap between phone box and person box
        overlap = self._iou(person_box, phone_box)

        # Phone center strictly inside person bounding box
        person_contains_phone = (
            person_box[0] <= phone_center[0] <= person_box[2] and
            person_box[1] <= phone_center[1] <= person_box[3]
        )

        # Phone is in the lower 2/3 of the person (body) but not near face
        # Use wider margins (80px horizontal, 80px below) to catch phones held at lap/side
        person_body_box = list(person_box)
        person_body_box[1] = person_box[1] + (person_box[3] - person_box[1]) * 0.20
        phone_near_body = (
            person_body_box[0] - 80 <= phone_center[0] <= person_body_box[2] + 80 and
            person_body_box[1]       <= phone_center[1] <= person_body_box[3] + 80
        )

        if distance < self.PROXIMITY_THRESHOLD * 0.5:
            return 'PHONE_AT_EAR', min(1.0, (self.PROXIMITY_THRESHOLD * 0.5 - distance)
                                       / (self.PROXIMITY_THRESHOLD * 0.5)), distance
        if distance < self.PROXIMITY_THRESHOLD:
            return 'PHONE_NEAR_FACE', min(1.0, (self.PROXIMITY_THRESHOLD - distance)
                                         / self.PROXIMITY_THRESHOLD), distance
        # IoU-based overlap is a reliable held-phone signal
        if overlap > 0.03 or person_contains_phone:
            return 'PHONE_HELD', max(0.6, overlap * 2.0), distance
        # Phone beside/under body — looser held check (catches phone at lap)
        if phone_near_body:
            return 'PHONE_HELD', 0.55, distance
        return 'NO_INTERACTION', 0.0, distance

    # ── Drawing (with DeepSORT track IDs) ───

    def draw_detections(self, frame, results):
        """
        Draw bounding boxes and run DeepSORT tracking.
        Phone tracks show their persistent track ID to make it easy to see
        when the same phone is re-detected vs a new one appearing.
        """
        annotated = frame  # ⚡ work directly on frame; copy deferred to alert overlay
        boxes = results[0].boxes

        # ⚡ Batch-extract all boxes at once (one GPU→CPU transfer)
        person_raw = []
        phone_raw  = []
        if len(boxes):
            all_cls  = boxes.cls.cpu().numpy().astype(int)
            all_conf = boxes.conf.cpu().numpy()
            all_xyxy = boxes.xyxy.cpu().numpy()
            for cls, conf, xyxy in zip(all_cls, all_conf, all_xyxy):
                if cls == self.PERSON_CLASS and conf > self.PERSON_CONFIDENCE:
                    person_raw.append((xyxy, conf))
                elif cls == self.CELL_PHONE_CLASS and conf > self.PHONE_CONFIDENCE:
                    phone_raw.append((xyxy, conf))

        # Update DeepSORT — get confirmed tracks with stable IDs
        person_tracks, phone_tracks = self.tracker.update(person_raw, phone_raw, frame)

        # Presence and posture checks use confirmed tracks
        self.check_person_presence(len(person_tracks) > 0)

        # Draw persons
        for tid, pb, conf in person_tracks:
            x1, y1, x2, y2 = map(int, pb)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f'Person#{tid} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw phones
        any_interaction    = False
        interactions_found = []

        for ph_tid, phb, ph_conf in phone_tracks:
            x1, y1, x2, y2 = map(int, phb)
            # Lighter colour when in linger window (ph_conf == 0.15 sentinel)
            ph_color = (255, 140, 0) if ph_conf <= 0.15 else (255, 0, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), ph_color, 2)
            linger_tag = ' [linger]' if ph_conf <= 0.15 else ''
            cv2.putText(annotated, f'Phone#{ph_tid}{linger_tag}',
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ph_color, 2)

            for p_tid, pb, _ in person_tracks:
                itype, iconf, dist = self.analyze_interaction(pb, phb)
                if itype != 'NO_INTERACTION':
                    any_interaction = True
                    interactions_found.append({
                        'type': itype, 'confidence': iconf, 'distance': dist,
                        'person_id': p_tid, 'phone_id': ph_tid
                    })
                    pc = self.get_box_center(pb)
                    ph = self.get_box_center(phb)
                    lc = (0, 0, 255) if self.pomodoro_active else (0, 255, 255)
                    cv2.line(annotated, (int(pc[0]), int(pc[1])), (int(ph[0]), int(ph[1])), lc, 2)
                    label = f'P#{p_tid}↔Ph#{ph_tid} {itype}'
                    cv2.putText(annotated, label,
                                (int((pc[0] + ph[0]) / 2), int((pc[1] + ph[1]) / 2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, lc, 2)
                    self.stats['interactions_detected'] += 1
                    self.stats['total_distractions']    += 1

                    if itype == 'PHONE_HELD':
                        self.stats['phone_held'] += 1
                    elif itype in ('PHONE_NEAR_FACE', 'PHONE_AT_EAR'):
                        self.stats['phone_near_face'] += 1

        # ── Rolling vote buffer — suppress single-frame false positives ───────
        self._interaction_votes.append(1 if any_interaction else 0)
        voted_interaction = sum(self._interaction_votes) >= self.VOTE_MIN
        self.record_distraction(interactions_found[0]['type'] if voted_interaction and interactions_found
                                else 'NO_INTERACTION')

        return annotated, interactions_found

    # ── Alert box helper ─────────────────────

    def _draw_alert(self, frame, title, sub, hint, color, dsize):
        aw, ah = 430, 160
        cx, cy = dsize[0] // 2, dsize[1] // 2
        x1, y1, x2, y2 = cx - aw//2, cy - ah//2, cx + aw//2, cy + ah//2
        ov = frame.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(ov, 0.70, frame, 0.30, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
        cv2.putText(frame, title, (x1 + 25, cy - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (255,255,255), 2)
        cv2.putText(frame, sub,   (x1 + 45, cy + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        cv2.putText(frame, hint,  (x1 + 55, cy + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,0),   2)

    # ── Distraction overlay ──────────────────

    def _draw_distractions(self, frame, dsize):
        if not self.pomodoro_active:
            return

        # Phone usage counter
        if self.phone_usage_start is not None:
            u  = int(self.continuous_phone_time)
            wc = (0,255,255) if (self.phone_usage_threshold - u) > 10 else (0,100,255)
            cv2.putText(frame, f'Phone: {u}s/{self.phone_usage_threshold}s',
                        (10, dsize[1]-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, wc, 2)

        # Absence counter
        if self.person_absent_start is not None:
            a  = int(self.continuous_absence_time)
            ac = (0,255,255) if (self.absence_threshold - a) > 30 else (0,100,255)
            cv2.putText(frame, f'Absent: {a}s/{self.absence_threshold}s',
                        (10, dsize[1]-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ac, 2)
        # Focus alert
        if self.show_focus_alert:
            if (datetime.now() - self.alert_display_time).total_seconds() < self.alert_duration:
                self._draw_alert(frame, '  [!] PLEASE FOCUS!',
                                 'Phone > 30s detected', '  ||  TIMER PAUSED',
                                 (0,0,210), dsize)
            else:
                self.show_focus_alert = False

        # Absence alert
        if self.show_absence_alert:
            if ((datetime.now() - self.absence_alert_time).total_seconds() < self.alert_duration
                    or self.timer_paused_absence):
                self._draw_alert(frame, '  [!] PERSON ABSENT!',
                                 'Please return to desk', '  ||  TIMER PAUSED',
                                 (0,120,220), dsize)
            else:
                self.show_absence_alert = False



    # ── Webcam loop (Jupyter) ────────────────

    def run_webcam(self, display_size=(640, 480)):
        """Run detection with Jupyter inline display."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print('❌ Could not open webcam!')
            return False
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  display_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_size[1])
        print('✅ Webcam opened. Ctrl+C to stop.')
        time.sleep(1)

        _t0 = time.time()
        fc = 0
        session_completed = False
        _skip_count = 0
        _last_annotated = None
        _last_interactions = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                # YOLO26 inference — Skip Frames for huge FPS boost
                _skip_count += 1
                if _skip_count >= 3:  # Process 1 frame, skip 2 (~3x speedup)
                    _skip_count = 0
                    results = self.model(frame, verbose=False, conf=0.10, iou=0.45, imgsz=416)
                    annotated, interactions = self.draw_detections(frame, results)
                    _last_annotated, _last_interactions = annotated.copy(), interactions
                else:
                    if _last_annotated is not None:
                        annotated = cv2.addWeighted(frame, 0.4, _last_annotated, 0.6, 0)
                        interactions = _last_interactions
                    else:
                        annotated, interactions = frame, []

                self.stats['total_frames'] = fc
                fps = fc / max(time.time() - _t0, 1e-6)
                cv2.putText(annotated, f'FPS: {fps:.1f}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                # Distractions and Alerts
                self._draw_distractions(annotated, display_size)
                if not (self.on_break or self.pomodoro_active):
                    cv2.putText(annotated, 'No Active Session', (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128,128,128), 2)

                # Track ID count overlay
                n_phones  = len(self.tracker._phone_linger) + len(
                    [t for t in self.tracker._phone_linger if self.tracker._phone_linger[t] == 0])
                cv2.putText(annotated, f'Tracked phones: {len(self.tracker._phone_last_box)}',
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,0), 2)
                cv2.putText(annotated, f'Interactions: {len(interactions)}', (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                # Jupyter display
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                buf = io.BytesIO()
                PILImage.fromarray(rgb).save(buf, format='JPEG', quality=50)
                buf.truncate(0)
                buf.seek(0)
                clear_output(wait=True)
                display(Image(data=buf.getvalue()))
                fc += 1

        except KeyboardInterrupt:
            print('\n⏸️  Stopped by user')
        finally:
            cap.release()
            self.print_statistics()
        return True

    # ── Webcam loop (native window) ──────────

    def run_webcam_native(self, display_size=(640, 480)):
        """Run detection in a native OpenCV window."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print('❌ Could not open webcam!')
            return False
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  display_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_size[1])
        print("✅ Webcam opened. Press 'q' to quit.")
        time.sleep(1)

        try:
            fc = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                results = self.model(frame, verbose=False, conf=0.10, iou=0.45, imgsz=416)
                annotated, _ = self.draw_detections(frame, results)

                # Distractions and Alerts
                self._draw_distractions(annotated, display_size)

                cv2.imshow('Pomodoro Focus — YOLO26+DeepSORT', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                fc += 1

        except KeyboardInterrupt:
            print('\n⏸️  Stopped.')
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.print_statistics()
        return True

    # ── Video file processing ────────────────

    def process_video(self, video_path, output_path='output.mp4'):
        cap = cv2.VideoCapture(video_path)
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        print(f'Processing {video_path} ({tot} frames)...')
        fc = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model(frame, verbose=False, conf=0.12, iou=0.4, imgsz=IMGSZ)
            annotated, _ = self.draw_detections(frame, results)
            out.write(annotated)
            fc += 1
            if fc % 30 == 0:
                print(f'  {fc}/{tot}')
        cap.release()
        out.release()
        print(f'Saved → {output_path}')
        self.print_statistics()

    # ── Summary / stats ──────────────────────

    def print_pomodoro_summary(self):
        if self.session_count == 0:
            return
        total_pause  = self.total_paused_time + self.total_absence_time
        actual_work  = self.pomodoro_duration - total_pause / 60
        productivity = max(0, 100 - self.distraction_count * 5 - self.absence_count * 3)
        print('\n' + '🍅'*25)
        print('POMODORO SESSION SUMMARY')
        print('🍅'*25)
        print(f'Session #{self.session_count}')
        print(f'Duration:            {self.pomodoro_duration} min')
        print(f'Paused (Phone):      {self.total_paused_time/60:.1f} min')
        print(f'Paused (Absent):     {self.total_absence_time/60:.1f} min')
        print(f'Total Pause:         {total_pause/60:.1f} min')
        print(f'Actual Work:         {actual_work:.1f} min')
        print(f'Phone Distractions:  {self.distraction_count}')
        print(f'Absences:            {self.absence_count}')
        if self.stats['total_frames'] > 0:
            pct = self.distraction_time / self.stats['total_frames'] * 100
            print(f'Distraction Time:    ~{pct:.1f}% of session')
        print(f'Productivity Score:  {productivity}/100')
        if self.distraction_count == 0 and self.absence_count == 0:
            print('🌟 Perfect focus!')
        elif self.distraction_count < 3 and self.absence_count < 2:
            print('👍 Good job! Minimal distractions.')
        elif self.distraction_count < 6 and self.absence_count < 4:
            print('⚠️  Moderate distractions. Try to improve.')
        else:
            print('❌ High distraction level. Consider a quieter environment.')
        print('🍅'*25 + '\n')

    def print_statistics(self):
        print('\n' + '='*50)
        print('DETECTION STATISTICS  (YOLO26 + DeepSORT)')
        print('='*50)
        print(f'Total Frames:       {self.stats["total_frames"]}')
        print(f'Total Interactions: {self.stats["interactions_detected"]}')
        print(f'Phone Held:         {self.stats["phone_held"]}')
        print(f'Phone Near Face:    {self.stats["phone_near_face"]}')
        if self.stats['total_frames'] > 0:
            rate = self.stats['interactions_detected'] / self.stats['total_frames'] * 100
            print(f'Interaction Rate:   {rate:.2f}%')
        print('='*50)


# ═══════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════

def test_sound():
    """Test all four alarm tones (blocking)."""
    print('🔔 Testing sound alarms...')
    for atype in ('phone', 'absence', 'complete', 'break'):
        print(f'  {atype}...')
        play_alarm(atype, blocking=True)
        time.sleep(0.5)
    print('✅ Sound test complete!')


def test_webcam():
    """Quick webcam check."""
    print('Testing webcam...')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('❌ Webcam not accessible!')
        cap.release()
        return False
    print('✅ Webcam accessible!')
    ret, frame = cap.read()
    if ret:
        print(f'✅ Frame: {frame.shape}')
    cap.release()
    return ret


# ═══════════════════════════════════════════════════════
#  INITIALIZE
# ═══════════════════════════════════════════════════════
detector = PhoneFaceDetector(model_size='n')  # change to 's' or 'm' for more accuracy


def start_focus_session(use_native_window=False):
    """Detection session launcher."""
    if not test_webcam():
        print('\n❌ Cannot start — webcam issue')
        return
    print('\n🍅 FOCUS DETECTOR STARTING  (YOLO26 + DeepSORT)')
    print('='*50)
    
    # We set it to active immediately to monitor distractions endlessly.
    detector.pomodoro_active = True
    
    if use_native_window:
        print("\n📺 OpenCV window — press 'q' to quit")
        detector.run_webcam_native()
    else:
        print('\n📺 Jupyter inline display')
        detector.run_webcam()
    
    print('\n' + '='*50)
    print('  🌟 GREAT WORK TODAY! 🌟')
    print('='*50 + '\n')


print('\n🍅 YOLO26 + DeepSORT Pomodoro Detector initialized!')
print('\n📋 Quick Start:')
print('  test_sound()                               — verify alarm tones')
print('  test_webcam()                              — verify camera')
print('  start_focus_session()                      — Jupyter inline display')
print('  start_focus_session(use_native_window=True) — OpenCV window')
print('\n🔔 Sound Alarms (cross-platform: Windows / macOS / Linux):')
print('  phone    → rising 3-tone  (distraction threshold exceeded)')
print('  absence  → falling 2-tone (person absent too long)')
print('  complete → C-E-G ascending (work session done)')
print('  break    → E-C descending (break session ended)')
print('  Cooldown: 15s between repeated alarms | runs in background thread')
print('\n📦 DeepSORT tracking:')
print('  • Persistent Person IDs  — no ID flicker across frames')
print('  • Phone confirm  = 2 frames before interaction counted')
print('  • Phone linger   = 8 frames after phone disappears')
print('  • Lower PHONE_CONFIDENCE = 0.15 (was 0.20)')
print('\n🚀 YOLO26 vs YOLO11:')
print('  NMS-free end-to-end | STAL small-object boost | ~43% faster CPU | DFL removed')
print('\n⚡ Speed optimizations active:')
print(f'  IMGSZ={IMGSZ} | SKIP_FRAMES={SKIP_FRAMES} | JPEG_QUALITY={JPEG_QUALITY}')
print('  MJPEG webcam codec | buffer=1 | batch box extraction | fast FPS counter')

if __name__ == '__main__':
    # Start the session automatically when the script is run directly
    start_focus_session(use_native_window=True)