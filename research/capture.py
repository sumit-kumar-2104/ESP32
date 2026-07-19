"""
ESP32 WiFi CSI + Webcam Fusion — Research Data Collection & Live Visualization

This script:
  1. Connects to 2 ESP32 sensors via their REST API (movement score + motion state)
  2. Opens your USB webcam and runs real-time person detection (MediaPipe or YOLO)
  3. Shows a LIVE view of your REAL room with:
     - Bounding box around detected person(s)
     - WiFi sensing overlay (color-coded zones showing which ESP32 detects motion)
     - Movement intensity bar from each sensor
  4. Records synchronized data (CSI scores + person bounding boxes + timestamps) to CSV
     for later ML training (Phase 2: learn to predict position from CSI alone)

Hardware needed:
  - 2x ESP32 (flashed with ESPectre, connected to same WiFi)
  - USB webcam (external or built-in)
  - Laptop on same WiFi network

Usage:
  python capture.py                         # live view only
  python capture.py --record                # live view + save to CSV
  python capture.py --record --duration 300 # record for 5 minutes
  python capture.py --esp1 192.168.0.10 --esp2 192.168.0.11  # custom IPs

Press 'q' to quit, 's' to take a snapshot, 'r' to toggle recording.

Author: Research tool for SNU MPL lab
"""

import cv2
import numpy as np
import time
import csv
import argparse
import threading
import os
from datetime import datetime
from pathlib import Path

try:
    import mediapipe as mp
    # Check if legacy API (solutions.pose) is available
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose'):
        USE_MEDIAPIPE = True
    else:
        USE_MEDIAPIPE = False
        print("[INFO] mediapipe installed but legacy API not available, using motion detection")
except ImportError:
    USE_MEDIAPIPE = False
    print("[INFO] mediapipe not installed, using basic motion detection")
    print("       Install for person tracking: pip install mediapipe")

try:
    import requests
except ImportError:
    print("[ERROR] 'requests' package required. Run: pip install requests")
    exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# ESP32 Sensor Reader (threaded, non-blocking)
# ═══════════════════════════════════════════════════════════════════════════════
class ESP32Sensor:
    """Reads movement score and motion state from an ESP32 running ESPectre."""

    def __init__(self, ip, name="ESP32"):
        self.ip = ip
        self.name = name
        self.base_url = f"http://{ip}"
        self.movement_score = 0.0
        self.motion_detected = False
        self.rssi = -100
        self.connected = False
        self.last_update = 0
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        """Start background polling thread."""
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _poll_loop(self):
        """Use Server-Sent Events (SSE) stream — same as the web dashboard uses."""
        import json
        while self._running:
            try:
                # Connect to the SSE events stream
                r = requests.get(f"{self.base_url}/events", stream=True, timeout=5)
                if r.status_code != 200:
                    raise ConnectionError(f"HTTP {r.status_code}")
                with self._lock:
                    self.connected = True
                for line in r.iter_lines(decode_unicode=True):
                    if not self._running:
                        break
                    if not line or not line.startswith('data:'):
                        continue
                    try:
                        data = json.loads(line[5:].strip())
                    except (json.JSONDecodeError, ValueError):
                        continue
                    eid = data.get('id', '')
                    with self._lock:
                        self.last_update = time.time()
                        if 'movement' in eid and data.get('value') is not None:
                            try:
                                self.movement_score = float(data['value'])
                            except (ValueError, TypeError):
                                pass
                        elif 'motion' in eid:
                            self.motion_detected = (
                                data.get('state') == 'ON' or
                                data.get('value') is True or
                                data.get('value') == 1
                            )
            except Exception:
                with self._lock:
                    self.connected = False
                time.sleep(1)  # wait before reconnecting

    def get_state(self):
        with self._lock:
            return {
                'name': self.name,
                'ip': self.ip,
                'movement': self.movement_score,
                'motion': self.motion_detected,
                'connected': self.connected,
                'age': time.time() - self.last_update if self.last_update > 0 else 999,
            }


# ═══════════════════════════════════════════════════════════════════════════════
# Person Detector — YOLO (best) > HOG > motion fallback
# ═══════════════════════════════════════════════════════════════════════════════
class PersonDetector:
    def __init__(self):
        self.mode = None
        self.yolo = None

        # 1) Try YOLO (ultralytics) — most reliable, one clean box
        try:
            from ultralytics import YOLO
            self.yolo = YOLO('yolov8n.pt')  # auto-downloads ~6MB on first run
            self.mode = 'yolo'
            print("[DET]   Using YOLOv8 person detection (accurate)")
            return
        except Exception as e:
            print(f"[DET]   YOLO unavailable ({e.__class__.__name__}); trying HOG...")

        # 2) Try MediaPipe legacy pose (skeleton)
        if USE_MEDIAPIPE:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False, model_complexity=1,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self.mp_draw = mp.solutions.drawing_utils
            self.mode = 'mediapipe'
            print("[DET]   Using MediaPipe Pose")
            return

        # 3) HOG people detector (ships with OpenCV, no download)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.mode = 'hog'
        print("[DET]   Using OpenCV HOG people detector")

    def detect(self, frame):
        """Returns list: [{'bbox':(x1,y1,x2,y2),'center':(cx,cy),'confidence':float}]"""
        h, w = frame.shape[:2]
        detections = []

        if self.mode == 'yolo':
            results = self.yolo(frame, classes=[0], verbose=False)  # class 0 = person
            best = None
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf < 0.4:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if best is None or conf > best['confidence']:
                        best = {'bbox': (x1, y1, x2, y2),
                                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                                'confidence': conf}
            if best:
                detections.append(best)  # keep only the single best person

        elif self.mode == 'mediapipe':
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                xs = [l.x * w for l in lm if l.visibility > 0.5]
                ys = [l.y * h for l in lm if l.visibility > 0.5]
                if xs and ys:
                    x1, x2 = max(0, int(min(xs)) - 20), min(w, int(max(xs)) + 20)
                    y1, y2 = max(0, int(min(ys)) - 20), min(h, int(max(ys)) + 20)
                    detections.append({'bbox': (x1, y1, x2, y2),
                                       'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                                       'confidence': 0.9})
                self.mp_draw.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 128), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(0, 200, 100), thickness=2))

        else:  # HOG — keep only the largest/most confident box
            rects, weights = self.hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
            best = None
            for (x, y, bw, bh), wt in zip(rects, weights):
                conf = float(wt)
                if conf < 0.3:
                    continue
                if best is None or conf > best['confidence']:
                    best = {'bbox': (x, y, x + bw, y + bh),
                            'center': (x + bw // 2, y + bh // 2),
                            'confidence': conf}
            if best:
                detections.append(best)

        return detections


# ═══════════════════════════════════════════════════════════════════════════════
# Data Recorder
# ═══════════════════════════════════════════════════════════════════════════════
class DataRecorder:
    def __init__(self, output_dir="data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.output_dir / f"capture_{ts}.csv"
        self.file = open(self.csv_path, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            'timestamp', 'elapsed_sec',
            'esp1_movement', 'esp1_motion',
            'esp2_movement', 'esp2_motion',
            'person_detected', 'person_x', 'person_y',
            'person_bbox_x1', 'person_bbox_y1', 'person_bbox_x2', 'person_bbox_y2',
        ])
        self.start_time = time.time()
        self.rows = 0

    def write(self, esp1_state, esp2_state, detections):
        now = time.time()
        elapsed = now - self.start_time
        det = detections[0] if detections else None
        self.writer.writerow([
            f"{now:.3f}", f"{elapsed:.2f}",
            f"{esp1_state['movement']:.4f}", int(esp1_state['motion']),
            f"{esp2_state['movement']:.4f}", int(esp2_state['motion']),
            int(det is not None),
            det['center'][0] if det else '',
            det['center'][1] if det else '',
            det['bbox'][0] if det else '', det['bbox'][1] if det else '',
            det['bbox'][2] if det else '', det['bbox'][3] if det else '',
        ])
        self.rows += 1

    def close(self):
        self.file.close()
        print(f"\n[DATA] Saved {self.rows} rows to {self.csv_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Overlay Drawing
# ═══════════════════════════════════════════════════════════════════════════════
def draw_overlay(frame, esp1, esp2, detections, recording):
    h, w = frame.shape[:2]
    s1 = esp1.get_state()
    s2 = esp2.get_state()

    # ─── ESP32 status bars (top) ───
    bar_h = 60
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (15, 15, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    for i, s in enumerate([s1, s2]):
        x_off = 10 + i * (w // 2)
        color = (0, 200, 100) if s['connected'] else (0, 0, 180)
        status = "●" if s['connected'] else "○"

        cv2.putText(frame, f"{status} {s['name']} ({s['ip']})", (x_off, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Movement bar
        bar_x, bar_y, bar_w = x_off, 28, w // 2 - 30
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 12), (30, 30, 50), -1)
        fill = int(bar_w * min(1.0, s['movement'] / 10.0))
        if fill > 0:
            # Color gradient green->yellow->red
            norm = s['movement'] / 10.0
            r = int(min(255, norm * 2 * 255))
            g = int(min(255, (1 - norm) * 2 * 255))
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + 12), (0, g, r), -1)
        cv2.putText(frame, f"{s['movement']:.1f}", (bar_x + bar_w + 5, bar_y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Motion state
        motion_text = "MOTION" if s['motion'] else "idle"
        motion_color = (0, 0, 255) if s['motion'] else (0, 180, 0)
        cv2.putText(frame, motion_text, (x_off, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, motion_color, 1 + int(s['motion']))

    # ─── Person bounding boxes ───
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cx, cy = det['center']
        # Green box + center dot
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 128), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
        cv2.putText(frame, f"Person ({cx},{cy})", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 1)

    # ─── Motion intensity heatmap overlay ───
    max_movement = max(s1['movement'], s2['movement'])
    if max_movement > 1.5:
        norm = min(1.0, max_movement / 10.0)
        heat_overlay = frame.copy()
        # Red tint proportional to motion
        red_layer = np.zeros_like(frame)
        red_layer[:, :, 2] = int(norm * 80)  # red channel
        cv2.addWeighted(frame, 1.0, red_layer, norm * 0.3, 0, frame)

    # ─── Recording indicator ───
    if recording:
        cv2.circle(frame, (w - 20, 20), 8, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w - 55, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # ─── Instructions ───
    cv2.putText(frame, "q=quit | r=record | s=snapshot", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    return frame


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="ESP32 WiFi Sensing + Webcam Fusion")
    parser.add_argument('--esp1', default='192.168.0.10', help='IP of ESP32 #1')
    parser.add_argument('--esp2', default='', help='IP of ESP32 #2 (optional)')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (0=default)')
    parser.add_argument('--record', action='store_true', help='Start recording immediately')
    parser.add_argument('--duration', type=int, default=0, help='Recording duration in seconds (0=manual)')
    parser.add_argument('--output', default='data', help='Output directory for recordings')
    args = parser.parse_args()

    print("=" * 60)
    print("  ESP32 WiFi CSI + Webcam — Research Capture Tool")
    print("=" * 60)
    print()

    # Initialize ESP32 sensors
    esp1 = ESP32Sensor(args.esp1, "ESP32-1")
    esp1.start()
    print(f"[ESP32] Connecting to sensor 1: {args.esp1}")

    if args.esp2:
        esp2 = ESP32Sensor(args.esp2, "ESP32-2")
        esp2.start()
        print(f"[ESP32] Connecting to sensor 2: {args.esp2}")
    else:
        # Dummy sensor if only one ESP32
        esp2 = ESP32Sensor("0.0.0.0", "ESP32-2 (not connected)")
        print("[ESP32] Only 1 sensor configured. Use --esp2 for second.")

    # Initialize camera
    print(f"[CAM]   Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check connection or try --camera 1")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[CAM]   Resolution: {actual_w}x{actual_h}")

    # Initialize person detector
    detector = PersonDetector()
    print(f"[DET]   Person detector: {'MediaPipe Pose' if USE_MEDIAPIPE else 'Motion-based (install mediapipe for better tracking)'}")

    # Recording
    recording = args.record
    recorder = None
    if recording:
        recorder = DataRecorder(args.output)
        print(f"[REC]   Recording to {recorder.csv_path}")

    print()
    print("  Window open. Controls:")
    print("    q = quit  |  r = toggle recording  |  s = screenshot")
    print()

    start_time = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect people
            detections = detector.detect(frame)

            # Draw overlay
            frame = draw_overlay(frame, esp1, esp2, detections, recording)

            # Record data
            if recording and recorder:
                recorder.write(esp1.get_state(), esp2.get_state(), detections)
                if args.duration > 0 and (time.time() - start_time) > args.duration:
                    print(f"\n[REC] Duration {args.duration}s reached, stopping recording.")
                    recording = False
                    recorder.close()
                    recorder = None

            # Show frame
            cv2.imshow("ESP32 WiFi Sensing + Camera", frame)
            frame_count += 1

            # Keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                if recording:
                    recording = False
                    if recorder:
                        recorder.close()
                        recorder = None
                else:
                    recording = True
                    recorder = DataRecorder(args.output)
                    print(f"[REC] Started recording to {recorder.csv_path}")
            elif key == ord('s'):
                snap_path = f"snapshot_{datetime.now().strftime('%H%M%S')}.jpg"
                cv2.imwrite(snap_path, frame)
                print(f"[SNAP] Saved {snap_path}")

    except KeyboardInterrupt:
        pass
    finally:
        esp1.stop()
        if args.esp2:
            esp2.stop()
        if recorder:
            recorder.close()
        cap.release()
        cv2.destroyAllWindows()
        elapsed = time.time() - start_time
        print(f"\n[DONE] {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} fps)")


if __name__ == '__main__':
    main()
