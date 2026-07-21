"""
WiFi CSI Zone Localization Experiment — Data Collection + Heatmap

Standard wireless perception research approach:
  1. Divide the lobby into a grid of zones (e.g. 3x3 or 4x3)
  2. Walk through each zone, recording WiFi CSI from both ESP32s + camera position
  3. Train a model: CSI features → zone prediction
  4. Visualize as a heatmap showing predicted presence

This script handles the DATA COLLECTION phase:
  - Shows your camera feed with a zone grid overlay
  - Records (zone_label, esp1_score, esp2_score, person_x, person_y, timestamp) to CSV
  - Two modes:
    * GUIDED: tells you which zone to stand in (systematic)
    * FREE: you walk freely and it auto-labels zones from camera position

Usage:
  python collect_zones.py --esp1 192.168.0.10 --esp2 192.168.0.16 --mode guided
  python collect_zones.py --esp1 192.168.0.10 --esp2 192.168.0.16 --mode free --duration 300

Hardware setup for your lobby:
  ┌──────────────────────────────────────────┐
  │              WALL (WiFi Router)            │
  │                    📶                      │
  │                                            │
  │  Zone(0,0)  Zone(1,0)  Zone(2,0)  Zone(3,0)│
  │                                            │
  │  Zone(0,1)  Zone(1,1)  Zone(2,1)  Zone(3,1)│
  │                                            │
  │  Zone(0,2)  Zone(1,2)  Zone(2,2)  Zone(3,2)│
  │                                            │
  │ [ESP32-1]                       [ESP32-2]  │
  │  (corner)                       (corner)   │
  └──────────────────────────────────────────┘
         📷 Camera (where you sit / laptop)

Author: SNU MPL Research
"""

import cv2
import numpy as np
import time
import csv
import json
import argparse
import threading
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    print("[ERROR] Run: pip install requests")
    exit(1)

try:
    from ultralytics import YOLO
    USE_YOLO = True
except ImportError:
    USE_YOLO = False


# ═══════════════════════════════════════════════════════════════════════════════
# ESP32 Sensor (SSE events stream)
# ═══════════════════════════════════════════════════════════════════════════════
class ESP32Sensor:
    def __init__(self, ip, name="ESP32"):
        self.ip = ip
        self.name = name
        self.movement = 0.0
        self.motion = False
        self.connected = False
        self.last_update = 0
        self._lock = threading.Lock()
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=self._stream_loop, daemon=True).start()

    def stop(self):
        self._running = False

    def _stream_loop(self):
        import json as _json
        while self._running:
            try:
                r = requests.get(f"http://{self.ip}/events", stream=True, timeout=5)
                with self._lock:
                    self.connected = True
                for line in r.iter_lines(decode_unicode=True):
                    if not self._running:
                        break
                    if not line or not line.startswith('data:'):
                        continue
                    try:
                        d = _json.loads(line[5:].strip())
                    except (ValueError, _json.JSONDecodeError):
                        continue
                    eid = d.get('id', '')
                    with self._lock:
                        self.last_update = time.time()
                        if 'movement' in eid and d.get('value') is not None:
                            try:
                                self.movement = float(d['value'])
                            except (ValueError, TypeError):
                                pass
                        elif 'motion' in eid:
                            self.motion = (d.get('state') == 'ON' or
                                           d.get('value') in [True, 1])
            except Exception:
                with self._lock:
                    self.connected = False
                time.sleep(1)

    def get(self):
        with self._lock:
            return {'name': self.name, 'movement': self.movement,
                    'motion': self.motion, 'connected': self.connected}


# ═══════════════════════════════════════════════════════════════════════════════
# Person Detector
# ═══════════════════════════════════════════════════════════════════════════════
class Detector:
    def __init__(self):
        if USE_YOLO:
            self.yolo = YOLO('yolov8n.pt')
            print("[DET] YOLOv8 loaded")
        else:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            print("[DET] OpenCV HOG loaded (install ultralytics for better detection)")

    def detect(self, frame):
        """Returns best person detection: {'bbox', 'center', 'conf'} or None"""
        if USE_YOLO:
            results = self.yolo(frame, classes=[0], verbose=False, conf=0.4)
            best = None
            for r in results:
                for box in r.boxes:
                    c = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if best is None or c > best['conf']:
                        best = {'bbox': (x1, y1, x2, y2),
                                'center': ((x1+x2)//2, (y1+y2)//2), 'conf': c}
            return best
        else:
            rects, weights = self.hog.detectMultiScale(frame, winStride=(8,8),
                                                       padding=(8,8), scale=1.05)
            best = None
            for (x,y,w,h), wt in zip(rects, weights):
                if wt > 0.3 and (best is None or wt > best['conf']):
                    best = {'bbox': (x,y,x+w,y+h),
                            'center': (x+w//2, y+h//2), 'conf': float(wt)}
            return best


# ═══════════════════════════════════════════════════════════════════════════════
# Zone Grid
# ═══════════════════════════════════════════════════════════════════════════════
class ZoneGrid:
    def __init__(self, cols=4, rows=3, frame_w=1280, frame_h=720):
        self.cols = cols
        self.rows = rows
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.cell_w = frame_w // cols
        self.cell_h = frame_h // rows
        # Accumulator for heatmap
        self.heat = np.zeros((rows, cols), dtype=np.float32)
        self.counts = np.zeros((rows, cols), dtype=np.int32)

    def point_to_zone(self, x, y):
        """Convert pixel (x, y) to grid zone (col, row)."""
        col = min(self.cols - 1, max(0, x // self.cell_w))
        row = min(self.rows - 1, max(0, y // self.cell_h))
        return int(col), int(row)

    def zone_label(self, col, row):
        return f"Z({col},{row})"

    def accumulate(self, col, row, esp1_mv, esp2_mv):
        """Add a data point to the heatmap accumulator."""
        self.heat[row, col] += max(esp1_mv, esp2_mv)
        self.counts[row, col] += 1

    def draw_grid(self, frame, active_zone=None, guided_zone=None):
        """Draw zone grid on frame."""
        h, w = frame.shape[:2]
        cw, ch = w // self.cols, h // self.rows

        for c in range(self.cols):
            for r in range(self.rows):
                x1, y1 = c * cw, r * ch
                x2, y2 = x1 + cw, y1 + ch

                # Fill active zone with green tint
                if active_zone == (c, r):
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 180, 0), -1)
                    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

                # Fill guided target with yellow tint
                if guided_zone == (c, r):
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 255), -1)
                    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

                # Grid lines
                cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 80), 1)

                # Zone label
                label = self.zone_label(c, r)
                cv2.putText(frame, label, (x1 + 5, y1 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 140), 1)

    def get_heatmap_image(self, width=400, height=300):
        """Generate a heatmap visualization from accumulated data."""
        # Normalize
        avg = np.zeros_like(self.heat)
        mask = self.counts > 0
        avg[mask] = self.heat[mask] / self.counts[mask]

        # Normalize to 0-255
        if avg.max() > 0:
            norm = (avg / avg.max() * 255).astype(np.uint8)
        else:
            norm = np.zeros_like(avg, dtype=np.uint8)

        # Resize to display size
        heatmap = cv2.resize(norm, (width, height), interpolation=cv2.INTER_NEAREST)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Add grid lines and labels
        cw, ch = width // self.cols, height // self.rows
        for c in range(self.cols):
            for r in range(self.rows):
                x1, y1 = c * cw, r * ch
                cv2.rectangle(heatmap_color, (x1, y1), (x1+cw, y1+ch), (255,255,255), 1)
                cv2.putText(heatmap_color, self.zone_label(c, r), (x1+4, y1+16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
                # Show average value
                val = avg[r, c]
                if val > 0:
                    cv2.putText(heatmap_color, f"{val:.1f}", (x1+4, y1+ch-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)

        return heatmap_color


# ═══════════════════════════════════════════════════════════════════════════════
# Data Logger
# ═══════════════════════════════════════════════════════════════════════════════
class DataLogger:
    def __init__(self, output_dir="data"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = Path(output_dir) / f"run_{ts}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.dir / f"zones_{ts}.csv"
        self.f = open(self.csv_path, 'w', newline='')
        self.w = csv.writer(self.f)
        self.w.writerow([
            'timestamp', 'elapsed_s',
            'zone_col', 'zone_row', 'zone_label',
            'esp1_movement', 'esp1_motion',
            'esp2_movement', 'esp2_motion',
            'person_x', 'person_y',
            'person_x_norm', 'person_y_norm',  # 0-1 normalized
        ])
        self.start = time.time()
        self.rows = 0

    def log(self, zone_col, zone_row, esp1, esp2, px, py, frame_w, frame_h):
        now = time.time()
        self.w.writerow([
            f"{now:.3f}", f"{now - self.start:.2f}",
            zone_col, zone_row, f"Z({zone_col},{zone_row})",
            f"{esp1['movement']:.4f}", int(esp1['motion']),
            f"{esp2['movement']:.4f}", int(esp2['motion']),
            px, py,
            f"{px/frame_w:.4f}", f"{py/frame_h:.4f}",
        ])
        self.rows += 1

    def close(self):
        self.f.close()
        print(f"\n[DATA] Saved {self.rows} rows → {self.csv_path}")
        return self.csv_path


# ═══════════════════════════════════════════════════════════════════════════════
# Guided Collection
# ═══════════════════════════════════════════════════════════════════════════════
class GuidedCollector:
    """Tells you which zone to stand in, collects N seconds per zone."""

    def __init__(self, grid, seconds_per_zone=15):
        self.grid = grid
        self.secs = seconds_per_zone
        self.zones = [(c, r) for r in range(grid.rows) for c in range(grid.cols)]
        self.zones.append(None)  # final = empty room
        self.current_idx = 0
        self.zone_start = None
        self.started = False

    def current_zone(self):
        if self.current_idx >= len(self.zones):
            return None
        return self.zones[self.current_idx]

    def start(self):
        self.started = True
        self.zone_start = time.time()

    def tick(self):
        """Returns True if all zones are done."""
        if not self.started or self.current_idx >= len(self.zones):
            return True
        if time.time() - self.zone_start >= self.secs:
            self.current_idx += 1
            self.zone_start = time.time()
            if self.current_idx >= len(self.zones):
                return True
        return False

    def instruction(self):
        z = self.current_zone()
        if z is None:
            if self.current_idx >= len(self.zones):
                return "DONE! All zones collected."
            return "EMPTY ROOM: Leave the area (baseline)"
        elapsed = time.time() - self.zone_start if self.zone_start else 0
        remain = max(0, self.secs - elapsed)
        return f"Stand in ZONE {self.grid.zone_label(*z)} — {remain:.0f}s remaining"


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="WiFi CSI Zone Localization — Data Collection")
    parser.add_argument('--esp1', default='192.168.0.10', help='IP of ESP32 #1')
    parser.add_argument('--esp2', default='192.168.0.16', help='IP of ESP32 #2')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--mode', choices=['free', 'guided'], default='free',
                        help='free=walk around, guided=zone-by-zone systematic collection')
    parser.add_argument('--duration', type=int, default=0, help='Duration in sec (0=manual stop)')
    parser.add_argument('--cols', type=int, default=4, help='Grid columns')
    parser.add_argument('--rows', type=int, default=3, help='Grid rows')
    parser.add_argument('--secs-per-zone', type=int, default=15, help='Seconds per zone (guided mode)')
    parser.add_argument('--output', default='data', help='Output directory')
    args = parser.parse_args()

    print("=" * 60)
    print("  WiFi CSI Zone Localization — Data Collection")
    print("=" * 60)
    print(f"  Mode: {args.mode}")
    print(f"  Grid: {args.cols} x {args.rows} = {args.cols * args.rows} zones")
    print(f"  ESP32-1: {args.esp1}")
    print(f"  ESP32-2: {args.esp2}")
    print()

    # Start sensors
    esp1 = ESP32Sensor(args.esp1, "ESP32-1")
    esp2 = ESP32Sensor(args.esp2, "ESP32-2")
    esp1.start()
    esp2.start()
    print("[ESP] Connecting to sensors...")
    time.sleep(2)

    # Camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera"); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[CAM] {fw}x{fh}")

    # Detector
    detector = Detector()

    # Grid
    grid = ZoneGrid(args.cols, args.rows, fw, fh)

    # Logger
    logger = DataLogger(args.output)

    # Guided mode
    guide = None
    if args.mode == 'guided':
        guide = GuidedCollector(grid, args.secs_per_zone)
        print(f"\n[GUIDED] {args.secs_per_zone}s per zone, {len(guide.zones)} zones total")
        print("         Press SPACE to start\n")

    recording = True
    start_time = time.time()
    show_heatmap = False

    print("[READY] Controls: q=quit, h=toggle heatmap, SPACE=start guided")
    print()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect person
            det = detector.detect(frame)
            person_pos = det['center'] if det else None
            active_zone = None

            if person_pos:
                active_zone = grid.point_to_zone(*person_pos)

            # Get sensor data
            s1 = esp1.get()
            s2 = esp2.get()

            # Record
            if recording and person_pos:
                col, row = active_zone
                grid.accumulate(col, row, s1['movement'], s2['movement'])
                logger.log(col, row, s1, s2, person_pos[0], person_pos[1], fw, fh)

            # Guided mode logic
            guided_zone = None
            if guide:
                if guide.started:
                    done = guide.tick()
                    if done:
                        print("\n[GUIDED] Collection complete!")
                        break
                guided_zone = guide.current_zone()

            # ─── Draw ───
            # Grid overlay
            grid.draw_grid(frame, active_zone=active_zone, guided_zone=guided_zone)

            # Person box
            if det:
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
                cv2.circle(frame, person_pos, 5, (0, 255, 255), -1)

            # ESP32 status bar (top)
            bar_h = 55
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (fw, bar_h), (10, 10, 25), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

            for i, s in enumerate([s1, s2]):
                xo = 10 + i * (fw // 2)
                col = (0, 200, 100) if s['connected'] else (0, 0, 180)
                sym = "●" if s['connected'] else "○"
                cv2.putText(frame, f"{sym} {s['name']}", (xo, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
                # Bar
                bw = fw // 2 - 40
                cv2.rectangle(frame, (xo, 22), (xo + bw, 34), (30, 30, 50), -1)
                fill = int(bw * min(1.0, s['movement'] / 10.0))
                if fill > 0:
                    n = s['movement'] / 10.0
                    r = int(min(255, n * 2 * 255))
                    g = int(min(255, (1 - n) * 2 * 255))
                    cv2.rectangle(frame, (xo, 22), (xo + fill, 34), (0, g, r), -1)
                cv2.putText(frame, f"{s['movement']:.1f}", (xo + bw + 4, 33),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                state = "MOTION" if s['motion'] else "idle"
                sc = (0, 0, 255) if s['motion'] else (0, 180, 0)
                cv2.putText(frame, state, (xo, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, sc, 1)

            # Guided instruction
            if guide:
                txt = guide.instruction()
                cv2.putText(frame, txt, (10, fh - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)

            # Zone label if person detected
            if active_zone:
                cv2.putText(frame, f"You: {grid.zone_label(*active_zone)}", (fw - 150, fh - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 2)

            # Recording indicator
            cv2.circle(frame, (fw - 15, 15), 7, (0, 0, 255), -1)
            cv2.putText(frame, f"REC {logger.rows}", (fw - 100, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # Duration check
            if args.duration > 0 and (time.time() - start_time) > args.duration:
                break

            # Show main window
            cv2.imshow("WiFi CSI Zone Collection", frame)

            # Show heatmap window
            if show_heatmap:
                hm = grid.get_heatmap_image(480, 360)
                cv2.imshow("WiFi Heatmap", hm)

            # Keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                show_heatmap = not show_heatmap
                if not show_heatmap:
                    cv2.destroyWindow("WiFi Heatmap")
            elif key == ord(' '):
                if guide and not guide.started:
                    guide.start()
                    print("[GUIDED] Started! Go to the first zone.")

    except KeyboardInterrupt:
        pass
    finally:
        csv_path = logger.close()
        esp1.stop()
        esp2.stop()
        cap.release()
        cv2.destroyAllWindows()

        # Save final heatmap as image
        hm = grid.get_heatmap_image(800, 600)
        hm_path = logger.dir / f"heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(str(hm_path), hm)
        print(f"[HEAT] Heatmap saved → {hm_path}")

        # Save experiment metadata
        meta = {
            'timestamp': datetime.now().isoformat(),
            'grid': {'cols': args.cols, 'rows': args.rows},
            'esp1': args.esp1, 'esp2': args.esp2,
            'mode': args.mode, 'data_file': str(csv_path),
            'frame_size': [fw, fh],
            'total_samples': logger.rows,
        }
        meta_path = logger.dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"[META] Metadata saved → {meta_path}")


if __name__ == '__main__':
    main()
