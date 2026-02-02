#Version 1.9.1
import cv2, numpy as np
from flask import Flask, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


FRAME_W, FRAME_H = 640, 480
ROI_TOP_FRAC = 0.40
HSV_LOW, HSV_HIGH = np.array([85, 80, 80]), np.array([115, 255, 255])
KERNEL = np.ones((5, 5), np.uint8)
BOUNDARY = "frame"

def open_camera(indices=(0, 1, 2)):
    for idx in indices:
        cam = cv2.VideoCapture(idx)
        ok, frame = cam.read()
        if cam.isOpened() and ok and frame is not None:
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
            print(f"Camera {idx} OK")
            return cam
        cam.release()
    raise RuntimeError("No working camera found")

camera = open_camera()

def detect_blue_lanes(frame):
    h, w = frame.shape[:2]
    roi_top = int(h * ROI_TOP_FRAC)
    roi = frame[roi_top:, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)

    mask_h = mask.shape[0]
    scan_y = (np.array([.2, .35, .5, .65, .8]) * (mask_h - 1)).astype(int)
    mid_x = w // 2
    left_pts, right_pts = [], []

    for y in scan_y:
        row = mask[y]
        xs = np.where(row > 0)[0]
        if len(xs) == 0:
            continue
    
        left_xs = xs[xs < mid_x]
        right_xs = xs[xs > mid_x]
    
        if len(left_xs) > 0:
            left_x = int(left_xs.max())
            left_pts.append((left_x, y))
    
        if len(right_xs) > 0:
            right_x = int(right_xs.min())
            right_pts.append((right_x, y))
    
    overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    overlay[mask > 0] = (0, 255, 0)
    roi = frame[roi_top:, :]
    frame[roi_top:, :] = cv2.addWeighted(roi, 0.7, overlay, 0.3, 0)

    if len(left_pts) < 3 or len(right_pts) < 3:
        return frame

    left_x = np.array([x for x, y in left_pts], dtype=np.int32)
    left_y = np.array([y for x, y in left_pts], dtype=np.int32)
    right_x = np.array([x for x, y in right_pts], dtype=np.int32)
    right_y = np.array([y for x, y in right_pts], dtype=np.int32)

    left_poly = np.polyfit(left_y, left_x, 2)
    right_poly = np.polyfit(right_y, right_x, 2)

    y = np.arange(mask_h)
    left_x = np.clip(np.polyval(left_poly, y), 0, w - 1).astype(int)
    right_x = np.clip(np.polyval(right_poly, y), 0, w - 1).astype(int)
    center_x = ((left_x + right_x) / 2).astype(int)
    y_img = (y + roi_top).astype(int)

    def draw_curve(x_vals, color):
        pts = np.stack([x_vals, y_img], axis=1).reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(frame, [pts], False, color, 3)

    draw_curve(left_x,  (255, 0, 0))
    draw_curve(right_x, (0, 0, 255))
    draw_curve(center_x,(0, 255, 255))

    bottom_center = int(center_x[-1])
    cv2.circle(frame, (bottom_center, h - 20), 10, (0, 255, 255), -1)
    cv2.putText(frame, f"offset: {bottom_center - mid_x}px", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame

def mjpeg_stream(processor=None):
    while True:
        ok, frame = camera.read()
        if not ok: 
            break
        if processor:
            frame = processor(frame)
        ok, jpg = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        yield (b"--" + BOUNDARY.encode() + b"\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_stream(), mimetype=f"multipart/x-mixed-replace; boundary={BOUNDARY}")

@app.route("/video_feed_processed")
def video_feed_processed():
    return Response(mjpeg_stream(detect_blue_lanes), mimetype=f"multipart/x-mixed-replace; boundary={BOUNDARY}")

@app.route("/")
def home():
    return ('<h1>Camera Server</h1>'
            '<p>Raw: <a href="/video_feed">/video_feed</a></p>'
            '<p>Lanes: <a href="/video_feed_processed">/video_feed_processed</a></p>')

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=8081, threaded=True)
    finally:
        camera.release()
