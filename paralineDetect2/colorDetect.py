import cv2
import numpy as np
from flask import Flask, Response
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)

# Try different camera indices
camera = None
for index in [0, 1, 2]:
    print(f"Trying camera index {index}...")
    cam = cv2.VideoCapture(index)
    if cam.isOpened():
        ret, frame = cam.read()
        if ret and frame is not None:
            print(f"Sucess! Camera {index} is working!")
            camera = cam
            break
        else:
            print(f"Camera {index} opened but can't read frames")
            cam.release()
    else:
        print(f"Camera {index} failed to open")

if camera is None:
    print("ERROR: No working camera found!")
    exit(1)

# Set camera properties
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def detect_blue_lanes(frame):
    img = frame.copy()
    h = img.shape[0]
    w = img.shape[1]
    
    # only look at bottom 60% of frame - thats where lanes actually are
    roi_start = int(h * 0.4)
    roi = img[roi_start:h, :]
    
    # convert roi to hsv
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # tuned for that light blue painters tape
    # the tape looks like a cyan-ish blue
    low = np.array([85, 80, 80])
    high = np.array([115, 255, 255])
    
    mask = cv2.inRange(hsv, low, high)
    
    # clean up the mask - get rid of small noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # scan more lines to get better curve fit
    scan_lines = [int(mask.shape[0] * 0.2), int(mask.shape[0] * 0.35), 
                  int(mask.shape[0] * 0.5), int(mask.shape[0] * 0.65), 
                  int(mask.shape[0] * 0.8)]
    
    left_pts = []
    right_pts = []
    mid_x = w // 2
    
    for scan_y in scan_lines:
        row = mask[scan_y, :]
        
        # find left lane - scan from middle going left
        left_x = None
        for x in range(mid_x, 0, -1):
            if row[x] > 0:
                left_x = x
                break
        
        # find right lane - scan from middle going right  
        right_x = None
        for x in range(mid_x, w):
            if row[x] > 0:
                right_x = x
                break
        
        if left_x is not None:
            left_pts.append((left_x, scan_y))
        if right_x is not None:
            right_pts.append((right_x, scan_y))
    
    # draw the mask overlay so we can see what its detecting
    roi_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    roi_color[mask > 0] = [0, 255, 0]  # green where tape detected
    
    # blend it with original
    img[roi_start:h, :] = cv2.addWeighted(img[roi_start:h, :], 0.7, roi_color, 0.3, 0)
    
    # check if we got pts
    got_left = len(left_pts) >= 3
    got_right = len(right_pts) >= 3
    if got_left == False or got_right == False:
        return img
    
    # split out x and y
    left_x_vals = []
    left_y_vals = []
    for pt in left_pts:
        left_x_vals.append(pt[0])
        left_y_vals.append(pt[1])
    
    right_x_vals = []
    right_y_vals = []
    for pt in right_pts:
        right_x_vals.append(pt[0])
        right_y_vals.append(pt[1])
    
    # fit curves
    left_curve = np.polyfit(left_y_vals, left_x_vals, 2)
    right_curve = np.polyfit(right_y_vals, right_x_vals, 2)
    
    # y vals for plotting
    plot_y = np.linspace(0, mask.shape[0]-1, mask.shape[0])
    
    # calculate x for each y using curve equation
    left_curve_x = []
    right_curve_x = []
    for y_val in plot_y:
        lx = left_curve[0]*y_val**2 + left_curve[1]*y_val + left_curve[2]
        rx = right_curve[0]*y_val**2 + right_curve[1]*y_val + right_curve[2]
        left_curve_x.append(lx)
        right_curve_x.append(rx)
    
    # draw left lane in blue
    i = 0
    while i < len(plot_y)-1:
        x1 = int(left_curve_x[i])
        y1 = int(plot_y[i]) + roi_start
        x2 = int(left_curve_x[i+1])
        y2 = int(plot_y[i+1]) + roi_start
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        i = i + 1
    
    # draw right lane in red
    i = 0
    while i < len(plot_y)-1:
        x1 = int(right_curve_x[i])
        y1 = int(plot_y[i]) + roi_start
        x2 = int(right_curve_x[i+1])
        y2 = int(plot_y[i+1]) + roi_start
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        i = i + 1
    
    # midpoint between lanes
    mid_curve_x = []
    for idx in range(len(left_curve_x)):
        mid_val = (left_curve_x[idx] + right_curve_x[idx]) / 2
        mid_curve_x.append(mid_val)
    
    # draw center line yellow
    i = 0
    while i < len(plot_y)-1:
        x1 = int(mid_curve_x[i])
        y1 = int(plot_y[i]) + roi_start
        x2 = int(mid_curve_x[i+1])
        y2 = int(plot_y[i+1]) + roi_start
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 3)
        i = i + 1
    
    # yellow dot at bottom center
    center_bottom = int(mid_curve_x[-1])
    cv2.circle(img, (center_bottom, h - 20), 10, (0, 255, 255), -1)
    
    # show how far off center
    offset = center_bottom - mid_x
    txt = f"offset: {offset}px"
    cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return img

def generate_frames_raw():
    # basic feed, no processing
    while True:
        ok, img = camera.read()
        if not ok:
            print("cam read broke")
            break
        
        works, buf = cv2.imencode('.jpg', img)
        if works:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
        else:
            print("jpg encode failed idk why")

def generate_frames_processed():
    # ok this one does the lane detection stuff
    while True:
        ok, img = camera.read()
        if not ok:
            print("ugh frame read failed")
            break
        
        img = detect_blue_lanes(img)  # run it thru detection
        
        ok2, buf = cv2.imencode('.jpg', img)
        if not ok2:
            print("encode didnt work")
            continue
        
        # send it back as mjpeg
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'


@app.route('/video_feed')
def video_feed():
    # just raw cam no processing
    return Response(generate_frames_raw(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_processed') 
def video_feed_processed():
    # this ones got the lane stuff on it
    return Response(generate_frames_processed(),
        mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def home():
    # quick page to check if servers up
    page = '<h1>Camera Server</h1>'
    page += '<p>Raw: <a href="/video_feed">/video_feed</a></p>'
    page += '<p>Lanes: <a href="/video_feed_processed">/video_feed_processed</a></p>'
    return page

if __name__ == '__main__':
    print("\nCamera server starting on http://localhost:8081")
    print("Raw feed at: http://localhost:8081/video_feed")
    print("Lane detection feed at: http://localhost:8081/video_feed_processed\n")
    try:
        app.run(host='0.0.0.0', port=8081, threaded=True)
    finally:
        camera.release()
        print("Camera released")
