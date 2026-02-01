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

def detect_lanes(frame):
    img = frame.copy()
    h = img.shape[0]
    w = img.shape[1]
    
    # only look at bottom 60% of frame - thats where lanes actually are
    roi_start = int(h * 0.4)
    roi = img[roi_start:h, :]
    
    # convert to grayscale for edge detection
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # blur it a bit to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # canny edge detection - finds edges based on intensity changes
    edges = cv2.Canny(blur, 50, 150)
    
    # clean up the edges
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    
    # scan more lines to get better curve fit
    scan_lines = [int(edges.shape[0] * 0.2), int(edges.shape[0] * 0.35), 
                  int(edges.shape[0] * 0.5), int(edges.shape[0] * 0.65), 
                  int(edges.shape[0] * 0.8)]
    
    left_pts = []
    right_pts = []
    mid_x = w // 2
    
    for scan_y in scan_lines:
        row = edges[scan_y, :]
        
        # find left lane - scan from middle going left
        # look for edge pixels
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
    
    # draw the edge overlay so we can see what its detecting
    edge_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edge_color[edges > 0] = [0, 255, 0]  # green where edges detected
    
    # blend it with original
    img[roi_start:h, :] = cv2.addWeighted(img[roi_start:h, :], 0.7, edge_color, 0.3, 0)
    
    # need at least 3 pts per side or this wont work
    if len(left_pts) < 3 or len(right_pts) < 3:
        return img
    
    # split pts into separate lists
    l_x = []
    l_y = []
    for pt in left_pts:
        l_x.append(pt[0])
        l_y.append(pt[1])
    
    r_x = []
    r_y = []
    for pt in right_pts:
        r_x.append(pt[0])
        r_y.append(pt[1])
    
    # fit curves thru the points
    left_fit = np.polyfit(l_y, l_x, 2)
    right_fit = np.polyfit(r_y, r_x, 2)
    
    # make array of y coords to plot
    y_range = np.linspace(0, edges.shape[0]-1, edges.shape[0])
    
    # calc where x should be for each y on the curve
    left_x_vals = left_fit[0]*y_range**2 + left_fit[1]*y_range + left_fit[2]
    right_x_vals = right_fit[0]*y_range**2 + right_fit[1]*y_range + right_fit[2]
    
    # draw left side
    for j in range(len(y_range)-1):
        cv2.line(img, 
                (int(left_x_vals[j]), int(y_range[j]) + roi_start),
                (int(left_x_vals[j+1]), int(y_range[j+1]) + roi_start),
                (255, 0, 0), 3)
    
    # draw right side  
    for j in range(len(y_range)-1):
        cv2.line(img,
                (int(right_x_vals[j]), int(y_range[j]) + roi_start),
                (int(right_x_vals[j+1]), int(y_range[j+1]) + roi_start),
                (0, 0, 255), 3)
    
    # center is halfway between
    center_x = (left_x_vals + right_x_vals) / 2
    
    # draw center path
    for j in range(len(y_range)-1):
        cv2.line(img,
                (int(center_x[j]), int(y_range[j]) + roi_start),
                (int(center_x[j+1]), int(y_range[j+1]) + roi_start),
                (0, 255, 255), 3)
    
    # put a dot at the bottom
    bot_x = int(center_x[-1])
    cv2.circle(img, (bot_x, h - 20), 10, (0, 255, 255), -1)
    
    # show how far from middle we are
    diff = bot_x - mid_x
    cv2.putText(img, f"offset: {diff}px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
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
        
        img = detect_lanes(img)  # run it thru detection
        
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

