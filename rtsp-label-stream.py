import subprocess
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os

# Set the camera URI
camera_uri = os.environ['CAMERA_URI'] = 'rtsp://admin:VgDkf2dPW3YHEn@10.0.30.10:554/h264Preview_05_sub'
if camera_uri is None:
    raise ValueError('CAMERA_URI environment variable not set')

# Start RTSP receiver and sender
rtsp_url = 'rtsp://127.0.0.1:8554/mystream'
ffplay_process = subprocess.Popen(['ffplay', '-rtsp_flags', 'listen', rtsp_url]) 
ffmpeg_cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '1280x720', '-r', '25', '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'ultrafast', '-f', 'rtsp', rtsp_url ]
with open('ffmpeg.log', 'w') as f:
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=f, stderr=f)

# Start YOLO
model = YOLO('yolov8n.pt')
results = model(camera_uri, stream=True)

# Process results and send to RTSP sender
for result in results:
    pil_image = result.plot(pil=True)
    image_np = np.array(pil_image)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    proc.stdin.write(image_cv2.tobytes())
        
