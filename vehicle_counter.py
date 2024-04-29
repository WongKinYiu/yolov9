import cv2
import torch
import numpy as np
import time
from absl import app, flags
from absl.flags import FLAGS
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape

# Define command line flags
flags.DEFINE_string('weights', 'yolov9-c-converted.pt', 'Path to YOLOv9 Weights')
flags.DEFINE_string('video', 'inference/video/highway.mp4', 'Path to input video or webcam index (0)')
flags.DEFINE_string('classes', 'coco.names', 'Class Names')
flags.DEFINE_float('conf', 0.50, 'confidence threshold')

def show_fps(frame, fps):    
    x, y, w, h = 10, 10, 330, 45

    # Draw black background rectangle
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,0), -1)

    # Text FPS
    cv2.putText(frame, "FPS: " + str(fps), (20,52), cv2.FONT_HERSHEY_PLAIN, 3.5, (0,255,0), 3)

def main(_argv):
    # Initialize the video capture
    video_input = FLAGS.video
    # Check if the video input is an integer (webcam index)
    if FLAGS.video.isdigit():
        video_input = int(video_input)
        cap = cv2.VideoCapture(video_input)
    else:
        cap = cv2.VideoCapture(video_input)

    if not cap.isOpened():
        print('Error: Unable to open video source.')
        return           

    # Initialize the DeepSort tracker
    tracker = DeepSort(max_age=5)
    # Select device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load YOLO model
    yolov9_weights = FLAGS.weights
    model = DetectMultiBackend(weights=yolov9_weights, device=device, fuse=True)
    model = AutoShape(model)

    # Load the COCO class labels
    classes_path = FLAGS.classes
    with open(classes_path, "r") as f:
        class_names = f.read().strip().split("\n")

    # Create a list of random colors to represent each class
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3)) 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # Run model on each frame
        results = model(frame)        

        detect = []
        for det in results.pred[0]:
            label, confidence, bbox = det[5], det[4], det[:4]            
            x1, y1, x2, y2 = map(int, bbox)
            class_id = int(label)

            # Filter out weak detections by confidence threshold
            if confidence < FLAGS.conf:
                continue        

            detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

        tracks = tracker.update_tracks(detect, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue            
            
            track_id = track.track_id
            ltrb = track.to_ltrb()            
            x1, y1, x2, y2 = map(int, ltrb)    
            class_id = track.get_det_class()
            color = colors[class_id]
            B, G, R = map(int, color)
            text = f"{track_id} - {class_names[class_id]}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 12, y1), (B, G, R), -1)
            cv2.putText(frame, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            center_x = int((x1 + x2) / 2 )
            center_y = int((y1 + y2) / 2 )                      
        
        end_time = time.time()

        # FPS Calculation
        fps = 1 / (end_time - start_time)
        fps = float("{:.2f}".format(fps))        

        # Show FPS
        show_fps(frame, fps)

        imS = cv2.resize(frame, (1280, 720))
        cv2.imshow('YOLOv9 Object tracking', imS)        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture
    cap.release()    

if __name__ == '__main__':
    app.run(main)