import cv2
from utils import load_config
import os
from ultralytics import YOLO
from datetime import datetime

def main():
    
    # load config
    config = load_config()
    
    # load model
    model = YOLO(config['model_path'])
    
    # load video
    video_path = config['video_path']
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # save video
    output_dir = os.path.dirname(config['output_path'])
    base_name = os.path.splitext(os.path.basename(config['output_path']))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{base_name}_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    ret = True
    # read frames
    while ret:
        ret, frame = cap.read()
        
        if ret:
            # detect people && track them
            results = model.track(frame, persist=True, classes=[0])
            
            # plot results
            frame_ = results[0].plot()
            
            # write frame to output video
            out.write(frame_)
            
            # show frame
            cv2.imshow('frame', frame_)
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    
    # release video
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()