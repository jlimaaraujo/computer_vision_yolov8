import cv2
from utils import load_config
import os
from ultralytics import YOLO
from datetime import datetime
from person_counter import PersonCounter
from visualization import Visualizer

def main():
    # load config
    config = load_config()

    # load model
    model = YOLO(config['model_path'])

    # output paths
    output_dir = config['output_path']
    analytics_dir = config['analytics_path']

    # load video
    video_path = config['video_path']
    cap = cv2.VideoCapture(video_path)

    # Verifica se o vídeo foi carregado com sucesso
    if not cap.isOpened():
        print(f"[ERRO] Não foi possível abrir o vídeo: {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"[INFO] Resolução do vídeo: {width}x{height}, FPS: {fps}")

    if width == 0 or height == 0 or fps == 0:
        print("[ERRO] Propriedades inválidas do vídeo. Verifica o ficheiro.")
        return

    # Initialize person counter
    counter = PersonCounter(width, height, fps)

    # Initialize visualizer
    visualizer = Visualizer(analytics_dir)


    # save video
    base_name = os.path.splitext(os.path.basename(config['output_path']))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{base_name}_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    ret = True
    # read frames
    while ret:
        ret, frame = cap.read()

        if ret:
            # detect people && track them
            results = model.track(frame, persist=True, classes=[0])
            
            # Update counter with new detections
            counter.update(results, frame_count)

            # plot results (bounding boxes)
            frame_ = results[0].plot()

            # Draw zones and metrics
            frame_ = counter.draw_zones(frame_)
            frame_ = counter.draw_metrics(frame_)
            frame_ = counter.draw_trajectories(frame_)

            # write frame to output video
            out.write(frame_)

            # show frame
            cv2.imshow('frame', frame_)

            # Generate periodic reports (every 100 frames)
            if frame_count % 100 == 0 and frame_count > 0:
                print(f"Processing frame {frame_count}...")

            frame_count += 1

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    # Generate final visualizations and reports
    visualizer.plot_count_over_time(counter, filename="count_over_time_convenience_yolov8_optuna.png")
    visualizer.plot_stay_duration_histogram(counter, filename="stay_duration_histogram_yolov8_optuna.png")
    visualizer.plot_cumulative_entries_exits(counter, filename="cumulative_entries_exits_convenience_yolov8_optuna.png")
    visualizer.plot_heatmap(counter, filename="heatmap_convenience_yolov8_optuna.png")

    # release video
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
