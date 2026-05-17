import os
import cv2

TRAIN_DIR = r".\dataset\train"

count = 0

for label in os.listdir(TRAIN_DIR):
    label_dir = os.path.join(TRAIN_DIR, label)

    if not os.path.isdir(label_dir):
        continue

    for file_name in os.listdir(label_dir):
        if not file_name.lower().endswith(".mp4"):
            continue

        video_path = os.path.join(label_dir, file_name)

        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        cap.release()

        print("Label:", label)
        print("Video:", file_name)
        print("Frames:", total_frames)
        print("FPS:", fps)
        print("Duration:", round(duration, 2), "seconds")
        print("-" * 40)

        count += 1

        if count >= 10:
            break

    if count >= 10:
        break