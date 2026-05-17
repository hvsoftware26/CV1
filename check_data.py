import os
import sys

from services.config import TRAIN_DIR, TEST_DIR, show_config

sys.stdout.reconfigure(encoding="utf-8")


def main():
    show_config()

    print("\n===== TRAIN =====")

    if not os.path.exists(TRAIN_DIR):
        raise FileNotFoundError(f"Không tìm thấy TRAIN_DIR: {os.path.abspath(TRAIN_DIR)}")

    labels = sorted([
        name for name in os.listdir(TRAIN_DIR)
        if os.path.isdir(os.path.join(TRAIN_DIR, name))
    ])

    total_train = 0

    for label in labels:
        label_path = os.path.join(TRAIN_DIR, label)

        videos = [
            file_name for file_name in os.listdir(label_path)
            if file_name.lower().endswith(".mp4")
        ]

        total_train += len(videos)

        print(label, ":", len(videos))

    print("Số class:", len(labels))
    print("Tổng video train:", total_train)

    print("\n===== TEST =====")

    if os.path.exists(TEST_DIR):
        test_videos = [
            file_name for file_name in os.listdir(TEST_DIR)
            if file_name.lower().endswith(".mp4")
        ]

        print("Tổng video test:", len(test_videos))
        print("Ví dụ:", test_videos[:5])
    else:
        print("Chưa thấy thư mục test:", os.path.abspath(TEST_DIR))


if __name__ == "__main__":
    main()