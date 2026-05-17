import os
import pandas as pd

from services.config import TEST_DIR, SUBMISSION_PATH


def save_submission(results):
    df = pd.DataFrame(results)

    df.to_csv(
        SUBMISSION_PATH,
        index=False,
        encoding="utf-8"
    )

    print("Saved submission:", os.path.abspath(SUBMISSION_PATH))
    print(df.head())

    return df


def check_submission_file():
    df = pd.read_csv(SUBMISSION_PATH)

    print(df.head())
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    assert list(df.columns) == ["video_name", "label"], "Sai tên cột"
    assert df["video_name"].isnull().sum() == 0, "Có video_name bị thiếu"
    assert df["label"].isnull().sum() == 0, "Có label bị thiếu"
    assert df["video_name"].duplicated().sum() == 0, "Có video bị trùng"

    test_videos = sorted([
        file_name for file_name in os.listdir(TEST_DIR)
        if file_name.lower().endswith(".mp4")
    ])

    submission_videos = sorted(df["video_name"].tolist())

    assert test_videos == submission_videos, "Submission không khớp danh sách video test"

    print("Submission hợp lệ.")