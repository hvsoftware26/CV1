import os
from dotenv import load_dotenv

load_dotenv()

TRAIN_DIR = os.getenv("TRAIN_DIR", "./dataset/train")
TEST_DIR = os.getenv("TEST_DIR", "./dataset/public_test")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")

NUM_FRAMES = int(os.getenv("NUM_FRAMES", 30))
IMG_SIZE = int(os.getenv("IMG_SIZE", 128))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 2))
EPOCHS = int(os.getenv("EPOCHS", 50))
LR = float(os.getenv("LR", 0.001))
SEED = int(os.getenv("SEED", 42))

MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
LABEL_MAP_PATH = os.path.join(OUTPUT_DIR, "label_map.json")
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "cv_submission.csv")


def show_config():
    print("===== CONFIG =====")
    print("TRAIN_DIR:", os.path.abspath(TRAIN_DIR))
    print("TEST_DIR:", os.path.abspath(TEST_DIR))
    print("OUTPUT_DIR:", os.path.abspath(OUTPUT_DIR))
    print("MODEL_PATH:", os.path.abspath(MODEL_PATH))
    print("LABEL_MAP_PATH:", os.path.abspath(LABEL_MAP_PATH))
    print("SUBMISSION_PATH:", os.path.abspath(SUBMISSION_PATH))
    print("NUM_FRAMES:", NUM_FRAMES)
    print("IMG_SIZE:", IMG_SIZE)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("EPOCHS:", EPOCHS)
    print("LR:", LR)
    print("SEED:", SEED)