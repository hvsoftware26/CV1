import os
import sys

from services.config import OUTPUT_DIR, show_config
from services.train_service import run_training

sys.stdout.reconfigure(encoding="utf-8")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    show_config()

    run_training()


if __name__ == "__main__":
    main()