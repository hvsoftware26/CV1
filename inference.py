import sys

from services.config import show_config
from services.inference_service import predict_test
from services.submission_service import save_submission

sys.stdout.reconfigure(encoding="utf-8")


def main():
    show_config()

    results = predict_test()

    save_submission(results)


if __name__ == "__main__":
    main()