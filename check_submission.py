import sys

from services.config import show_config
from services.submission_service import check_submission_file

sys.stdout.reconfigure(encoding="utf-8")


def main():
    show_config()

    check_submission_file()


if __name__ == "__main__":
    main()