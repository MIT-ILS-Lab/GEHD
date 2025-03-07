import os
import time
import subprocess
from datetime import datetime

from main_model.src.utils.config import load_config, parse_args


def get_dir_size(path="."):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total


def filter_filenames(filenames):
    out = []
    now = datetime.now()
    for filename in filenames:
        if not filename.startswith("offline-run"):
            continue
        time_str = filename.split("-")[2]
        time_obj = datetime.strptime(time_str, "%Y%m%d_%H%M%S")
        d = now - time_obj
        if d.total_seconds() < 7 * 24 * 60 * 60:
            out.append(filename)
    return out


if __name__ == "__main__":
    # Load the config file
    args = parse_args()
    config = load_config(args.config)

    log_dir = config["solver"]["logdir"] + "/wandb"

    while True:
        filenames = os.listdir(log_dir)
        filenames = filter_filenames(filenames)
        filenames = [os.path.join(log_dir, i) for i in filenames]

        for filename in filenames:
            size = get_dir_size(filename)
            print("%s %d" % (filename, size))
            subprocess.run(["wandb", "sync", filename])
            time.sleep(10)
