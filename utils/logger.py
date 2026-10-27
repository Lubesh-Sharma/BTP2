"""Mirror everything printed (stdout + stderr, including tracebacks) into a log file.

    from utils.logger import start_logging
    log_path = start_logging(result_dir, 'train')
    # -> <result_dir>/logs/train_YYYYmmdd_HHMMSS.log
"""
import os
import sys
import time
import json


class _Tee:
    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, data):
        self.stream.write(data)
        self.fh.write(data)
        self.fh.flush()

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def isatty(self):
        return getattr(self.stream, 'isatty', lambda: False)()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def start_logging(result_dir, tag):
    log_dir = os.path.join(result_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    fh = open(path, 'a', buffering=1)
    sys.stdout = _Tee(sys.__stdout__, fh)
    sys.stderr = _Tee(sys.__stderr__, fh)
    print(f"[log] {' '.join(sys.argv)}")
    print(f"[log] writing to {path}")
    return path


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=str)
