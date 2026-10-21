"""Rewriting mapping.csv safely from the tools: hold mapping.csv.lock (the fcntl lock add_video.py, helper_script.py,
the review pages and tools/update_traffic_index.py take) while reading, changing and writing it, and swap the new
file in whole, so a run stopped part way leaves the old mapping intact."""
import contextlib
import fcntl
import os


@contextlib.contextmanager
def locked(path):
    """Hold path + '.lock' exclusively for the block."""
    with open(path + '.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def replace(path, write):
    """Call write(tmp) for a file beside path, then swap it in for path (the temporary file is removed on errors)."""
    tmp = f'{path}.{os.getpid()}.tmp'
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
