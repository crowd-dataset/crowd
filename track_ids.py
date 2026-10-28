"""Keep the track ids of every thread apart.

Ultralytics hands out track ids from one counter that all trackers of the process share. When segments are processed
in several threads (main.py with max_workers > 1), install() gives each thread its own counter, so every segment is
numbered independently (1, 2, 3, ... per segment, as with one worker).

    import track_ids
    track_ids.install()     # once, before the first model.track(...)

It is safe to call more than once.
"""
import threading

_local = threading.local()


def _next_id():
    _local.count = getattr(_local, "count", 0) + 1
    return _local.count


def _reset_id():
    _local.count = 0


def install():
    """Replace the shared counter of ultralytics' trackers by one counter per thread."""
    from ultralytics.trackers.basetrack import BaseTrack
    if getattr(BaseTrack, "_per_thread_ids", False):
        return
    BaseTrack.next_id = staticmethod(_next_id)
    BaseTrack.reset_id = staticmethod(_reset_id)
    BaseTrack._per_thread_ids = True
