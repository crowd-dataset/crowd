"""Keep the track ids of every thread apart.

Ultralytics hands out track ids from one counter, BaseTrack._count, that all trackers of the process share, and every
new tracker (BYTETracker.__init__ -> reset_id) sets it back to 0. When segments run in threads (main.py with
max_workers > 1) a thread that starts a new segment therefore restarts the numbering of all the others, and ids that
were used before are issued again: one id ends up on several objects, hundreds of seconds apart. install() gives each
thread its own counter, so a tracker only ever sees its own numbering (1, 2, 3, ... per segment, as with one worker).

    import track_ids
    track_ids.install()     # once, before the first model.track(...)

It is safe to call more than once. Files written before this fix are repaired by tools/repair_track_ids.py.
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
