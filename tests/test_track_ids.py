import argparse
import threading

import pytest

pytest.importorskip("ultralytics")


def trackers():
    from ultralytics.trackers.byte_tracker import BYTETracker
    cfg = argparse.Namespace(track_high_thresh=0.7, track_low_thresh=0.3, new_track_thresh=0.7, track_buffer=60,
                             match_thresh=0.8, fuse_score=True)
    return lambda: BYTETracker(cfg, 30)


def issue(n):
    from ultralytics.trackers.basetrack import BaseTrack
    return [BaseTrack.next_id() for _ in range(n)]


def test_a_new_tracker_in_another_thread_does_not_restart_the_ids_of_a_running_one():
    import track_ids
    track_ids.install()
    track_ids.install()  # safe to repeat
    make = trackers()
    make()
    first = issue(3)
    t = threading.Thread(target=lambda: (make(), issue(5)))  # another segment starts in another thread
    t.start()
    t.join()
    assert first + issue(2) == [1, 2, 3, 4, 5]


def test_every_thread_numbers_from_one():
    import track_ids
    track_ids.install()
    out = {}

    def work(name):
        trackers()()
        out[name] = issue(4)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert all(v == [1, 2, 3, 4] for v in out.values())
