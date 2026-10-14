"""tools/download_videos.py: which drive gets the next video, and which videos count as present."""
import collections
import os

from tools import download_videos as dv

Usage = collections.namedtuple('Usage', 'total used free')


def test_target_fills_drives_in_order(tmp_path, monkeypatch):
    a, b = tmp_path / 'a' / 'videos', tmp_path / 'b' / 'videos'
    free = {str(tmp_path / 'a'): 104e9, str(tmp_path / 'b'): 900e9}  # a: just under 100 GB reserve + 5 GB
    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    monkeypatch.setattr(dv.shutil, 'disk_usage', lambda p: Usage(0, 0, free[p]))
    assert dv.target([str(a), str(b)], 100) == str(b)  # a is full down to its reserve: the next drive
    free[str(tmp_path / 'a')] = 200e9
    assert dv.target([str(a), str(b)], 100) == str(a)  # room on a again: a first
    free[str(tmp_path / 'b')] = 1e9
    free[str(tmp_path / 'a')] = 1e9
    assert dv.target([str(a), str(b)], 100) is None


def test_unmounted_media_drive_is_skipped():
    assert dv.mounted('/home/someone/videos')
    assert not dv.mounted('/media/nobody/not-a-drive-here/videos')


def test_present(tmp_path):
    (tmp_path / 'x.mp4').write_bytes(b'')
    (tmp_path / 'y.part').write_bytes(b'')
    assert dv.present([str(tmp_path), os.path.join(str(tmp_path), 'missing')]) == {'x'}
