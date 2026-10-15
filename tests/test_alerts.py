"""dashboards/alerts.py: when the video downloader is reported."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'dashboards'))
import alerts  # noqa: E402

NOW = 1_000_000.0


def status(ok_ago, failed_ago, queue=100):
    return dict(host='lin4080', queue=queue, state='downloading', started=NOW - 86400,
                recent=[dict(t=NOW - a) for a in ok_ago], failures=[dict(t=NOW - a) for a in failed_ago])


def test_download_alerts():
    assert alerts.download_problems(status([600], [60] * 5), NOW) == {}  # working, a few failures
    assert 'downloads:stalled' in alerts.download_problems(status([3 * 3600], []), NOW)  # nothing for 3 h
    assert alerts.download_problems(status([3 * 3600], [], queue=0), NOW) == {}  # nothing left to download
    assert 'downloads:failing' in alerts.download_problems(status([600] * 4, [60] * 12), NOW)
    assert alerts.download_problems(status([600] * 20, [60] * 12), NOW) == {}  # more downloaded than failed
    assert alerts.download_problems(dict(error='ssh failed', queue=5), NOW) == {}
