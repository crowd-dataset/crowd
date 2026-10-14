# Dashboards of the CROWD pipeline

Pages and widgets for following the channel pipeline (`run_channels.py`, its workers and the review hub) and the dataset
as it grows. They run on the reviewing machine, next to `review_hub.py`, and need no login: serve them on a trusted
network only. The workstations themselves (CPU, GPU, disks of the worker machines) are shown by the separate
[dashboard-workstations](https://github.com/bazilinskyy/dashboard-workstations) project, which `alerts.py` reads too.

| Page | Port | File | What it shows |
|---|---|---|---|
| 🚚 Pipeline status | 8771 | `pipeline_status.py` | What each worker does (channel, progress, videos per hour, problems in its log), batches waiting for review, sheet updates waiting, the last worker sync, the refresh queue, channels by state; `/status.json` |
| 🚐 Dataset page | 8772 | `dataset_page.py`, `build_dataset_page.sh` | The README's dataset sections and figures, rebuilt nightly from the review mapping (reviews not pushed yet included) |
| 🚌 Globe widget | (8771) | `globe/widget_globe.html`, `globe/globe_data.py` | A turning globe of all localities: segments added in the last 24 h pulse red, in the last 3 days amber |
| Alerts | | `alerts.py` | Posts to a Discord channel when a worker is blocked, stalled or offline, a machine is down or a disk runs low |

The review hub (`review_hub.py`, port 8770) links to the pipeline status page.

## Pipeline status (`pipeline_status.py`)

Run from the repository's top folder:

```bash
REVIEW_HOST=0.0.0.0 python dashboards/pipeline_status.py
```

Every 2 minutes it reads the end of each worker's `runner.log` over SSH (the workers of `sync_workers.py`) and this
machine's `_output/proposals` state; the page shows that snapshot, so opening it costs the workers nothing. A worker is
*blocked* when a problem (sign-in check, quota, error) came after its last progress, *stalled* when its log has been
quiet for an hour, *idle* when it has no channel left. It also serves the globe widget (`/globe`) and its data
(`/globe.json`).

## Globe widget (`globe/widget_globe.html`, `globe/globe_data.py`)

The mapping has no dates per segment, so `globe_data.py` compares the mapping here with GitHub's `main` as it was one
and three days ago (git history of `mapping.csv`); a segment is a (video, start) pair. The globe is drawn with d3's
orthographic projection on a canvas (d3, topojson and world-atlas from jsdelivr): drag to turn it, hover a place for its
footage and what is new there. It turns only while someone can see the screen and stops its animation loop otherwise.

It is shown by the widget server of dashboard-workstations; in its `widgets.json`:

```json
{
  "pages": {"/globe": "http://<reviewing machine>:8771/globe"},
  "proxies": {"/globe.json": "http://<reviewing machine>:8771/globe.json"},
  "vars": {"__DATASET__": "http://<reviewing machine>:8772/"},
  "windows": [["globe", "globe", 330, 480]]
}
```

## Dataset page (`build_dataset_page.sh`, `dataset_page.py`)

Nightly (cron) `build_dataset_page.sh` copies the repository to `~/dataset-page/work`, runs `analysis.py` there with the
settings of the *Dataset figures* GitHub workflow (no data or video folders, empty keys) and builds
`~/dataset-page/site/index.html` with `dataset_page.py`. It never runs `analysis.py` in the repository itself, which
would rewrite `README.md` and `figures/`. Serve the site folder, e.g. `python3 -m http.server 8772` in it. The analysis
uses its own environment with the workflow's package versions (polars, pyarrow, pandas 2.2.3, plotly 5.24.1, kaleido
0.2.1, pillow, tqdm, numpy, pycountry, requests, markdown).

```text
0 4 * * * $HOME/repos/crowd/dashboards/build_dataset_page.sh >> $HOME/Library/Logs/dataset_page.log 2>&1
```

## Alerts (`alerts.py`)

Every 10 minutes (cron) it checks the workstations dashboard (port 8090) and the pipeline status page (8771) on this
machine and posts to Discord: a problem when it starts, again every 6 hours while it lasts, and once when it clears.
The webhook URL is read from `~/.config/crowd-alerts/discord_webhook` (mode 600, never in a repository); links in the
posts use `CROWD_HOST`. `python3 alerts.py --dry-run` prints what it would post.

```text
*/10 * * * * cd $HOME/repos/crowd/dashboards && CROWD_HOST=<address> /usr/bin/python3 alerts.py >> $HOME/Library/Logs/crowd-alerts.log 2>&1
```
