#!/usr/bin/env python3
"""Web tool to review the candidate cuts of cut_times.csv (step 2 of the review of cuts).

It plays the YouTube video from CLIP_BEFORE_SEC before to CLIP_AFTER_SEC after each candidate that tools/detect_cuts.py
found. The reviewer marks the candidate "yes" (a real cut), "no" (not a cut) or "unavailable" (the video cannot be
played or embedded) and can add a note; the decision and the time are saved in cut_times.csv after every candidate.
Also look for footage that is not at normal speed and for advertisements, and note them.

    python -m tools.review_cuts

then open http://127.0.0.1:5050. tools/apply_cuts.py applies the candidates marked "yes".
"""
import os
import sys
from pathlib import Path
from datetime import datetime
from math import floor, ceil
from urllib.parse import urlencode

import pandas as pd
from flask import Flask, request, redirect, url_for, render_template_string

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # top folder
os.chdir(sys.path[0])  # cut_times.csv is opened relative to the top folder


OUTPUT_CSV = Path("cut_times.csv")

CLIP_BEFORE_SEC = 2
CLIP_AFTER_SEC = 2

APP_PORT = 5050

app = Flask(__name__)


HTML = """
<!doctype html>
<html>
<head>
    <title>Review cut candidates</title>
    <meta charset="UTF-8">
    <script src="https://www.youtube.com/iframe_api"></script>

    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f7f7f7;
        }

        .layout {
            display: grid;
            grid-template-columns: 38% 60%;
            gap: 20px;
        }

        .panel {
            background: white;
            padding: 18px;
            border: 1px solid #ddd;
            border-radius: 8px;
        }

        iframe {
            width: 100%;
            height: 620px;
            border: 0;
            border-radius: 8px;
            background: black;
        }

        .big {
            font-size: 22px;
            font-weight: bold;
        }

        .row {
            margin-bottom: 10px;
            font-size: 16px;
        }

        .mono {
            font-family: monospace;
            word-break: break-all;
        }

        button {
            width: 48%;
            padding: 16px;
            font-size: 20px;
            border: 0;
            border-radius: 6px;
            cursor: pointer;
            color: white;
            margin-top: 10px;
        }

        .yes {
            background: #16833a;
        }

        .no {
            background: #b42318;
        }

        .skip {
            width: 100%;
            background: #555;
        }

        textarea {
            width: 100%;
            height: 80px;
            font-size: 16px;
            padding: 8px;
            box-sizing: border-box;
        }

        .small {
            color: #666;
            font-size: 14px;
        }

        .warning {
            color: #b42318;
            font-weight: bold;
        }

        a {
            color: #0645ad;
        }
    </style>
</head>

<body>

{% if finished %}
    <div class="panel">
        <h1>All candidates reviewed</h1>
        <p class="big">{{ reviewed_count }} / {{ total_count }} rows reviewed.</p>
        <p>Saved in: <span class="mono">{{ output_csv }}</span></p>
    </div>
{% else %}

<div class="layout">
    <div class="panel">
        <h1>Review cut candidate</h1>

        <div class="row big">
            {{ reviewed_count }} / {{ total_count }} reviewed
        </div>

        <div class="row">
            <strong>CSV file:</strong><br>
            <span class="mono">{{ csv_file }}</span>
        </div>

        <div class="row">
            <strong>Video ID:</strong><br>
            <span class="mono">{{ video_id }}</span>
        </div>

        <div class="row">
            <strong>Absolute candidate time:</strong>
            {{ absolute_time_sec }} s
        </div>

        <div class="row">
            <strong>Clip shown:</strong>
            {{ clip_start_sec }} s to {{ clip_end_sec }} s
        </div>

        <div class="row">
            <strong>Current player time:</strong>
            <span id="currentTime">0</span> s
        </div>

        <div class="row warning" id="videoErrorMessage" style="display:none;">
            Video unavailable. Skipping automatically...
        </div>

        <div class="row">
            <a href="{{ youtube_url }}" target="_blank">Open same clip in YouTube</a>
        </div>

        <form method="POST" id="reviewForm">
            <input type="hidden" name="row_index" value="{{ row_index }}">

            <label for="notes"><strong>Notes optional:</strong></label>
            <textarea id="notes" name="notes"></textarea>

            <div>
                <button class="yes" type="submit" name="is_correct" value="yes">
                    Yes correct
                </button>

                <button class="no" type="submit" name="is_correct" value="no">
                    No wrong
                </button>
            </div>

            <div>
                <button class="skip" type="submit" name="is_correct" value="unavailable">
                    Skip unavailable
                </button>
            </div>
        </form>

        <p class="small">
            Keyboard shortcuts: Y means yes, N means no, U means unavailable.
        </p>

        <p class="small">
            The same 10 second clip repeats until you answer.
        </p>

        <p class="small">
            If YouTube reports the video as unavailable, it is marked as unavailable and skipped.
        </p>

        <p class="small">
            Saved into: <span class="mono">{{ output_csv }}</span>
        </p>
    </div>

    <div class="panel">
        <iframe
            id="embeddedVideo"
            src="{{ embed_url }}"
            allow="autoplay; encrypted-media"
            allowfullscreen>
        </iframe>
    </div>
</div>

<script>
let player = null;
let submitted = false;

const CLIP_START_SEC = {{ clip_start_sec }};
const CLIP_END_SEC = {{ clip_end_sec }};

function onYouTubeIframeAPIReady() {
    player = new YT.Player("embeddedVideo", {
        events: {
            "onReady": function () {
                player.seekTo(CLIP_START_SEC, true);
                player.playVideo();
                updateCurrentTime();
                loopClipUntilAnswered();
            },
            "onStateChange": function (event) {
                if (!submitted && event.data === YT.PlayerState.ENDED) {
                    restartClip();
                }
            },
            "onError": function (event) {
                markUnavailableAndSkip(event.data);
            }
        }
    });
}

function restartClip() {
    if (!player || submitted) {
        return;
    }

    player.seekTo(CLIP_START_SEC, true);
    player.playVideo();
}

function loopClipUntilAnswered() {
    if (!player || submitted) {
        return;
    }

    const currentTime = player.getCurrentTime();

    if (currentTime >= CLIP_END_SEC - 0.20) {
        restartClip();
    }

    setTimeout(loopClipUntilAnswered, 300);
}

function updateCurrentTime() {
    if (player && player.getCurrentTime) {
        const current = Math.floor(player.getCurrentTime());
        document.getElementById("currentTime").textContent = current;
    }

    if (!submitted) {
        setTimeout(updateCurrentTime, 500);
    }
}

function markUnavailableAndSkip(errorCode) {
    if (submitted) {
        return;
    }

    submitted = true;

    const message = document.getElementById("videoErrorMessage");
    if (message) {
        message.style.display = "block";
        message.textContent = "Video unavailable or cannot be embedded. Skipping automatically. Error code: "
            + errorCode;
    }

    const notes = document.getElementById("notes");
    if (notes) {
        notes.value = "YouTube unavailable or embed error. Error code: " + errorCode;
    }

    setTimeout(function () {
        document.querySelector('button[name="is_correct"][value="unavailable"]').click();
    }, 1200);
}

document.getElementById("reviewForm").addEventListener("submit", function () {
    submitted = true;
});

document.addEventListener("keydown", function (event) {
    const tag = document.activeElement.tagName.toLowerCase();

    if (["input", "textarea", "select"].includes(tag)) {
        return;
    }

    const key = event.key.toLowerCase();

    if (key === "y") {
        submitted = true;
        document.querySelector('button[name="is_correct"][value="yes"]').click();
    }

    if (key === "n") {
        submitted = true;
        document.querySelector('button[name="is_correct"][value="no"]').click();
    }

    if (key === "u") {
        submitted = true;
        document.querySelector('button[name="is_correct"][value="unavailable"]').click();
    }
});
</script>

{% endif %}

</body>
</html>
"""


def parse_csv_file_name(csv_file):
    """
    Expected file name:
    {video_id}_{start_time}_{fps}.csv

    Example:
    -0fvEC1AZRU_16619_30.csv
    """
    stem = Path(str(csv_file)).stem
    parts = stem.rsplit("_", 2)

    if len(parts) != 3:
        return stem, "", ""

    video_id = parts[0]

    try:
        segment_start_time = float(parts[1])
    except ValueError:
        segment_start_time = ""

    try:
        fps = float(parts[2])
    except ValueError:
        fps = ""

    return video_id, segment_start_time, fps


def save_table(df):
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)


def load_table():
    if not OUTPUT_CSV.exists():
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["csv_file", "absolute_time_sec"]).to_csv(
            OUTPUT_CSV,
            index=False,
        )

    df = pd.read_csv(OUTPUT_CSV, dtype={"csv_file": str})

    required_columns = ["csv_file", "absolute_time_sec"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    extra_columns = [
        "video_id",
        "segment_start_time",
        "fps",
        "clip_start_sec",
        "clip_end_sec",
        "is_correct",
        "notes",
        "reviewed_at",
    ]

    changed = False

    for column in extra_columns:
        if column not in df.columns:
            df[column] = ""
            changed = True

    for idx, row in df.iterrows():
        csv_file = str(row["csv_file"])
        absolute_time_sec = float(row["absolute_time_sec"])

        video_id, segment_start_time, fps = parse_csv_file_name(csv_file)

        clip_start_sec = max(0, floor(absolute_time_sec - CLIP_BEFORE_SEC))
        clip_end_sec = ceil(absolute_time_sec + CLIP_AFTER_SEC)

        values = {
            "video_id": video_id,
            "segment_start_time": segment_start_time,
            "fps": fps,
            "clip_start_sec": clip_start_sec,
            "clip_end_sec": clip_end_sec,
        }

        for column, value in values.items():
            current_value = row.get(column, "")

            if pd.isna(current_value) or str(current_value).strip() == "":
                df.at[idx, column] = value
                changed = True

    if changed:
        save_table(df)

    return df


def is_reviewed(value):
    if pd.isna(value):
        return False

    value = str(value).strip().lower()

    return value in {
        "yes",
        "no",
        "unavailable",
    }


def get_next_row_index(df):
    for idx, row in df.iterrows():
        if not is_reviewed(row.get("is_correct", "")):
            return idx

    return None


def make_embed_url(video_id, clip_start_sec):
    query = urlencode({
        "start": int(clip_start_sec),
        "autoplay": 1,
        "mute": 1,
        "enablejsapi": 1,
        "rel": 0,
    })

    return f"https://www.youtube.com/embed/{video_id}?{query}"


def make_youtube_url(video_id, clip_start_sec):
    return f"https://www.youtube.com/watch?v={video_id}&t={int(clip_start_sec)}s"


@app.route("/", methods=["GET", "POST"])
def review():
    df = load_table()

    if request.method == "POST":
        row_index = int(request.form["row_index"])
        is_correct = request.form["is_correct"].strip().lower()
        notes = request.form.get("notes", "").strip()

        allowed_answers = {
            "yes",
            "no",
            "unavailable",
        }

        if is_correct not in allowed_answers:
            return "Invalid answer", 400

        df.at[row_index, "is_correct"] = is_correct
        df.at[row_index, "notes"] = notes
        df.at[row_index, "reviewed_at"] = datetime.now().isoformat(timespec="seconds")

        save_table(df)

        return redirect(url_for("review"))

    total_count = len(df)
    reviewed_count = int(df["is_correct"].apply(is_reviewed).sum())

    row_index = get_next_row_index(df)

    if row_index is None:
        return render_template_string(
            HTML,
            finished=True,
            total_count=total_count,
            reviewed_count=reviewed_count,
            output_csv=str(OUTPUT_CSV),
        )

    row = df.loc[row_index]

    csv_file = str(row["csv_file"])
    absolute_time_sec = float(row["absolute_time_sec"])
    video_id = str(row["video_id"])
    clip_start_sec = int(float(row["clip_start_sec"]))
    clip_end_sec = int(float(row["clip_end_sec"]))

    embed_url = make_embed_url(video_id, clip_start_sec)
    youtube_url = make_youtube_url(video_id, clip_start_sec)

    return render_template_string(
        HTML,
        finished=False,
        row_index=row_index,
        csv_file=csv_file,
        video_id=video_id,
        absolute_time_sec=absolute_time_sec,
        clip_start_sec=clip_start_sec,
        clip_end_sec=clip_end_sec,
        embed_url=embed_url,
        youtube_url=youtube_url,
        total_count=total_count,
        reviewed_count=reviewed_count,
        output_csv=str(OUTPUT_CSV),
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", debug=True, use_reloader=False, port=APP_PORT)  # the reloader cannot find the script after the chdir
