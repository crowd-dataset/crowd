# Datasheet: CROWD (City Road Observations With Dashcams)

This datasheet follows the structure of *Datasheets for Datasets* (Gebru et al., 2021). Figures are as of 30 September 2026; the current totals are in the "Dataset overview" at the top of the [README](README.md), which `analysis.py` regenerates.

## Motivation

**Purpose.** CROWD supports cross-country and cross-cultural research on how pedestrians and other road users behave in cities, in particular for the design and evaluation of automated vehicles that must interact with them. It provides long, continuous, unedited dashcam driving footage from cities worldwide, indexed by location and time of day, together with automatic YOLO object detections and tracks.

**Creators.** Md Shadab Alam, Olena Bazilinska and Pavlo Bazilinskyy. See [CITATION.cff](CITATION.cff) and the paper: Alam, M. S., Bazilinska, O., & Bazilinskyy, P. (2026). *A global dataset of continuous urban dashcam driving*. arXiv:2604.01044.

## Composition

**Instances.** The unit is a *segment*: a continuous time range within a publicly available YouTube driving video, assigned to one locality (city, town or village).

| | |
|---|---|
| Footage | 33,616 hours |
| Segments | 94,473 (median 15.5 min; 10th–90th percentile 4.0–45.7 min) |
| Videos | 74,005 unique YouTube videos, from 5,113 channels |
| Localities | 9,575 |
| Countries and territories | 238 |

**What each record contains.** `mapping.csv` has one row per locality with:

- **Location:** locality name, alternative names, state or region, country, ISO3 code, continent, latitude and longitude.
- **Context indicators:** locality population; country population, road traffic deaths per 100,000, literacy rate, Gini index, median age and average height; locality GDP (sparse) and a traffic index (see below).
- **Per video:** the YouTube ID, upload date, channel and the type of vehicle the footage is filmed from (car, bus, truck, two-wheeler, bicycle, and others).
- **Per segment:** start and end time in seconds, and time of day (day or night).

`crowd_index.jsonl`, built by `make_crowd_jsonl.py`, holds one record per segment.

**Automatic outputs.** For each processed segment, YOLO (`yolo11x`) detections with tracking IDs are stored as CSV files named `{video}_{start}_{fps}.csv` (class, bounding box, track ID, confidence, frame). These are automatic outputs, not ground truth.

**Missing data.** Empty cells mean "not available". This applies, among others, to:

- **Locality GDP:** empty for most localities.
- **Country indicators:** empty where the World Bank has no value.
- **Locality population:** 0 for a few localities.

**Traffic index:** one TomTom reading per locality of how much slower than free flow traffic was on the nearest road when it was queried (%). It is a snapshot, so it depends on the time of the query. Empty: no TomTom coverage (e.g., Ukraine, Russia, Belarus, China) or no reading yet. A 0 can mean free-flowing traffic, but older entries also stored failed requests as 0.

**Relationships.** A video can span several localities. Its segments then belong to different localities and never overlap in time: no stretch of video is counted twice (checked on 94,473 segments).

**Sampling and representativeness.** CROWD is a convenience sample of what drivers have chosen to upload, not a random sample of the world's roads. The imbalances below should be considered in any comparison between regions.

| Imbalance | Details | Figure |
|---|---|---|
| Geographic | North America 31.7% of footage, Asia 29.0%, Europe 27.0%, Oceania 5.4%, South America 3.8%, Africa 3.0%. The USA alone is 27.4%. | per-continent maps in the README, `dumbbell_footage_population_share` |
| Time of day | 14.5% of footage is at night, varying strongly by country. | `map_night_share` |
| Vehicle | 92.0% is filmed from cars, 2.7% from bicycles, 2.7% from buses and 1.6% from two-wheelers. | `bar_vehicle_type_time_of_day` |
| Upload date | 91.3% of footage was uploaded in 2020 or later. | `hist_months` |
| Channels | The 10 largest channels supply 18.6% of footage and the 100 largest 57.8%. A channel's route, camera and style can dominate a city's footage. | `line_channel_concentration`, `map_effective_channels` |
| Coverage vs wealth | Footage per person grows with GDP per person; several of the most populous countries (China, India, Nigeria, Egypt, Mexico) have less than half the footage their wealth predicts. | `bubble_footage_per_capita_gdp` |
| Coverage vs population | Coverage grows with city size but varies widely at the same size; 5,106 localities have a single segment. | `scatter_population_footage`, `bar_footage_population_band`, `scatter_indicators_footage` |

**Sensitive content.** The footage shows public streets and incidentally captures people, faces and licence plates. CROWD does not redistribute video; it distributes video IDs, time ranges, labels and automatic detections.

## Collection process

**Source.** Public YouTube videos of continuous driving (dashcam or similar), collected and annotated manually.

**Selection.** Segments are the parts of a video with continuous, usable driving footage. Footage is excluded when:

- frames are skipped;
- the camera is moving or being adjusted;
- the camera is unstable or shaking;
- the vehicle is in a parking area;
- another video is embedded in the main video.

See "Selection procedure" in the README for examples.

**Annotation.** Videos are added with `add_video.py`, a local web form. For each video, an annotator records:

- the segments and their start and end times;
- the time of day of each segment;
- the vehicle type;
- the locality.

**Context data.** Locality populations and coordinates come from GeoNames. Country indicators come from the World Bank (`update_params.py`). The traffic index comes from TomTom (`add_video.py`).

**Time frame.** Videos were uploaded between 2005 and 2026. Upload dates come from YouTube; recording dates are not known.

## Preprocessing, cleaning and labelling

- **Processed duration.** The detection pipeline processes each segment up to its end time minus 1 second. All durations in the dataset and figures use this processed duration.
- **Detection.** `main.py` downloads each video, runs YOLO with tracking on each segment, and writes the detection CSVs. Analyses count unique track IDs with confidence ≥ 0.7.
- **Data audits (September 2026).** Locality coordinates and populations were checked against GeoNames (and, for disagreements, Wikidata and OpenStreetMap):
  - About 500 populations were corrected, mostly values copied from a same-named place or another row, district or county figures on single towns, and impossible values.
  - 12 localities geocoded to a same-named place in another region were moved.
  - Wrong states and invalid upload dates were fixed, where they could be determined.
  - Zeros that meant "no data" were cleared.
- **Known remaining issues.**
  - 1,283 localities could not be matched to GeoNames, so their populations are unchecked.
  - Two videos (Mumbai and Cairo) are no longer viewable and have no valid upload date.
  - Traffic index: about 650 values that were country-level copies (often from Numbeo, above 100) were replaced with local TomTom readings in October 2026, or cleared where TomTom has no coverage. Of the remaining zeros, it is not known which are real readings and which were failed requests.

## Uses

**Intended uses.** Comparative research on road-user behaviour across cities, countries and cultures (for example pedestrian crossing behaviour and road-user mix). Other intended uses are benchmarking perception and tracking methods on diverse real-world driving footage, and studying how behaviour varies with time of day and country-level indicators.

**Uses to avoid.**

- Identifying or tracking individuals.
- Inferring sensitive attributes of people shown in the footage.
- Treating the automatic detections as ground truth.
- Comparing regions without accounting for the imbalances above, since differences in coverage, channel, vehicle or time of day can masquerade as cultural differences.

## Distribution

- **Where:** [GitHub](https://github.com/crowd-dataset/crowd) (code and mapping) and [Kaggle](https://www.kaggle.com/datasets/anonymousauthor123/pedestrian-in-youtubepyt); a permanent FAIR repository is planned.
- **Licences:** code under the MIT licence ([LICENSE](LICENSE)); dataset, annotations, metadata and derived outputs under CC BY 4.0 ([DATA_LICENSE.md](DATA_LICENSE.md)).
- **Video:** not redistributed. The underlying YouTube videos remain the property of their uploaders and are referenced by ID only. Using them is subject to YouTube's terms and to the uploaders' rights.

## Maintenance

- **Maintainers and contact:** Md Shadab Alam (md_shadab_alam@outlook.com) and [Pavlo Bazilinskyy](https://bazilinskyy.github.io) (p.bazilinskyy@tue.nl).
- **Updates:** new videos and localities are added continuously. Each change to `mapping.csv` on GitHub refreshes the README statistics and dataset figures automatically (`.github/workflows/dataset-figures.yml`). Detection-based figures are updated where the YOLO outputs are available.
- **Errata:** report problems through GitHub issues. Videos that are removed from YouTube can be dropped with `remove_offline_videos.py`.
