"""Build the queue of channels done before whose new uploads run_channels.py fetches once the sheet is done.

    python refresh_queue.py

In this order (user, 2026-10-10): 1. the channels subscribed to (_output/subscriptions/channels.txt, "channel id
@handle" per line, read from youtube.com/feed/channels) that have videos in the mapping, or none yet but upload
drives; 2. the sheet's Processed channels and its Offline ones that are online again; 3. its Not finished ones.
Phases 1 and 2 fetch the uploads newer than the channel's newest video in the mapping, phase 3 every video not in
it. Channels whose only country is Russia are Olena's and left out. Writes run_channels.REFRESH_QUEUE ([{url,
country, name, why, phase, new_only}]) and a readable list beside the subscriptions.
"""
import collections
import json
import os
import re

import add_video
import propose_segments as ps
import run_channels as rc

SUBSCRIPTIONS = '_output/subscriptions/channels.txt'
SUMMARY = '_output/subscriptions/refresh_queue.md'
HANDLES = '_output/subscriptions/channel_ids.json'  # sheet URL -> channel id (None: no such channel now)
# a drive in the title; walks, rides and talks are something else
DRIVE_TITLE = re.compile(r'\bdriv|dash ?cam|road ?trip|\bpov\b.*\b(?:car|road)|\bhighway\b|\bby car\b', re.I)


def channel_id(url, known):
    """The channel id of a sheet URL (/channel/UC..., @handle), None when YouTube has no such channel now."""
    if url in known:
        return known[url]
    m = re.search(r'/channel/(UC[\w-]{22})', url)
    if m:
        found = ps._api('channels', part='id', id=m.group(1)).get('items')
    else:
        handle = re.search(r'/@([^/?]+)', url)
        found = ps._api('channels', part='id', forHandle='@' + ps.unquote(handle.group(1))).get('items') \
            if handle else None
    known[url] = found[0]['id'] if found else None
    return known[url]


def drives(channel, share=0.3):
    """Whether a channel's latest 30 uploads are mostly drives by their titles (share of them or more)."""
    items = ps._api('playlistItems', part='snippet', playlistId='UU' + channel[2:], maxResults=30).get('items', [])
    titles = [i['snippet']['title'] for i in items]
    return bool(titles) and sum(bool(DRIVE_TITLE.search(t)) for t in titles) >= share * len(titles)


def main():
    df = add_video.load_csv(add_video.FILE_PATH)
    mapped, countries = collections.Counter(), collections.defaultdict(collections.Counter)
    for chans, country in zip(df['channel'], df['country']):
        for c in str(chans).strip('[]').split(','):
            if c.strip().startswith('UC'):
                mapped[c.strip()] += 1
                countries[c.strip()][country] += 1
    try:
        with open(HANDLES) as f:
            known = json.load(f)
    except (OSError, ValueError):
        known = {}
    queue, left_out, seen = [], [], set()

    def add(cid, url, country, why, phase, new_only=True):
        if cid in seen:
            return
        seen.add(cid)
        # "India, USA", "Europe", "Mix": the country most of its mapped videos are in
        if rc.COUNTRY.get(country, country) not in ps.COUNTRIES:
            country = ''
        country = country or (countries[cid].most_common(1)[0][0] if countries[cid] else '')
        if country.strip().lower() == 'russia':
            left_out.append((url, "Russia only: Olena's"))
            return
        queue.append({'url': url, 'country': country, 'name': rc.channel_name(url), 'why': why, 'phase': phase,
                      'new_only': new_only})

    # the sheet's channels by id; its URLs keep the search a channel was done with (…/search?query=driving)
    sheet = {}
    for n, url, status, country in rc.sheet_rows():
        if status not in ('Processed', 'Offline', 'Not finished'):
            continue
        try:
            cid = channel_id(url, known)
        except RuntimeError as e:
            left_out.append((url, f'not looked up: {e}'))
            continue
        if cid is None:
            if status != 'Offline':
                left_out.append((url, f'row {n}: no such channel on YouTube now'))
            continue  # Offline and still gone
        sheet.setdefault(cid, (n, url, status, country))
    # in this order (user, 2026-10-10): 1. the subscriptions, 2. new uploads of the sheet's processed channels
    # (and of those online again), 3. the not finished ones, all their videos not in the mapping
    for line in open(SUBSCRIPTIONS):
        cid, handle = line.split()[:2]
        n, url, status, country = sheet.get(cid, (None, f'https://www.youtube.com/{handle}/videos', None, ''))
        if status == 'Not finished':
            continue  # last, with the not finished ones
        if mapped[cid]:
            add(cid, url, country, f'subscription: {mapped[cid]} videos in the mapping', 1)
            continue
        try:
            drive = drives(cid)
        except RuntimeError as e:
            left_out.append((url, f'subscription: not looked up ({e}): run this again'))
            continue
        if drive:
            add(cid, url, country, 'subscription: uploads drives, none in the mapping yet', 1)
        else:
            left_out.append((url, 'subscription: its latest uploads are no drives'))
    for cid, (n, url, status, country) in sheet.items():
        if status == 'Processed' and not mapped[cid]:
            left_out.append((url, f'row {n}: processed, but none of its videos is in the mapping'))
        elif status in ('Processed', 'Offline'):
            add(cid, url, country, f'sheet row {n}: ' + ('processed' if status == 'Processed' else 'online again'), 2)
    for cid, (n, url, status, country) in sheet.items():
        if status == 'Not finished':
            add(cid, url, country, f'sheet row {n}: not finished', 3, new_only=False)
    with open(HANDLES, 'w') as f:
        json.dump(known, f, indent=1)
    with open(rc.REFRESH_QUEUE + '.tmp', 'w') as f:
        json.dump(queue, f, indent=1)
    os.replace(rc.REFRESH_QUEUE + '.tmp', rc.REFRESH_QUEUE)
    with open(SUMMARY, 'w') as f:
        f.write(f'# Channels to fetch again ({len(queue)}), in this order\n\n'
                + ''.join(f"- {q['phase']}. {q['name']} ({q['country']}): {q['why']}"
                          f"{'' if q['new_only'] else ', all its videos not in the mapping'}\n" for q in queue)
                + f'\n# Left out ({len(left_out)})\n\n' + ''.join(f'- {u}: {why}\n' for u, why in left_out))
    print(f'{len(queue)} channels queued, {len(left_out)} left out: see {SUMMARY}')


if __name__ == '__main__':
    main()
